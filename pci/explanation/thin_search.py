import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import compress
from typing import cast

import pyro
import torch
from chirho.interventional.handlers import do
from pyro.poutine.trace_struct import Trace
from tqdm import tqdm

from pci.explanation.searchable import SearchableModel
from pci.explanation.utils import (
    get_alternative_sample,
    sample_k_indices,
)
from pci.tools.data_processing import get_variable_from_structured_data


@dataclass
class ThinSearchResult:
    sufficiency: Mapping[str, torch.Tensor]  #: sufficiency tensors for each site
    necessity: Mapping[str, torch.Tensor]  #: necessity tensors for each site

    #: boolean tensors indicating active antecedents
    antecedent_indicators: Mapping[str, torch.Tensor]

    #: boolean tensors indicating active witnesses
    witness_indicators: Mapping[str, torch.Tensor]

    sufficiency_trace: Trace | None = None
    necessity_trace: Trace | None = None

    # Optional: make it dict-like so old code won’t break immediately
    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self):
        return (
            "sufficiency",
            "necessity",
            "antecedent_indicators",
            "witness_indicators",
            "sufficiency_trace",
            "necessity_trace",
        )


class ThinSearchSampler(pyro.nn.PyroModule):
    """
    A sampler for conducting thin search over interventional regimes in a structured
    probabilistic model.

    This class supports sampling sufficiency and necessity outcomes under
    randomized interventions on antecedent and witness variables, while
    sharing noise between intervened worlds,
    and not conditioning on quasi-deterministic sites, whose values
    are supposed to be fixed by upstream variables.
    """

    def __init__(
        self,
        structured_model: SearchableModel,
        conditioned_alternatives: bool = False,
        factual_exclusion: bool = True,
        max_antecedents: int = 4,
        max_witnesses_dropped: int = 4,
        equality_epsilon: float = 0.05,
        aliases: dict[str, str] | None = None,
        gamma_weighting: str = "cardinality",
    ):
        """Initialize the ThinSearchSampler with the given structured model and parameters.

        :param structured_model: The structured model to be used for sampling,
                it needs to have attributes specified in the corresponding `
                `SearchableABC` specification.
        :param conditioned_alternatives: If `True`, alternative interventions
                are sampled from a model conditioned on witness interventions;
                this means the sampling distribution parameters are constrained
                by the **upstream** witness states; defaults to False.
        :param factual_exclusion: if true, alternative interventions will
                exclude values within `equality_epsilon` from the factual values,
                defaults to True.
        :param max_antecedents: the maximal set size for active antecedent search,
                defaults to 4
        :param max_witnesses_dropped: the maximal set size of dropped witnesses,
                defaults to 4
        :param equality_epsilon: tolerance for factual exclusion of alternative values,
                defaults to 0.05
        :param aliases: dictionary of the form {'model_key': 'data_key'}, where `data_key`
                is expected to exist in `factual_feautres`; if those differ, defaults to None
        :param gamma_weighting: distribution over the active-cardinality $k$ used by
                :py:func:`pci.explanation.utils.sample_k_indices` when drawing antecedent
                and witness indicator masks. ``"cardinality"`` (default) gives equal
                weight to each $k$; ``"subsets"`` weights $k$ by $\\binom{n}{k}$ so the
                marginal distribution over subsets is uniform across the allowed range.
        :raises ValueError: if any suspect is not in sites_of_interest
        """
        super().__init__()
        for attr in [
            "sites_of_interest",
            "suspects",
            "hard_evidence_sites",
            "shared_noise_sites",
            "deterministic_sites",
        ]:
            setattr(self, attr, getattr(structured_model, attr))

        self.outcome_variable = structured_model.outcome_variable

        self.return_sites = list(
            set(
                self.sites_of_interest
                + [
                    self.outcome_variable,
                ]
            )
        )

        self.factual_exclusion = factual_exclusion
        self.equality_epsilon = equality_epsilon

        self.aliases = aliases

        self.conditioned_alternatives = conditioned_alternatives
        self.structured_model = structured_model
        self.gamma_weighting = gamma_weighting

        self.max_antecedents = max_antecedents
        self.max_witnesses_dropped = max_witnesses_dropped

        # ensure basics of search logic
        self.suspects = [
            s
            for s in structured_model.suspects
            if s not in structured_model.deterministic_sites
        ]
        if getattr(structured_model, "witness_sites", None) is None:
            self.witnesses = [
                s
                for s in structured_model.sites_of_interest
                if s not in structured_model.deterministic_sites + [self.outcome_variable]
            ]
        else:
            # When witness_sites is explicitly provided, respect that choice:
            # deterministic sites can be witnesses (pinning them at factual is
            # well-defined via do-interventions). Only the outcome variable is
            # always excluded.
            self.witnesses = [
                s
                for s in structured_model.witness_sites
                if s != self.outcome_variable
            ]

        for suspect in structured_model.suspects:
            if suspect not in structured_model.sites_of_interest:
                raise ValueError(
                    f"Suspect {suspect} is not in sites of interest {structured_model.sites_of_interest}"
                )

    def sample(
        self,
        factual_features: dict[str, dict[str, torch.Tensor]],
        num_samples: int = 3,
        return_traces: bool = False,
        max_resample_attempts: int = 10,
    ) -> ThinSearchResult:
        """
        Perform sampling of the structured model under random interventions.

        :param factual_features: Dictionary of factual features separated by type ('continuous' or 'categorical').
        :param num_samples: Number of interventional samples to generate.
        :param return_traces: If True, returns the Pyro trace objects along with sampled tensors.
        :param max_resample_attempts: Maximum number of attempts to resample when encountering fully excised intervals.
        """
        factual_dictionary = {
            key: get_variable_from_structured_data(
                factual_features, key, aliases=self.aliases
            )
            for key in self.sites_of_interest
        }

        batch_size = next(iter(factual_dictionary.values())).shape[0]

        # assert all factuals have the same batch size
        for key, value in factual_dictionary.items():
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"All factual features must have the same batch size; got {batch_size} and {value.shape[0]} for {key}"
                )

        sufficiency_world_collection = {}
        necessity_world_collection = {}

        antecedent_indicators_collection = []
        witness_indicators_collection = []

        global_antecedent_indicators = sample_k_indices(
            k_min=1,
            k_max=min(self.max_antecedents, len(self.suspects)),
            n=len(self.suspects),
            sample_size=num_samples,
            weighting=self.gamma_weighting,
        )

        global_witness_indicators = sample_k_indices(
            k_min=0,
            k_max=min(len(self.witnesses), self.max_witnesses_dropped),
            n=len(self.witnesses),
            sample_size=num_samples,
            invert_selection=True,
            weighting=self.gamma_weighting,
        )

        for sample_index in tqdm(range(num_samples)):
            antecedent_indicators = global_antecedent_indicators[sample_index]
            witness_indicators = global_witness_indicators[sample_index]

            assert antecedent_indicators.sum() > 0, (
                "At least one antecedent must be active for each sample."
            )

            active_antecedents = list(
                compress(self.suspects, antecedent_indicators.tolist())
            )
            prelim_active_witnesses = list(
                compress(self.witnesses, witness_indicators.tolist())
            )

            deduped_active_witnesses = [
                w for w in prelim_active_witnesses if w not in active_antecedents
            ]
            deduped_hard_evidence = [
                h for h in self.hard_evidence_sites if h not in active_antecedents
            ]

            active_witnesses = deduped_hard_evidence + deduped_active_witnesses

            witness_indicators = torch.tensor(
                [w in deduped_active_witnesses for w in self.witnesses],
                dtype=torch.bool,
            )

            antecedent_indicators_collection.append(antecedent_indicators)
            witness_indicators_collection.append(witness_indicators)

            antecedent_sufficiency_interventions = {
                key: factual_dictionary[key] for key in active_antecedents
            }

            witness_interventions = {
                key: factual_dictionary[key] for key in active_witnesses
            }

            if self.conditioned_alternatives:
                witness_categorical = {
                    key: value
                    for key, value in witness_interventions.items()
                    if key in factual_features["categorical"].keys()
                }
                witness_continuous = {
                    key: value
                    for key, value in witness_interventions.items()
                    if key in factual_features["continuous"].keys()
                }

                witness_kwargs_iterable = [
                    {
                        "observations_dict": {
                            "continuous": witness_continuous,
                            "categorical": witness_categorical,
                        },
                        "n_size": batch_size,
                    },
                    {},
                    {},
                ]

                _kwargs_iterable = witness_kwargs_iterable
            else:
                _kwargs_iterable = None

            if self.factual_exclusion:
                _factual_dictionary = factual_dictionary
            else:
                _factual_dictionary = None

            attempt = 0

            while attempt < max_resample_attempts:
                try:
                    alternative_interventions = get_alternative_sample(
                        self.structured_model,
                        active_antecedents=active_antecedents,
                        n_size=batch_size,
                        factual_dictionary=_factual_dictionary,
                        kwargs_iterable=_kwargs_iterable,
                        equality_epsilon=self.equality_epsilon,
                    )
                    break
                except ValueError as e:
                    # Only retry if it's the excised-mass error
                    if (
                        "Total probability mass in excised intervals" in str(e)
                        or "have all logits excised" in str(e)
                        or "found invalid values" in str(e)
                        or "satisfy the constraint" in str(e)
                    ):
                        attempt += 1
                        if attempt == max_resample_attempts:
                            raise ValueError(
                                f"Failed to sample after {max_resample_attempts} attempts due to fully excised intervals."
                            ) from e

                        warnings.warn(
                            f"Sample attempt {attempt} failed due to fully excised intervals; retrying..."
                        )
                        continue
                    else:
                        # Re-raise other ValueErrors
                        raise

            with torch.no_grad():
                with do(
                    actions=antecedent_sufficiency_interventions | witness_interventions
                ):
                    with pyro.poutine.trace() as tr_sufficiency_world:
                        self.structured_model(
                            kwargs_iterable=[
                                {"observations_dict": None, "n_size": batch_size},
                                {},
                                {},
                            ],
                        )

                shared_noise = {
                    key: tr_sufficiency_world.trace.nodes[key]["value"]
                    for key in self.shared_noise_sites
                }

                with do(
                    actions=alternative_interventions
                    | witness_interventions
                    | shared_noise
                ):
                    with pyro.poutine.trace() as tr_necessity_world:
                        self.structured_model(
                            kwargs_iterable=[
                                {"observations_dict": None, "n_size": batch_size},
                                {},
                                {},
                            ],
                        )

            # the unsqueeze here is for later stacking
            sufficiency_nodes = {
                key: cast(
                    torch.Tensor, tr_sufficiency_world.trace.nodes[key]["value"]
                ).unsqueeze(-4)
                for key in self.return_sites
            }

            necessity_nodes = {
                key: cast(
                    torch.Tensor, tr_necessity_world.trace.nodes[key]["value"]
                ).unsqueeze(-4)
                for key in self.return_sites
            }

            for key, value in sufficiency_nodes.items():
                if key not in sufficiency_world_collection:
                    sufficiency_world_collection[key] = value
                else:
                    sufficiency_world_collection[key] = torch.cat(
                        [sufficiency_world_collection[key], value], dim=0
                    )

            for key, value in necessity_nodes.items():
                if key not in necessity_world_collection:
                    necessity_world_collection[key] = value
                else:
                    necessity_world_collection[key] = torch.cat(
                        [necessity_world_collection[key], value], dim=0
                    )

        antecedent_indicators_dictionary: dict[str, torch.Tensor] = {}
        stacked_antecedents_indicators = torch.stack(antecedent_indicators_collection)

        witness_indicators_dictionary: dict[str, torch.Tensor] = {}
        stacked_witness_indicators = torch.stack(witness_indicators_collection)

        for i, key in enumerate(self.suspects):
            antecedent_indicators_dictionary[key] = stacked_antecedents_indicators[:, i]

        for i, key in enumerate(self.witnesses):
            witness_indicators_dictionary[key] = stacked_witness_indicators[:, i]

        return ThinSearchResult(
            sufficiency=sufficiency_world_collection,
            necessity=necessity_world_collection,
            antecedent_indicators=antecedent_indicators_dictionary,
            witness_indicators=witness_indicators_dictionary,
            sufficiency_trace=tr_sufficiency_world.trace if return_traces else None,
            necessity_trace=tr_necessity_world.trace if return_traces else None,
        )
