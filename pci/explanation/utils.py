import logging
from collections.abc import Callable
from typing import Any

import pyro
import pyro.distributions as dist
import torch

from pci.explanation.excised import sample_alternatives

logger = logging.getLogger(__name__)


def get_alternative_sample(
    structured_model: Callable,
    n_size: int,
    active_antecedents: list[str],
    factual_dictionary: dict | None = None,
    kwargs_iterable: list[dict[str, Any]] | None = None,
    distribution_type_selectors: dict["str", "str"] | None = None,
    equality_epsilon: float = 0.05,
    batch_dim: int = -3,
) -> dict[str, Any | None]:
    """
    Generate alternative samples from a structured Pyro model, either with or without
    conditioning on factual observations.

    This function uses Pyro's tracing mechanism to sample from the given model.
    If no ``factual_dictionary`` is provided, it simply samples values for the
    ``active_antecedents``. If a ``factual_dictionary`` is given, it ensures sampled
    values differ from factual values within a specified tolerance (``equality_epsilon``).

    :param structured_model:
        A Pyro model that accepts ``kwargs_iterable`` as input.
    :param n_size:
        The number of samples to generate.
    :param active_antecedents:
        Keys of antecedent variables to extract from the trace.
    :param factual_dictionary:
        Dictionary mapping antecedent names to factual tensors.
        If provided, the function avoids returning samples too close to factual values.
    :param kwargs_iterable:
        Iterable of kwargs to be passed into ``structured_model``.
        If ``None``, default values are constructed. This assumes the model has been
        built using the ``ModelComposer`` class.
    :param distribution_type_selectors:
        Optional mapping from antecedent names to distribution types. If not provided,
        the function attempts to infer types from name suffixes. Supported types are
        ``normal``, ``categorical``, and ``truncated_normal``.
    :param equality_epsilon:
        Absolute tolerance for checking closeness of sampled vs. factual values.
    :param batch_dim:
        Batch dimension along which to concatenate samples.

    :returns:
        Dictionary mapping each active antecedent to its sampled values.
        If ``factual_dictionary`` is provided, samples are guaranteed to differ
        from factuals within the specified tolerance.
    """
    if kwargs_iterable is None:
        kwargs_iterable = [
            {"observations_dict": None, "n_size": n_size},
            dict(),
            dict(),
        ]

    if factual_dictionary is None:
        with pyro.poutine.trace() as tr:
            with torch.no_grad():
                structured_model(kwargs_iterable=kwargs_iterable)

        sampled_values = {
            name: site["value"]
            for name, site in tr.trace.nodes.items()
            if name in active_antecedents
        }

    else:
        factual_to_avoid = {
            key: val
            for key, val in factual_dictionary.items()
            if key in active_antecedents
        }

        if distribution_type_selectors is None:
            # if not passed, try to infer from name suffixes
            def get_distribution_type_selector(dict):
                suffix_to_type = {
                    "_reind": "categorical",
                    "_logp": "normal",
                    "_epsilon": "normal",
                    "_std": "normal",
                    "_minmax": "truncated_normal",
                }

                distribution_type_selector = {
                    name: dist_type
                    for name in active_antecedents
                    for suffix, dist_type in suffix_to_type.items()
                    if name.endswith(suffix)
                }

                return distribution_type_selector

            distribution_type_selectors = get_distribution_type_selector(
                factual_to_avoid
            )

        with pyro.poutine.trace() as tr:
            with sample_alternatives(
                factuals=factual_to_avoid,
                distribution_type_selectors=distribution_type_selectors,
                epsilon=torch.tensor(equality_epsilon),
            ):
                with torch.no_grad():
                    structured_model(kwargs_iterable=kwargs_iterable)

        sampled_values = {
            name: site["value"]
            for name, site in tr.trace.nodes.items()
            if name in active_antecedents
        }

        # assert all sampled values differ from factuals within tolerance with
        # some correction for minmax features and boundary points due to model
        # clamping effects
        for key in factual_to_avoid.keys():
            sampled = sampled_values[key]
            assert isinstance(sampled, torch.Tensor)

            factual = factual_to_avoid[key]

            # boundary factuals for minmax features
            if key.endswith("_minmax"):
                boundary_mask = torch.isclose(
                    factual, torch.tensor(0.0)
                ) | torch.isclose(factual, torch.tensor(1.0))
            else:
                boundary_mask = torch.zeros_like(factual, dtype=torch.bool)

            factor = 40.0 if key.endswith("_minmax") else 10.0
            too_close = torch.isclose(sampled, factual, atol=equality_epsilon / factor)

            # only enforce away-from-factual for non-boundary points
            if not torch.all(~too_close | boundary_mask):
                logger.warn(f"Sampled values for {key} are too close to factuals.")

            # moreover, frequency check without constraints
            not_close = ~torch.isclose(
                sampled_values[key],  # type: ignore
                factual_to_avoid[key],
                atol=equality_epsilon / factor,
            )

            fraction_ok = not_close.float().mean()

            if fraction_ok < 0.95:
                logger.warn(
                    f"Sampled values for {key} are too close to factuals "
                    f"for {(1 - fraction_ok):.3%} of datapoints"
                )

    return sampled_values


def broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Given a batch mask of potentially fewer dimensions,
    locates the batch dimension in the tensor and reshapes the mask appropriately.
    Assumes the target tensor has only one dimension of the same size as the batch size.
    """
    # Find the dimension in target that matches mask length
    matches = [i for i, s in enumerate(target.shape) if s == mask.shape[0]]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one matching dimension, found {matches}, num rows can't be num samples."
        )
    match_dim = matches[0]

    # Reshape mask to match that dimension, with 1's everywhere else
    view_shape = [1] * target.ndim
    view_shape[match_dim] = mask.shape[0]

    return mask.view(*view_shape).expand(target.shape)


def sample_k_indices(
    k_min=1,
    k_max=4,
    n=24,
    sample_size=5,
    invert_selection=False,
    weighting="cardinality",
):
    """
    Sample multiple sets of indices with random sizes. The set size $k$ is sampled
    from ``[k_min, k_max]`` according to ``weighting``; then $k$ (or $n-k$ if
    ``invert_selection=True``) of $n$ indices are uniformly selected without
    replacement.

    :param k_min:
        Minimum number of indices to select.
    :param k_max:
        Maximum number of indices to select.
    :param n:
        Total number of available indices.
    :param sample_size:
        Number of samples to draw (batch size).
    :param invert_selection:
        If True, select ``n - k`` indices instead of ``k``.
    :param weighting:
        How to weight the cardinality $k$ before the uniform within-cardinality
        draw. One of:

        * ``"cardinality"`` (default) — uniform on $k$, then uniform over
          size-$k$ subsets. Marginal distribution over subsets is
          $\\propto 1/\\binom{n}{k}$.
        * ``"subsets"`` — $k$ weighted by $\\binom{n}{k}$, then uniform
          over size-$k$ subsets. Marginal is uniform over all subsets in the
          allowed cardinality range.

    :returns:
        List of ``sample_size`` boolean tensors of length ``n``,
        where True indicates a selected index.

    .. note::
        - The number of True entries in each tensor is random, drawn from ``k_min..k_max``.
        - If ``invert_selection=True``, the number of selected indices is ``n - k``.
    """
    import math

    if weighting == "cardinality":
        probs = torch.ones(k_max - k_min + 1)
    elif weighting == "subsets":
        probs = torch.tensor([float(math.comb(n, k)) for k in range(k_min, k_max + 1)])
    else:
        raise ValueError(
            f"Unknown weighting={weighting!r}; expected 'cardinality' or 'subsets'."
        )

    with pyro.plate("batch", sample_size):
        k = pyro.sample("k", dist.Categorical(probs=probs)) + k_min

    indices = []

    for size in k.tolist():
        if invert_selection:
            size = n - size

        indicators = torch.zeros(n, dtype=torch.bool)
        if size == 0:
            indices.append(indicators)
            continue

        idx = torch.multinomial(
            input=torch.ones(n),  # weights
            num_samples=size,
            replacement=False,
        )
        indicators[idx] = True

        indices.append(indicators)

    return indices


def drop_nans(tensor: torch.Tensor) -> torch.Tensor:
    """Drop all nan values from a tensor."""
    return tensor[~torch.isnan(tensor)]
