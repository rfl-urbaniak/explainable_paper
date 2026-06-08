from abc import ABC, abstractmethod
from collections.abc import Callable


class SearchableABC(ABC):
    """Abstract interface a model must satisfy to be driven by a causal search.

    Defines the contract of site-name accessors that a search procedure relies
    on to know which variables are fixed as context, which are deterministic,
    which are candidate causes, and so on. Concrete subclasses expose an
    underlying structured (Pyro) model and implement each abstract property.
    """

    @property
    @abstractmethod
    def structured_model(self):
        """The underlying structured model that subclasses must attach.

        :returns: The callable structured (Pyro) model wrapped by this object.
        """

    def __call__(self, *args, **kwargs):
        """Invoke the wrapped structured model, forwarding all arguments.

        :param args: Positional arguments passed through to the structured model.
        :param kwargs: Keyword arguments passed through to the structured model.
        :returns: Whatever the structured model returns when called.
        """
        return self.structured_model(*args, **kwargs)

    @property
    @abstractmethod
    def hard_evidence_sites(self) -> list[str]:
        """Sites that are always included as elements of the context that are
        kept fixed

        """
        pass

    @property
    @abstractmethod
    def deterministic_sites(self) -> list[str]:
        """Quasi-deterministic variables whose values are conceptually
        unambiguously fixed by upstream variables, and should not be considered
        causal suspects.

        """
        pass

    @property
    @abstractmethod
    def sites_of_interest(self) -> list[str]:
        """Sites in the model to be preserved in the sample"""
        pass

    @property
    @abstractmethod
    def suspects(self) -> list[str]:
        """Variables considered as potential causes"""
        pass

    @property
    @abstractmethod
    def shared_noise_sites(self) -> list[str]:
        """Noise sites whose values are shared across possible worlds. Using
        these requires the model to have employed a reparametrization strategy
        that exposes such sites in the pyro model. If left empty, no noise is
        shared in unconditioned intervened model between intervened worlds

        """
        pass

    @property
    @abstractmethod
    def outcome_variable(self) -> str:
        """The site name of the outcome whose causes are being explained."""
        pass


class SearchableModel(SearchableABC):
    """A helper class for converting a Pyro model into one compatible with
    :py:class:`pci.explanation.thin_search.ThinSearchSampler`. Other than
    indicating the appropriate site names (see the documentation for
    :py:class:`SearchableABC`), the model needs to accept a named argument,
    `iterable_kwargs`, whose 0-th element needs to comprise `observations_dict`
    (potentially set to `None`, otherwise typically a dict with two keys,
    `continuous` and `categorical`, with corresponding values being dictionaries
    with variable names as keys as tensors as values) and `n_size`, informing
    the model of the batch size.

    """

    def __init__(
        self,
        structured_model: Callable,
        hard_evidence_sites: list[str] = [],
        deterministic_sites: list[str] = [],
        sites_of_interest: list[str] = [],
        suspects: list[str] = [],
        shared_noise_sites: list[str] = [],
        outcome_variable: str = "y",
        witness_sites: list[str] | None = None,
    ):
        """Wrap a structured model together with its search-relevant site names.

        :param structured_model: The structured (Pyro) model to make searchable.
        :param hard_evidence_sites: Sites kept fixed as context across worlds.
        :param deterministic_sites: Quasi-deterministic sites that are fixed by
            upstream variables and excluded from being causal suspects.
        :param sites_of_interest: Sites to preserve in the sample.
        :param suspects: Variables considered as potential causes; each must
            also appear in ``sites_of_interest``.
        :param shared_noise_sites: Noise sites whose values are shared across
            possible worlds.
        :param outcome_variable: Site name of the outcome to be explained.
        :param witness_sites: Sites acting as witnesses, or ``None`` for none.
        :raises ValueError: If a suspect is not among the sites of interest, or
            if no search-relevant sites are provided at all.
        """
        for suspect in suspects:
            if suspect not in sites_of_interest:
                raise ValueError(
                    f"Suspect {suspect} is not in sites of interest {sites_of_interest}"
                )

        self._structured_model = structured_model

        self._hard_evidence_sites = hard_evidence_sites
        self._deterministic_sites = deterministic_sites
        self._sites_of_interest = sites_of_interest
        self._witness_sites = witness_sites
        self._suspects = suspects
        self._shared_noise_sites = shared_noise_sites

        if all(
            len(lst) == 0
            for lst in [
                self._hard_evidence_sites,
                self._deterministic_sites,
                self._sites_of_interest,
                self._suspects,
                self._shared_noise_sites,
            ]
        ):
            raise ValueError(
                "You have not provided any search-relevant sites, your search would be fruitless."
            )

        self._outcome_variable = outcome_variable

    @property
    def structured_model(self):
        """The wrapped structured (Pyro) model."""
        return self._structured_model

    @property
    def hard_evidence_sites(self):
        """Sites kept fixed as context across possible worlds."""
        return self._hard_evidence_sites

    @property
    def deterministic_sites(self):
        """Quasi-deterministic sites excluded from being causal suspects."""
        return self._deterministic_sites

    @property
    def witness_sites(self):
        """Sites acting as witnesses, or ``None`` if none were provided."""
        return self._witness_sites

    @property
    def sites_of_interest(self):
        """Sites preserved in the sample."""
        return self._sites_of_interest

    @property
    def suspects(self):
        """Variables considered as potential causes."""
        return self._suspects

    @property
    def shared_noise_sites(self):
        """Noise sites whose values are shared across possible worlds."""
        return self._shared_noise_sites

    @property
    def outcome_variable(self):
        """Site name of the outcome to be explained."""
        return self._outcome_variable
