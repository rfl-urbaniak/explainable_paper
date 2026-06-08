from collections.abc import Callable, Hashable, Mapping
from contextlib import contextmanager
from typing import TypeVar

import pyro
import pyro.distributions as dist
import torch
from chirho.observational.ops import ExcisedCategorical, ExcisedNormal

T = TypeVar("T")


class Excisions(pyro.poutine.messenger.Messenger):
    """
    Pyro effect handler that excises (removes) intervals from sample sites.

    At each registered sample site, the original distribution is replaced by an
    excised counterpart (``ExcisedNormal`` or ``ExcisedCategorical``) whose
    support has the specified intervals removed, and any existing observation is
    overridden with a fresh sample from that excised distribution. This makes the
    handler suitable for sampling counterfactual alternatives but incompatible
    with inference.
    """

    def __init__(
        self,
        intervals: Mapping[Hashable, list[tuple[torch.Tensor, torch.Tensor]]],
        distribution_type_selectors: dict["str", "str"] | None = None,
    ):
        """
        :param intervals: A mapping from names of sample sites to a list of intervals.
        :param distribution_type_selectors: Optional mapping from names of sample sites to distribution type indicators.
            Supported indicators are "normal" and "categorical". Needed, because in more complex models one might
            encounter `MaskedDistributions` whose official `base_dist` are `Delta`,
            but which should be treated as Normal or Categorical.

        Note: This class currently supports excision for Normal and Categorical distributions, and MaskedDistributions
        that wrap them. It contains a hack to work with reparametrized neural distributions in their current shape, to be refactored.
        Note also that `Excision` overrides observations at excised sites, so, unlike raw `ExcisedNormal` and `ExcisedCategorical`,
        is incompatible with inference.
        """

        for interval_list in intervals.values():
            for interval in interval_list:
                low_i, high_i = interval
                if not torch.all(low_i <= high_i).item():
                    raise ValueError("Each interval must satisfy low <= high.")

        self.intervals = intervals
        self.distribution_type_selectors = distribution_type_selectors

        super().__init__()

    def _pyro_sample(self, msg):
        """
        Replace the distribution at a registered sample site with an excised one.

        For sites named in ``intervals`` (and not flagged with
        ``_do_not_intervene``), determine the distribution type, build the
        corresponding excised distribution with the configured intervals removed,
        install it on the message, and resample the site's value from it.

        :param msg: The Pyro sample message being processed.
        :raises NotImplementedError: If logits cannot be extracted from a
            categorical site, or the distribution type is unsupported.
        """
        if msg["name"] not in self.intervals.keys() or msg["infer"].get(
            "_do_not_intervene", None
        ):
            return

        assert msg["name"] in self.intervals.keys()

        msg["is_observed"] = False

        distr = msg["fn"]
        name = msg["name"]
        intervals = self.intervals[name]

        distribution_type_indicator = (
            self.distribution_type_selectors.get(name, None)
            if self.distribution_type_selectors
            else None
        )

        if isinstance(distr, dist.Normal):
            distribution_type_indicator = "normal"
        elif isinstance(distr, dist.Categorical):
            distribution_type_indicator = "categorical"
        elif isinstance(distr, dist.Independent):
            base = distr.base_dist
            if isinstance(base, dist.Normal):
                distribution_type_indicator = "normal"
            elif isinstance(base, dist.Categorical):
                distribution_type_indicator = "categorical"

        # The use of "infer" here is a bit of a hack to get around MaskedDistributions
        # that result from our reparametrized distributions.
        # TODO simplify once reparametrization has been refactored
        def get_normal_params(distr):
            """
            Extract the location and scale of a (possibly wrapped) Normal site.

            Reads ``loc``/``scale`` directly from a Normal, falls back to the
            ``base_dist`` of the message's distribution, or to ``loc``/``scale``
            stashed in the message's ``infer`` dict for masked/independent
            reparametrized distributions.

            :param distr: The distribution to extract parameters from.
            :returns: The location and scale of the underlying Normal.
            """
            if isinstance(distr, dist.Normal):
                base_loc = getattr(distr, "loc", getattr(distr, "mean"))
                base_scale = getattr(distr, "scale", getattr(distr, "stddev", None))

            elif isinstance(msg["fn"].base_dist, dist.Normal):
                base = msg["fn"].base_dist
                base_loc = getattr(base, "loc", getattr(base, "mean"))
                base_scale = getattr(base, "scale", getattr(base, "stddev", None))

            elif isinstance(distr, dist.MaskedDistribution) or isinstance(
                distr, dist.Independent
            ):
                base_loc = msg["infer"].get("loc", None)
                base_scale = msg["infer"].get("scale", None)

            return base_loc, base_scale

        if isinstance(distr, dist.Normal) or distribution_type_indicator == "normal":
            base_loc, base_scale = get_normal_params(distr)

            excised_distr = ExcisedNormal(base_loc, base_scale, intervals)
            msg["fn"] = excised_distr
            if msg["value"] is not None:
                msg["value"] = excised_distr.sample().view(msg["value"].shape)

        elif (
            isinstance(distr, dist.Categorical)
            or distribution_type_indicator == "categorical"
        ):
            logits = getattr(distr, "logits", None)
            if (
                logits is None
                and isinstance(distr, dist.MaskedDistribution)
                or isinstance(distr, dist.Independent)
            ):
                base = getattr(distr, "base_dist", None)
                logits = getattr(base, "logits", None) if base is not None else None

                # fallback for custom 1d conditional distribution
                if logits is None and "infer" in msg:
                    logits = msg["infer"].get("logits", None)

            if logits is None:
                raise NotImplementedError(
                    f"Cannot extract logits from {type(distr).__name__}"
                )
            logits = logits.clone()

            if msg["value"] is not None:
                if len(logits.shape) == len(msg["value"].shape):
                    logits = logits.unsqueeze(-2)

                assert logits.shape[:-1] == msg["value"].shape

            excised_distr = ExcisedCategorical(intervals=intervals, logits=logits)
            msg["fn"] = excised_distr

            if msg["value"] is not None:
                msg["value"] = excised_distr.sample().view(msg["value"].shape)

        elif distribution_type_indicator == "truncated_normal":
            base_loc, base_scale = get_normal_params(distr)

            clamped_intervals = [
                (torch.clamp(low, 0.0, 1.0), torch.clamp(high, 0.0, 1.0))
                for low, high in intervals
            ]

            excised_distr = ExcisedNormal(base_loc, base_scale, clamped_intervals)
            msg["fn"] = excised_distr
            msg["value"] = torch.clamp(
                excised_distr.sample().view(msg["value"].shape), 0.0, 1.0
            )

        else:
            raise NotImplementedError(
                f"Unsupported distribution type: {type(distr).__name__}"
            )


if isinstance(pyro.poutine.handlers._make_handler(Excisions), tuple):
    excise = pyro.poutine.handlers._make_handler(Excisions)[1]
else:

    @pyro.poutine.handlers._make_handler(Excisions)
    def excise(
        fn: Callable,
        truncated_intervals: Mapping[Hashable, list[tuple[torch.Tensor, torch.Tensor]]],
        distribution_type_selectors: dict["str", "str"] | None = None,
    ):
        """
        Context-manager / decorator form of the :class:`Excisions` handler.

        Applies excision of the given intervals to ``fn`` (or to the enclosing
        model when used as a context manager).

        :param fn: The model or callable to apply excision to.
        :param truncated_intervals: A mapping from sample-site names to lists of
            intervals to remove from each site's support.
        :param distribution_type_selectors: Optional mapping from sample-site
            names to distribution type indicators ("normal" or "categorical").
        """
        ...


@contextmanager
def sample_alternatives(
    factuals: dict[str, torch.Tensor],
    epsilon: torch.Tensor,
    distribution_type_selectors: dict["str", "str"] | None = None,
):
    """
    Wraps `excise` to excise small intervals around given factual values.
    """
    intervals = {}
    for name, val in factuals.items():
        # ensure val is tensor
        val = torch.as_tensor(val)
        intervals[name] = [(val - epsilon, val + epsilon)]

    with excise(
        intervals=intervals, distribution_type_selectors=distribution_type_selectors
    ):
        yield
