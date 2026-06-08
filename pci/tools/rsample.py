import functools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable as TypingCallable
from copy import copy
from dataclasses import dataclass, field

import pyro
import torch
from chirho.counterfactual.handlers.selection import get_factual_indices
from chirho.indexed.ops import IndexSet, gather, get_index_plates, indices_of, scatter
from pyro.distributions import Delta, Independent, Normal
from pyro.distributions import TorchDistribution as Dist
from pyro.distributions.transforms import AffineTransform, ComposeTransform, Transform
from pyro.poutine.runtime import get_plates
from torch.distributions.utils import _sum_rightmost

# warning lives in the function for future generations
# but not an issue for categorical or normal as we use it
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

AnalyticLogProbFn = TypingCallable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass
class RSampleDistConfig:
    """
    Configuration object describing how to obtain reparameterized samples
    (``rsample``-style) from a base distribution combined with a sequence
    of transforms. It bundles together all information required to

    * invert a transformed random variable back to base noise,
    * compute the associated log-probability correction terms, and
    * optionally bypass generic inversion logic using an analytic routine.

    Attributes
    ----------
    base_dist
        The underlying base distribution prior to any transformations.
        This is typically a simple distribution such as a Normal or
        Uniform distribution.

    transforms
        A sequence of transforms applied to samples from ``base_dist``.
        The transforms are assumed to be compatible with PyTorch's
        ``Transform`` API and are applied in order.

    analytic_log_prob_and_inv
        Optional callable providing a closed-form computation of the
        log-probability and inverse mapping. If provided, this function
        is used instead of the generic transform inversion machinery.
    """

    base_dist: Dist
    transforms: list[Transform]
    analytic_log_prob_and_inv: AnalyticLogProbFn | None = None

    @functools.cached_property
    def transformed_event_dim(self) -> int:
        """
        Compute the event dimension of the distribution obtained by applying
        the configured transforms to ``base_dist``.

        Mirrors the event-dimension logic of
        ``torch.distributions.TransformedDistribution``: it accounts for any
        change in event dimension introduced by the composed transform and
        takes the larger of the transform's own coupling and the base
        distribution's event dimension shifted by that change.

        :returns: The event dimension of the transformed distribution.
        """
        # This logic extracted from torch.distributions.TransformedDistribution.__init__
        base_event_dim = len(self.base_dist.event_shape)

        transform = ComposeTransform(self.transforms)

        transform_change_in_event_dim = (
            transform.codomain.event_dim - transform.domain.event_dim
        )

        event_dim = max(
            transform.codomain.event_dim,  # the transform is coupled
            base_event_dim + transform_change_in_event_dim,  # the base dist is coupled
        )

        return event_dim

    def get_log_prob_and_u(
        self, value: torch.Tensor, base_sample: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recover the base noise underlying an observed value and the associated
        log-probability correction.

        If an analytic log-prob/inverse callable is configured, it is used
        directly (with log-probability masking, so any internal ``pyro.sample``
        calls do not contribute to the trace). Otherwise the transforms are
        inverted in reverse order, accumulating the negated log absolute
        determinant of the Jacobian at each step to form the correction term.
        The base distribution's own log-probability is deliberately omitted,
        since the base noise sample site contributes it separately.

        :param value: The transformed (observed) value to invert.
        :param base_sample: Optional base noise sample forwarded to the
            analytic callable so it can reuse identical noise.
        :returns: A pair of the log-probability correction and the recovered
            base noise.
        """
        # If analytic function is provided, use it instead of transform inversion
        if self.analytic_log_prob_and_inv is not None:
            # Mask log probabilities in case the analytic function uses pyro.sample
            with pyro.poutine.mask(mask=False):
                log_prob, u = self.analytic_log_prob_and_inv(
                    value, base_sample=base_sample
                )
            return log_prob, u

        # Default approach: use transform inversion machinery
        # This logic extracted from torch.distributions.TransformedDistribution.log_prob

        event_dim = self.transformed_event_dim
        log_prob = torch.zeros((), dtype=value.dtype, device=value.device)
        z = self.get_factual_values(value)
        for transform in reversed(self.transforms):
            transform = self.factualize_transform_params(transform)
            x = self.get_factual_values(transform.inv(z))
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, z),
                event_dim - transform.domain.event_dim,
            )
            z = x

        # not doing this b/c base dist sample site adds its own log prob.
        # log_prob = log_prob + _sum_rightmost(
        #     self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        # )
        return log_prob, z

    def get_factual_values(self, v, event_dim=None) -> torch.Tensor:
        """
        Gather the factual (non-counterfactual) slice of a tensor along the
        active counterfactual index plates.

        :param v: The tensor to gather factual values from.
        :param event_dim: The event dimension to respect when gathering;
            defaults to the transformed distribution's event dimension.
        :returns: The factual slice of ``v``.
        """
        if event_dim is None:
            # FIXME this won't work generally, as the event dim depends on where we are in the transform stack.
            event_dim = self.transformed_event_dim
        return gather(v, get_factual_indices(), event_dim=event_dim)

    def factualize_transform_params(self, transform: Transform) -> Transform:
        """
        Return a copy of a transform whose tensor-valued parameters have been
        replaced by their factual slices.

        Each attribute of the transform is gathered to its factual values where
        possible; scalar numeric parameters are left untouched when an event
        dimension is present (since gathering does not apply to them), and
        parameters that cannot be gathered are skipped. This lets transforms
        carrying counterfactual parameters be evaluated on factual data.

        :param transform: The transform whose parameters should be factualized.
        :returns: A copy of the transform with factualized parameters.
        """
        warnings.warn(
            "Using very invasive strategy in RSampleDistConfig.factualize_transform_params."
        )
        _transform = copy(transform)
        event_dim = self.transformed_event_dim
        for k, v in transform.__dict__.items():
            # Skip scalar/number values when event_dim > 0, as gather doesn't support event_dim for scalars
            if event_dim > 0 and isinstance(v, int | float) and not torch.is_tensor(v):
                continue
            try:
                setattr(_transform, k, self.get_factual_values(v, event_dim=event_dim))
            except (ValueError, NotImplementedError):
                pass
        return _transform


@pyro.poutine.runtime.effectful(type="rsample")
def rsample(
    name: str,
    base_dist: Dist,
    transforms: list[Transform],
    analytic_log_prob_and_inv: AnalyticLogProbFn | None = None,
) -> torch.Tensor:
    """
    Effectful entry point for drawing a reparameterized sample from a base
    distribution composed with a sequence of transforms.

    This function has no default implementation; it is intended to be handled
    by an effect handler such as :class:`Exogenate`, which interprets the
    ``rsample`` effect to materialize the base noise site and the transformed
    value. Calling it without an active handler raises
    :exc:`NotImplementedError`.

    :param name: The sample site name for the transformed value.
    :param base_dist: The base distribution to sample noise from.
    :param transforms: The transforms applied in order to the base noise.
    :param analytic_log_prob_and_inv: Optional analytic log-prob/inverse
        callable used in place of generic transform inversion.
    :returns: The reparameterized sample (supplied by the handling effect
        handler).
    :raises NotImplementedError: If invoked without an effect handler.
    """
    raise NotImplementedError()


class Exogenate(pyro.poutine.messenger.Messenger):
    """
    Pyro effect handler that exogenizes ``rsample`` sites by exposing their
    base noise as explicit sample sites.

    When this messenger is active, each ``rsample`` effect is interpreted to
    create a separate base-noise sample site alongside the transformed value,
    making the underlying exogenous randomness available for conditioning and
    counterfactual reasoning. The handler tracks per-site reparameterization
    configurations, the names of their base-noise sites, and the noise samples
    drawn for them. Instances are single-use and may not be re-entered.
    """

    def __init__(self, noise_suffix: str | None = "_u"):
        """
        Initialize the messenger and its per-site bookkeeping.

        :param noise_suffix: Suffix appended to a site's name to form the name
            of its exogenous base-noise sample site.
        """
        super().__init__()

        self.noise_suffix = noise_suffix

        self.rsample_configs: dict[str, RSampleDistConfig] = {}
        self.base_noise_sites: dict[str, str] = {}
        self.base_noise_samples: dict[str, torch.Tensor] = {}

        self._used = False

    # messenger is stateful, reuse not allowed
    def __enter__(self):
        """
        Enter the messenger context, enforcing single use.

        :returns: The activated messenger context.
        :raises RuntimeError: If this instance has already been used; a fresh
            :class:`Exogenate` must be created for each model execution.
        """
        if self._used:
            raise RuntimeError(
                "Exogenate messenger instances are single-use. "
                "Create a new Exogenate() for each model execution."
            )
        self._used = True
        return super().__enter__()

    def _pyro_sample(self, msg) -> None:
        """
        Handle ``sample`` effects at exogenated sites.

        For a site that has an associated reparameterization configuration,
        this dispatches to the observation or non-observation handler depending
        on whether the site is observed; sites without a configuration are left
        untouched.

        :param msg: The Pyro effect message for the sample site.
        """
        rsample_config = self.rsample_configs.get(msg["name"], None)

        if rsample_config is None:
            return

        if msg["is_observed"]:
            self.handle_observation_of_exogenated_sites(msg, rsample_config)
        else:
            self.handle_nonobservation_of_exogenated_sites(msg, rsample_config)

    def handle_observation_of_exogenated_sites(
        self, msg, rsample_config: RSampleDistConfig
    ) -> None:
        """
        Handle an observed exogenated site by inferring its base noise and
        re-expressing the site as a deterministic projection of that noise.

        The observed value is inverted to recover the base noise, which is
        broadcast to the base distribution's shape if needed and then pushed
        back through the transforms to obtain a projected value. The factual
        observation is scattered into this projected value so factual and
        counterfactual worlds coexist, the site's function is replaced by a
        :class:`Delta` carrying the log-probability correction, and the inferred
        noise is recorded by observing the base-noise sample site. When index
        plates are active, downstream plate expansion of the ``Delta`` is
        suppressed.

        :param msg: The Pyro effect message for the observed sample site.
        :param rsample_config: The reparameterization configuration for the
            site.
        """
        _lp, inferred_u = rsample_config.get_log_prob_and_u(msg["value"])

        # Get event_dim first
        event_dim = rsample_config.transformed_event_dim
        has_index_plates = len(get_index_plates()) > 0

        # Compute target batch shape including active pyro plates
        target_batch_shape = rsample_config.base_dist.batch_shape
        target_event_shape = rsample_config.base_dist.event_shape
        target_full_shape = target_batch_shape + target_event_shape

        # Check if we need to expand for plate compatibility
        # Expansion is needed when base_dist has larger batch shape than inferred values,
        # which happens when the site is inside active pyro plates but the inferred values
        # don't yet have those plate dimensions. If target_batch_shape is actually smaller
        # or equal to the inferred shape, no expansion is needed and we avoid unnecessary ops.
        needs_expansion = len(target_full_shape) > len(inferred_u.shape)
        if needs_expansion:
            inferred_u = inferred_u.broadcast_to(target_full_shape)

        # Let scatter infer the result shape based on indexset and index plates.
        # It will create a result that can hold both factual and counterfactual log probs.
        lp = scatter(
            _lp,
            get_factual_indices(),
            # No event dim gets passed here, as log probs have no event dim.
            event_dim=0,
        )

        projected_value = inferred_u
        for transform in rsample_config.transforms:
            projected_value = transform(projected_value)

        # Gather factual observation and scatter into projected value
        gathered_value = gather(
            msg["value"], get_factual_indices(), event_dim=event_dim
        )
        scatter(
            gathered_value,
            get_factual_indices(),
            result=projected_value,
            event_dim=event_dim,
        )

        msg["value"] = projected_value
        msg["fn"] = Delta(msg["value"], event_dim=event_dim, log_density=lp)

        # Prevent plate messenger from expanding Delta with CF splits
        if has_index_plates:
            msg["stop"] = True

        pyro.sample(
            self.base_noise_sites[msg["name"]], rsample_config.base_dist, obs=inferred_u
        )

    def handle_nonobservation_of_exogenated_sites(
        self, msg, rsample_config: RSampleDistConfig
    ) -> None:
        """
        Handle an unobserved exogenated site by replaying its base noise.

        The base-noise sample drawn earlier for this site is re-observed at its
        base-noise sample site, so the exogenous randomness is pinned while the
        site's transformed value continues to be produced by the ``rsample``
        machinery.

        :param msg: The Pyro effect message for the unobserved sample site.
        :param rsample_config: The reparameterization configuration for the
            site.
        """
        base_noise_name = self.base_noise_sites[msg["name"]]
        pyro.sample(
            base_noise_name,
            rsample_config.base_dist,
            # TODO observing this is weird.
            obs=self.base_noise_samples[base_noise_name],
        )

        # lp sent to Delta does not contain the base dist log prob.
        # It's just the log abs det jac part, then the base dist adds its own log prob.

    def _pyro_rsample(self, msg) -> None:
        """
        Interpret an ``rsample`` effect by drawing base noise and emitting the
        transformed value.

        A reparameterization configuration is built from the effect arguments
        and validated to ensure the base distribution carries no counterfactual
        indices. Base noise is sampled (without contributing to the trace),
        recorded for later replay, and pushed through the transforms. The
        resulting value is emitted at the named site as a :class:`Delta`
        carrying the log-probability correction, with the original base sample
        forwarded so analytic helpers can reuse identical noise.

        :param msg: The Pyro effect message for the ``rsample`` effect.
        :raises ValueError: If the base distribution has counterfactual
            indices, which typically indicates an intervened variable is used
            as one of its parameters.
        """
        name = msg["name"] = msg["args"][0]

        rsample_config = self.rsample_configs[name] = RSampleDistConfig(**msg["kwargs"])

        # Check that base_dist doesn't have any counterfactual indices
        # This would happen if e.g. we intervene on x and then use x as a param to base_dist
        base_dist = rsample_config.base_dist
        base_event_dim = len(base_dist.event_shape)

        idxof = indices_of(base_dist, event_dim=base_event_dim)
        if idxof != IndexSet():
            raise ValueError(
                f"Base distribution has counterfactual indices {idxof}. "
                f"Base distributions in rsample must not have counterfactual dimensions. "
                f"This typically happens when an intervened variable is used as a parameter to the base distribution."
            )

        base_noise_name = self.base_noise_sites[name] = f"{name}{self.noise_suffix}"
        with pyro.poutine.mask(mask=False):
            u = y = pyro.sample(
                f"_{base_noise_name}_unconditioned", rsample_config.base_dist
            )
        self.base_noise_samples[base_noise_name] = u

        for transform in rsample_config.transforms:
            y = transform(y)

        msg["value"] = pyro.sample(
            name,
            Delta(
                y,
                event_dim=rsample_config.transformed_event_dim,
                # NOTE r9t7k1: forward the actual base sample so analytic helpers that
                # approximate absent log-Jacobians can reuse identical noise.
                log_density=rsample_config.get_log_prob_and_u(y, base_sample=u)[0],
            ),
        )


class RSampleSites(pyro.poutine.messenger.Messenger, ABC):
    """
    Base class for rsampling sites with specific distribution types.

    Subclasses should implement abstract methods to specify:
    - The target distribution type to handle
    - The standardized base distribution for reparameterization
    - The transforms to apply to the base distribution
    """

    def __init__(self, *sites: str):
        """
        Initialize the messenger with the set of sites to reparameterize.

        :param sites: Names of the sample sites this messenger should rewrite
            via reparameterized sampling.
        """
        super().__init__()
        self.sites = set(sites)

    @abstractmethod
    def _get_target_distribution_type(self):
        """Return the distribution class this messenger handles (e.g., Normal, Categorical)."""
        pass

    def _get_device(self, original_dist: Dist) -> torch.device:
        """Get the device from the original distribution's parameters."""
        for attr in ("loc", "mean"):
            if hasattr(original_dist, attr):
                param = getattr(original_dist, attr)
                if isinstance(param, torch.Tensor):
                    return param.device
        return torch.device("cpu")

    @abstractmethod
    def _create_base_distribution(self, device: torch.device) -> Dist:
        """Create the standardized base distribution for rsampling (e.g., Normal(0, 1), Gumbel(0, 1))."""
        pass

    @abstractmethod
    def _create_transforms(self, original_dist: Dist) -> list[Transform]:
        """Create the list of transforms to apply to the base distribution."""
        pass

    def _create_analytic_log_prob_and_inv(
        self, original_dist: Dist, base_dist: Dist, event_shape: torch.Size
    ) -> AnalyticLogProbFn | None:
        """
        Optionally create an analytic log prob and inversion function.

        This is useful for non-invertible transforms where we can't compute
        the log abs det Jacobian analytically but can sample from the preimage.

        Returns None by default (use automatic transform inversion).
        """
        return None

    def _get_base_event_shape(self, original_dist: Dist, fn: Dist) -> torch.Size:
        """
        Get the event shape for the base distribution.

        By default, uses fn.event_shape. Subclasses can override this if the
        base distribution needs a different event shape than the output distribution.
        """
        return fn.event_shape

    def _extract_distribution(self, fn: Dist) -> Dist | None:
        """
        Extract the target distribution from fn, handling Independent wrappers.

        Returns the unwrapped distribution if it matches the target type, None otherwise.
        """
        target_type = self._get_target_distribution_type()

        if isinstance(fn, target_type):
            return fn
        elif isinstance(fn, Independent) and isinstance(fn.base_dist, target_type):
            return fn.base_dist

        # TODO this should warn/error that a specified site didn't have a matching base dist.

        return None

    def _pyro_sample(self, msg) -> None:
        """
        Rewrite a targeted sample site into a reparameterized sample.

        For sites in this messenger's set whose function matches the target
        distribution type (unwrapping :class:`Independent` if needed), this
        builds a standardized base distribution and the transforms mapping it to
        the original distribution, expands the base distribution to the
        appropriate batch and event shape (accounting for both interventional
        index plates and active Pyro plates), optionally constructs an analytic
        log-prob/inverse function, and replaces the site's value with an
        :func:`rsample` draw. Non-matching sites are left untouched.

        :param msg: The Pyro effect message for the sample site.
        """
        if msg["name"] not in self.sites:
            return

        fn = msg["fn"]

        # Extract the target distribution, handling Independent wrappers
        original_dist = self._extract_distribution(fn)

        if original_dist is None:
            return

        # Use fn.batch_shape instead of original_dist.batch_shape to correctly handle
        # Independent distributions where event dimensions have been moved
        noninterventional_batch_shape = list(fn.batch_shape)
        for interventional_plate in get_index_plates().values():
            if abs(interventional_plate.dim) <= len(noninterventional_batch_shape):
                noninterventional_batch_shape[interventional_plate.dim] = 1

        # Also account for active pyro plates that may not be in fn.batch_shape
        # (e.g., when the distribution parameters don't have plate dimensions)
        active_plates = get_plates()
        for plate in active_plates:
            plate_dim = plate.dim
            assert plate_dim is not None

            # Ensure the batch shape is large enough to include this plate dimension
            while abs(plate_dim) > len(noninterventional_batch_shape):
                noninterventional_batch_shape.insert(0, 1)
            # Set the plate size at the appropriate dimension
            if noninterventional_batch_shape[plate_dim] == 1:
                noninterventional_batch_shape[plate_dim] = plate.size

        # Get device from original distribution parameters
        device = self._get_device(original_dist)
        base_dist = self._create_base_distribution(device=device)
        transforms = self._create_transforms(original_dist)

        # Get the event shape for the base distribution (may differ from output event shape)
        base_event_shape = self._get_base_event_shape(original_dist, fn)

        # Create the expanded base distribution
        expanded_base = base_dist.expand(
            torch.Size(noninterventional_batch_shape) + base_event_shape
        ).to_event(len(base_event_shape))

        # Optionally create analytic log prob and inversion function
        analytic_fn = self._create_analytic_log_prob_and_inv(
            original_dist, expanded_base, base_event_shape
        )

        msg["value"] = rsample(
            msg["name"],
            base_dist=expanded_base,
            transforms=transforms,
            analytic_log_prob_and_inv=analytic_fn,
        )
        msg["stop"] = True


class RSampleNormalSites(RSampleSites):
    """Rsample messenger for Normal distributions."""

    def _get_target_distribution_type(self):
        """
        Identify the distribution type handled by this messenger.

        :returns: The :class:`Normal` distribution class.
        """
        return Normal

    def _create_base_distribution(self, device: torch.device) -> Dist:
        """
        Create the standard normal base distribution for reparameterization.

        :param device: The device on which to place the distribution's
            parameters.
        :returns: A standard normal base distribution.
        """
        return Normal(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        )

    def _create_transforms(self, original_dist: Dist) -> list[Transform]:
        """
        Build the affine transform mapping standard normal noise to the
        original normal distribution.

        :param original_dist: The original normal distribution whose location
            and scale parameterize the transform.
        :returns: A list containing the affine transform.
        """
        assert isinstance(original_dist, Normal)
        return [AffineTransform(original_dist.loc, original_dist.scale)]

    def _create_analytic_log_prob_and_inv(
        self, original_dist: Dist, base_dist: Dist, event_shape: torch.Size
    ) -> AnalyticLogProbFn | None:
        """
        Build the analytic log-prob/inverse function for the normal site's
        affine transform.

        :param original_dist: The original normal distribution supplying the
            location and scale parameters.
        :param base_dist: The expanded base distribution (unused here).
        :param event_shape: The event shape used to determine the event
            dimension.
        :returns: An analytic log-prob/inverse callable for the affine
            transform.
        """
        assert isinstance(original_dist, Normal)
        return make_analytic_log_prob_and_inv(
            original_dist.loc, original_dist.scale, event_shape
        )


@dataclass
class AffineLogProbAndInv:
    """
    Analytic log-probability and inverse for a single affine transform.

    Captures the location and scale of an affine transform and provides a
    closed-form inversion together with the corresponding log absolute
    determinant of the Jacobian, avoiding the generic transform-inversion
    machinery. A mutable call counter records how many times it has been
    invoked.
    """

    x: torch.Tensor
    scale: torch.Tensor
    event_dim: int
    call_count: list[int] = field(default_factory=lambda: [0])

    def __call__(
        self, value: torch.Tensor, base_sample: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Invert an affine-transformed value and return its log-probability
        correction.

        The factual slices of the value, location, and scale are gathered, the
        base noise is recovered as ``(value - loc) / scale``, and the
        log-probability correction is ``-log(scale)`` summed over the event
        dimensions. Increments the call counter on each invocation.

        :param value: The affine-transformed value to invert.
        :param base_sample: Unused here, since the log absolute determinant of
            the Jacobian is available analytically.
        :returns: A pair of the log-probability correction and the recovered
            base noise.
        """
        self.call_count[0] += 1

        # NOTE r9t7k1: base_sample unused here because log abs det jac is available,
        # so this analytic helper only needs the inverse logic.
        # Extract factual values - gather removes counterfactual dimensions
        # Use event_dim parameter to match get_factual_values behavior
        value_factual = gather(value, get_factual_indices(), event_dim=self.event_dim)
        x_factual = gather(self.x, get_factual_indices(), event_dim=self.event_dim)
        scale_factual = gather(
            self.scale, get_factual_indices(), event_dim=self.event_dim
        )

        # Inverse transform: u = (value - loc) / scale
        u = (value_factual - x_factual) / scale_factual

        # Log abs det jacobian for inverse: -log(scale)
        log_prob = -torch.log(scale_factual)

        # Sum over event dimensions to ensure log_prob has only batch shape
        if self.event_dim > 0:
            log_prob = _sum_rightmost(log_prob, self.event_dim)

        return log_prob, u


def make_analytic_log_prob_and_inv(
    x: torch.Tensor, scale: torch.Tensor, event_shape: torch.Size
) -> AffineLogProbAndInv:
    """Create an analytic log prob and inversion function for AffineTransform.

    This is a closure that captures x and scale, and returns a function that
    emulates get_log_prob_and_u for a single AffineTransform.

    Args:
        x: Location parameter
        scale: Scale parameter
        event_shape: Event shape tuple (e.g., () for scalar, (3,) for event dim of size 3)
    """
    return AffineLogProbAndInv(x, scale, len(event_shape))
