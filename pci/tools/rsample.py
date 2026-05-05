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
        if event_dim is None:
            # FIXME this won't work generally, as the event dim depends on where we are in the transform stack.
            event_dim = self.transformed_event_dim
        return gather(v, get_factual_indices(), event_dim=event_dim)

    def factualize_transform_params(self, transform: Transform) -> Transform:
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
    raise NotImplementedError()


class Exogenate(pyro.poutine.messenger.Messenger):
    def __init__(self, noise_suffix: str | None = "_u"):
        super().__init__()

        self.noise_suffix = noise_suffix

        self.rsample_configs: dict[str, RSampleDistConfig] = {}
        self.base_noise_sites: dict[str, str] = {}
        self.base_noise_samples: dict[str, torch.Tensor] = {}

        self._used = False

    # messenger is stateful, reuse not allowed
    def __enter__(self):
        if self._used:
            raise RuntimeError(
                "Exogenate messenger instances are single-use. "
                "Create a new Exogenate() for each model execution."
            )
        self._used = True
        return super().__enter__()

    def _pyro_sample(self, msg) -> None:
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
        return Normal

    def _create_base_distribution(self, device: torch.device) -> Dist:
        return Normal(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        )

    def _create_transforms(self, original_dist: Dist) -> list[Transform]:
        assert isinstance(original_dist, Normal)
        return [AffineTransform(original_dist.loc, original_dist.scale)]

    def _create_analytic_log_prob_and_inv(
        self, original_dist: Dist, base_dist: Dist, event_shape: torch.Size
    ) -> AnalyticLogProbFn | None:
        assert isinstance(original_dist, Normal)
        return make_analytic_log_prob_and_inv(
            original_dist.loc, original_dist.scale, event_shape
        )


@dataclass
class AffineLogProbAndInv:
    x: torch.Tensor
    scale: torch.Tensor
    event_dim: int
    call_count: list[int] = field(default_factory=lambda: [0])

    def __call__(
        self, value: torch.Tensor, base_sample: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
