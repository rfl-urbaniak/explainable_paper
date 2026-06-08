import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from pyro.nn import PyroModule
from torch import nn


class SplineFlow(PyroModule):
    """Normalizing flow that transforms a standard normal base distribution
    through a stack of spline transforms.

    A Pyro module wrapping one or more (optionally coupling) spline transforms
    applied to a multivariate standard normal base, yielding a learnable
    transformed distribution suitable for flexible density modelling.
    """

    def __init__(
        self,
        dim: int,
        coupling: bool,
        count_bins: list[int],
        bound: float | None = None,
        event_dim: int = 1,
    ):
        """Build the base distribution buffers and the stack of spline transforms.

        :param dim: Dimensionality of the base distribution and each transform.
        :param coupling: Whether to use spline coupling transforms; otherwise
            plain spline transforms are used.
        :param count_bins: Number of spline bins for each transform in the
            stack, one entry per transform.
        :param bound: Boundary of the spline's support region; if not given the
            transform's default is used.
        :param event_dim: Number of rightmost dimensions treated as event
            dimensions of the base distribution.
        """
        super().__init__()

        self.event_dim = event_dim
        self.register_buffer("base_loc", torch.zeros(dim))
        self.register_buffer("base_scale", torch.ones(dim))

        self.transforms = []
        self.transform_modules = nn.ModuleList()
        for bins in count_bins:
            if coupling:
                transform = T.spline_coupling(dim, count_bins=bins, bound=bound)
            else:
                transform = T.Spline(dim, count_bins=bins, bound=bound)

            self.transforms.append(transform)
            self.transform_modules.append(transform)

    def forward(self):
        """Construct the transformed distribution from the base normal.

        Resolves the base location, scale, and event dimension (falling back to
        a previously stored ``base_dist`` for backwards compatibility with
        loaded objects lacking the registered buffers), then applies the spline
        transforms to the standard normal base.

        :returns: The transformed distribution defined by the spline stack over
            the standard normal base.
        """
        # Backwards compatibility with loaded SplineFlow objects that do not
        # have base_loc or base_scale registered as buffers.
        if hasattr(self, "base_dist"):
            device = next(self.parameters()).device
            base_loc = self.base_dist.base_dist.loc.to(device)
            base_scale = self.base_dist.base_dist.scale.to(device)
            event_dim = self.base_dist.event_dim
        else:
            base_loc = self.base_loc
            base_scale = self.base_scale
            event_dim = self.event_dim

        base_dist = dist.Normal(base_loc, base_scale).to_event(event_dim)
        return dist.TransformedDistribution(base_dist, self.transforms)
