import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from pyro.nn import PyroModule
from torch import nn


class SplineFlow(PyroModule):
    def __init__(
        self,
        dim: int,
        coupling: bool,
        count_bins: list[int],
        bound: float | None = None,
        event_dim: int = 1,
    ):
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
