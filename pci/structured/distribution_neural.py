import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from torch.nn import Sequential
from torchvision.ops import MLP


# for type hinting
class NeuralDistribution(pyro.nn.PyroModule):
    """Abstract base for neural distribution wrappers used as causal kernels.

    Subclasses parameterise a Pyro distribution (categorical or normal,
    conditional or unconditional) and sample a named outcome inside their
    :meth:`forward`. This base only provides a common type and an unimplemented
    forward contract.
    """

    def forward(
        self,
        condition: torch.Tensor,
    ):
        """Sample the wrapped distribution.

        :param condition: The conditioning context driving the distribution
            parameters; ignored by unconditional subclasses.
        :raises NotImplementedError: Always, since subclasses must override this.
        """
        raise NotImplementedError("Subclasses must implement a forward method")


class NeuralDistributionUnconditionalCategorical(NeuralDistribution):
    """Unconditional categorical distribution over a fixed set of outcomes.

    The category logits are free learnable parameters and do not depend on any
    conditioning context.
    """

    def __init__(
        self,
        outcome_name: str,
        num_outcome_cat: int,
        event_dim: int = 1,
    ):
        """Initialise the unconditional categorical wrapper.

        :param outcome_name: Name under which the sampled outcome is registered
            with Pyro.
        :param num_outcome_cat: Number of categories the outcome can take.
        :param event_dim: Number of trailing dimensions reinterpreted as event
            dimensions when sampling.
        """
        super().__init__()

        self.event_dim = event_dim
        self.outcome_name = outcome_name

        self.logits = nn.Parameter(torch.ones((1, num_outcome_cat)))

    def forward(
        self,
    ):
        """Sample the named categorical outcome from the learned logits.

        :returns: The sampled categorical outcome.
        """
        return pyro.sample(
            self.outcome_name,
            dist.Categorical(logits=self.logits).to_event(self.event_dim),
        )


class NeuralDistributionConditionalCategorical(NeuralDistribution):
    """Categorical distribution whose logits are predicted from a context.

    A neural network maps the conditioning context to per-category logits, so
    the outcome distribution depends on the input.
    """

    def __init__(
        self,
        outcome_name: str,
        num_outcome_cat: int,
        neural_net: torch.nn.Module | None = None,
        in_channels: int = 1,
        event_dim: int = 1,
    ):
        """Initialise the conditional categorical wrapper.

        :param outcome_name: Name under which the sampled outcome is registered
            with Pyro.
        :param num_outcome_cat: Number of categories the outcome can take.
        :param neural_net: Network mapping the context to category logits. If
            ``None``, a default MLP is constructed.
        :param in_channels: Width of the conditioning context, used to build the
            default network.
        :param event_dim: Number of trailing dimensions reinterpreted as event
            dimensions when sampling.
        """
        super().__init__()

        if neural_net is None:
            self.net: torch.nn.Module = make_default_categorical_neural_net(
                in_channels=in_channels,
                hidden_channels=[15, 10],
                activation_layer=nn.LeakyReLU,
                num_outcome_cat=num_outcome_cat,
            )

        else:
            self.net = neural_net

        self.outcome_name = outcome_name
        self.num_outcome_cat = num_outcome_cat
        self.in_channels = in_channels
        self.event_dim = event_dim

    def forward(self, condition: torch.Tensor):
        """Predict logits from the context and sample the categorical outcome.

        :param condition: The conditioning context fed to the network to produce
            the category logits.
        :returns: The sampled categorical outcome.
        """
        logits = self.net(condition)

        sample = pyro.sample(
            self.outcome_name, dist.Categorical(logits=logits.unsqueeze(-2)).to_event(1)
        )

        # warning, adding to event might causes issues with log prob sum in traces, needs to be checked
        # NOTE: the mw dimension for categorical seems to be different from the one for normal, potentially investigate
        # TODO make sure output passing between causal kernels in Roc360 works as intended
        # w/o violating shape and event assumptions
        # remove before merging staging into master

        return sample


class NeuralDistributionUnconditionalNormal(NeuralDistribution):
    """Unconditional normal distribution with learnable location and scale.

    The location and (softplus-transformed) scale are free parameters and do not
    depend on any conditioning context.
    """

    def __init__(
        self,
        outcome_name: str,
        loc: torch.Tensor = torch.tensor([0.0]),
        raw_scale: torch.Tensor = torch.tensor([1.0]),
        event_dim: int = 1,
    ):
        """Initialise the unconditional normal wrapper.

        :param outcome_name: Name under which the sampled outcome is registered
            with Pyro.
        :param loc: Initial value for the learnable location parameter.
        :param raw_scale: Initial pre-softplus value for the scale parameter; the
            effective scale is obtained by applying softplus.
        :param event_dim: Number of trailing dimensions reinterpreted as event
            dimensions when sampling.
        """
        super().__init__()
        self.loc = nn.Parameter(loc)
        self.raw_scale = nn.Parameter(raw_scale)
        self.softplus = nn.Softplus()
        self.event_dim = event_dim
        self.outcome_name = outcome_name

    @property
    def scale(self) -> torch.Tensor:
        """The positive scale, obtained from the raw parameter via softplus.

        :returns: The softplus-transformed scale with a small positive floor for
            numerical stability.
        """
        return self.softplus(self.raw_scale) + 1e-6

    def forward(
        self,
    ):
        """Sample the named normal outcome from the learned location and scale.

        :returns: The sampled normal outcome.
        """
        scale = self.softplus(self.raw_scale)
        return pyro.sample(
            self.outcome_name, dist.Normal(self.loc, scale).to_event(self.event_dim)
        )


class NeuralDistributionConditionalNormal(NeuralDistribution):
    """Normal distribution whose location and scale are predicted from a context.

    Separate neural networks map the conditioning context to the location and the
    (positive) scale, so the outcome distribution depends on the input.
    """

    def __init__(
        self,
        outcome_name: str,
        loc_neural_net: torch.nn.Module | None = None,
        scale_neural_net: torch.nn.Module | None = None,
        in_channels: int = 1,
        event_dim: int = 1,
    ):
        """Initialise the conditional normal wrapper.

        :param outcome_name: Name under which the sampled outcome is registered
            with Pyro.
        :param loc_neural_net: Network mapping the context to the location. If
            ``None`` (or together with ``scale_neural_net``), a default is built.
        :param scale_neural_net: Network mapping the context to the scale. If
            ``None`` (or together with ``loc_neural_net``), a default is built.
        :param in_channels: Width of the conditioning context, used to build the
            default networks.
        :param event_dim: Number of trailing dimensions reinterpreted as event
            dimensions when sampling.
        """
        super().__init__()

        self.outcome_name = outcome_name

        if loc_neural_net is None or scale_neural_net is None:
            _loc_neural_net, _scale_neural_net = make_default_normal_neural_nets(
                in_channels=in_channels,
                hidden_channels=[15, 10],
                activation_layer=nn.LeakyReLU,
            )

            self.loc_neural_net = (
                loc_neural_net if loc_neural_net is not None else _loc_neural_net
            )
            self.scale_neural_net = (
                scale_neural_net if scale_neural_net is not None else _scale_neural_net
            )

        else:
            self.loc_neural_net = loc_neural_net
            self.scale_neural_net = scale_neural_net

        self.event_dim = event_dim

    def forward(self, condition: torch.Tensor):
        """Predict location and scale from the context and sample the outcome.

        The predicted scale is floored at a small positive value to avoid
        underflow when random initialisation yields large negative inputs.

        :param condition: The conditioning context fed to the location and scale
            networks.
        :returns: The sampled normal outcome.
        """
        loc = self.loc_neural_net(condition)
        scale = self.scale_neural_net(condition).clamp_min(
            1e-4
        )  # with random init, condition can be large negative
        # avoid underflow
        return pyro.sample(
            self.outcome_name, dist.Normal(loc, scale).to_event(self.event_dim)
        )


def make_default_categorical_neural_net(
    in_channels: int,
    hidden_channels: list[int],
    num_outcome_cat: int,
    activation_layer=nn.LeakyReLU,
) -> torch.nn.Module:
    """Build the default MLP that maps a context to categorical logits.

    Stacks a hidden MLP with a final linear layer producing one logit per
    category.

    :param in_channels: Width of the conditioning context input.
    :param hidden_channels: Widths of the hidden layers; must be non-empty.
    :param num_outcome_cat: Number of categories, i.e. the size of the logit
        output.
    :param activation_layer: Activation module class used between hidden layers.
    :returns: The constructed logit-producing network.
    """
    assert len(hidden_channels) > 0, "hidden_channels must be non-empty"

    assert activation_layer is not None

    net = MLP(
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        activation_layer=activation_layer,
    )

    logit_layer = nn.Linear(hidden_channels[-1], num_outcome_cat)

    net = Sequential(net, logit_layer)

    return net


def make_default_normal_neural_nets(
    in_channels: int, hidden_channels: list[int], activation_layer=nn.LeakyReLU
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build the default pair of MLPs for a conditional normal distribution.

    The location network ends in a linear layer; the scale network ends in a
    linear layer followed by softplus to keep the scale positive.

    :param in_channels: Width of the conditioning context input.
    :param hidden_channels: Widths of the hidden layers; must be non-empty.
    :param activation_layer: Activation module class used between hidden layers.
    :returns: A pair ``(loc_net, scale_net)`` predicting the location and the
        positive scale respectively.
    """
    assert len(hidden_channels) > 0, "hidden_channels must be non-empty"
    assert activation_layer is not None

    loc_net = MLP(
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        activation_layer=activation_layer,
    )

    scale_net = MLP(
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        activation_layer=activation_layer,
    )

    loc_layer = nn.Linear(hidden_channels[-1], 1)
    scale_layer = nn.Linear(hidden_channels[-1], 1)
    softplus = nn.Softplus()

    loc_net = Sequential(loc_net, loc_layer)
    scale_net = Sequential(scale_net, scale_layer, softplus)

    return loc_net, scale_net


class DataEmbedder(nn.Module):
    """Embeds categorical inputs and concatenates them with continuous inputs.

    Each named categorical input is passed through its own learned embedding, the
    results are concatenated with the continuous inputs, and the whole is
    broadcast to a common batch shape to form a single feature tensor.
    """

    def __init__(
        self,
        input_cat_cardinalities: dict[str, int],
        cat_embed_dims: dict[str, int],
        event_dim: int = 1,
    ):
        """
        Embeds categorical variables and concatenates them with continuous variables.

        :param input_cat_cardinalities: Mapping from categorical variable names to their number of categories,
            e.g. ``{"cat1": 3, "cat2": 5}``.
        :param cat_embed_dims: Mapping from categorical variable names to their embedding dimensions,
            e.g. ``{"cat1": 2, "cat2": 3}``.
        :param event_dim: Number of trailing dimensions to treat as event dimensions.

        .. note::
            All categorical variables must have corresponding embedding dimensions. We assume the
            dimensions of the input tensors are (..., Obs, 1) for categorical variables,
            and (..., Obs, event_dim) for continuous variables. This makes sense, as the last dimension
            of categorical variables is just a placeholder for the category index.
        """
        super().__init__()
        self.embedders = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=input_cat_cardinalities[name],
                    embedding_dim=cat_embed_dims[name],
                )
                for name in input_cat_cardinalities
            }
        )

        self.event_dim = event_dim

    def forward(
        self,
        **kwargs: torch.Tensor,
    ):
        """
        Named arguments for all features.
        Categorical tensors must match keys in self.embedders.
        Continuous tensors can be any other named argument.

        Example::

            x, y = embedder(cat1=..., cat2=..., cont1=..., cont2=..., outcome=...)
        """
        cat_embeds = []
        cont_features = []

        for name, tensor in kwargs.items():
            if name in self.embedders:
                cat_embeds.append(self.embedders[name](tensor.squeeze(-1)))
            else:
                cont_features.append(tensor)

        xs = []
        if cat_embeds:
            x_cat = torch.cat(
                self.broadcast_batch(*cat_embeds, event_dim=self.event_dim),
                dim=-1,
            )
            xs.append(x_cat)

        if cont_features:
            x_cont = torch.cat(
                self.broadcast_batch(*cont_features, event_dim=self.event_dim),
                dim=-1,
            )
            xs.append(x_cont)

        x = torch.cat(
            self.broadcast_batch(*xs, event_dim=self.event_dim),
            dim=-1,
        )

        return x

    def broadcast_batch(self, *tensors, event_dim=1):
        """Broadcast several tensors to a common batch shape, keeping event dims.

        Computes the broadcast of the batch shapes (all dimensions except the
        trailing ``event_dim`` ones) and expands every tensor to it, leaving each
        tensor's own event dimensions untouched.

        :param tensors: Tensors to align to a shared batch shape.
        :param event_dim: Number of trailing dimensions treated as event
            dimensions and excluded from broadcasting.
        :returns: The input tensors expanded to the common batch shape.
        """
        batch_shapes = [t.shape[:-event_dim] for t in tensors]
        broadcasted_batch_shape = torch.broadcast_shapes(*batch_shapes)
        return [
            t.expand(*broadcasted_batch_shape, *t.shape[-event_dim:]) for t in tensors
        ]
