"""
Structured causal models with topologically ordered kernels.

This module provides the core infrastructure for building structured causal models
using PyTorch and Pyro. It includes various types of causal kernels (unconditional
and conditional, for different distribution types) and a StructuredModel class that
orchestrates their execution in topological order.

Classes
-------
CausalKernel
    Base class for all causal kernels
CausalKernelUnconditional
    Base class for unconditional causal kernels (root nodes)
CausalKernelUnconditionalCategorical
    Unconditional kernel for categorical distributions
CausalKernelUnconditionalNormal
    Unconditional kernel for normal distributions
CausalKernelUnconditionalFlow
    Unconditional kernel using normalizing flows
CausalKernelConditional
    Base class for conditional causal kernels
CausalKernelConditionalNormal
    Conditional kernel for normal distributions
CausalKernelConditionalCategorical
    Conditional kernel for categorical distributions
StructuredModel
    A structured causal model composed of topologically ordered kernels
"""

import graphlib
from collections.abc import Collection, Mapping

import pyro
import torch

from pci.structured.distribution_flow import SplineFlow
from pci.structured.distribution_neural import (
    DataEmbedder,
    NeuralDistributionConditionalCategorical,
    NeuralDistributionConditionalNormal,
    NeuralDistributionUnconditionalCategorical,
    NeuralDistributionUnconditionalNormal,
)
from pci.structured.schemata import Cat, Flow, Norm, Schema


class CausalKernel(pyro.nn.PyroModule):
    """
    Base class for all causal kernels in structured models.

    :ivar outcome_schema: Schema defining the outcome variable.
    :ivar outcome_name: Name of the outcome variable.
    :ivar event_dim: Dimensionality of the outcome variable.
    """

    outcome_schema: Schema

    def __init__(self, outcome_schema: Schema, **kwargs):
        """
        :param outcome_schema: Schema defining the outcome variable.
        """
        super().__init__()
        self.outcome_schema = outcome_schema
        self.outcome_name = list(outcome_schema.keys())[0]
        self.event_dim = len(outcome_schema[self.outcome_name].component_feature_names)

    def forward(self, *args, **kwargs):
        """
        :raises NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError


class CausalKernelUnconditional(CausalKernel):
    """
    Base class for unconditional causal kernels (no input variables).
    """

    def forward(self, n=None):
        """
        :param n: Number of samples to generate. If None, generates a single sample
        :returns: Sampled normal values
        """
        if n is not None:
            with pyro.plate(f"{self.outcome_name}_plate", n, dim=-2):
                return self.distribution()
        else:
            return self.distribution()


class CausalKernelUnconditionalCategorical(CausalKernelUnconditional):
    """
    Unconditional kernel for categorical outcomes.
    """

    def __init__(self, *args, **kwargs):
        """
        :raises ValueError: If outcome schema is not categorical
        """
        super().__init__(*args, **kwargs)

        outcome_schema_elem = list(self.outcome_schema.values())[0]
        if not isinstance(outcome_schema_elem, Cat):
            raise ValueError("Expected categorical outcome", self.outcome_schema)

        self.distribution = NeuralDistributionUnconditionalCategorical(
            outcome_name=self.outcome_name,
            num_outcome_cat=outcome_schema_elem.n_cat,
            event_dim=self.event_dim,
        )


class CausalKernelUnconditionalNormal(CausalKernelUnconditional):
    """
    Unconditional kernel for normal (Gaussian) outcomes.
    """

    def __init__(self, *args, **kwargs):
        """
        :raises ValueError: If outcome schema is not of type Norm
        """
        super().__init__(*args, **kwargs)

        outcome_schema_elem = list(self.outcome_schema.values())[0]
        if not isinstance(outcome_schema_elem, Norm):
            raise ValueError(
                "Expected Norm schema for the outcome", self.outcome_schema
            )

        self.distribution = NeuralDistributionUnconditionalNormal(
            outcome_name=self.outcome_name, event_dim=self.event_dim
        )


class CausalKernelUnconditionalFlow(CausalKernelUnconditional):
    """
    Unconditional kernel for outcomes modeled by a normalizing flow.

    :ivar dim: Dimensionality of the flow
    :ivar component_feature_names: Names of individual component features
    :ivar flow: SplineFlow module that builds the trainable distribution by stacking spline transformations
    :ivar dist: The transformed distribution produced by the flow, compatible with Pyro sampling
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: Positional arguments passed to parent class
        :param kwargs: Keyword arguments passed to parent class, including:
            - coupling: Whether to use coupling layers (default: True)
            - count_bins: Number of bins for spline transformations (default: [16, 16])
            - bound: Bound for the spline range (default: 3.0)
        :raises ValueError: If outcome schema is not of type Flow
        """
        super().__init__(*args, **kwargs)

        outcome_schema_elem = list(self.outcome_schema.values())[0]
        if not isinstance(outcome_schema_elem, Flow):
            raise ValueError(
                "Expected Flow schema for the outcome", self.outcome_schema
            )

        self.dim = len(outcome_schema_elem.component_feature_names)
        self.component_feature_names = outcome_schema_elem.component_feature_names

        self.flow = SplineFlow(
            dim=self.event_dim,
            coupling=kwargs.get("coupling", True),
            count_bins=kwargs.get("count_bins", [16, 16]),
            bound=kwargs.get("bound", 3.0),
        )

        self.dist = self.flow()

    def forward(self, n=None):
        """
        :param n: Number of samples to generate. If None, generates a single sample
        :returns: Sampled values from the flow
        """
        self.dist = self.flow()  # need to access it to clear cache in training,
        # see test_causal_kernels
        if n is not None:
            with pyro.plate(f"{self.outcome_name}_plate", n, dim=-2):
                y = pyro.sample(self.outcome_name, self.dist)
        else:
            y = pyro.sample(self.outcome_name, self.dist)

        for i, name in enumerate(self.component_feature_names):
            pyro.deterministic(f"{name}", y[..., i].unsqueeze(-1))

        return y


class CausalKernelConditional(CausalKernel):
    """
    Base class for conditional causal kernels (with input variables).

    :ivar input_schema: Schema defining the input (parent) variables
    :ivar continuous_input_dims: Dictionary mapping variable names to their dimensions
    :ivar data_embedder: Module for embedding categorical inputs
    :ivar in_channels: Total dimensionality of embedded inputs
    :ivar neural_distribution: The conditional distribution module
    """

    input_schema: Schema

    def __init__(
        self,
        input_schema: Schema,
        outcome_schema: Schema,
        *,
        cat_embed_dims: dict[str, int] | None = None,
    ):
        """
        :param input_schema: Schema defining the input (parent) variables
        :param outcome_schema: Schema defining the outcome variable
        :param cat_embed_dims: Dictionary mapping categorical variable names to
            their embedding dimensions. If None, defaults to dimension 1 for all
        :raises ValueError: If outcome schema has multiple elements
        """
        super().__init__(outcome_schema=outcome_schema)

        if len(outcome_schema) != 1:
            raise ValueError("Output schema has multiple elements", outcome_schema)

        self.input_schema = input_schema

        # potentially many-dimensional events
        self.continuous_input_dims = {
            k: len(input_schema[k].component_feature_names)
            for (k, v) in input_schema.items()
            if not isinstance(v, Cat)
        }

        if cat_embed_dims is None:
            cat_embed_dims = {
                k: 1 for (k, v) in input_schema.items() if isinstance(v, Cat)
            }
        input_cat_cardinalities = {
            k: v.n_cat for (k, v) in input_schema.items() if isinstance(v, Cat)
        }
        self.data_embedder = DataEmbedder(
            input_cat_cardinalities=input_cat_cardinalities,
            cat_embed_dims=cat_embed_dims,
        )

        # Categorical variables may be
        # embedded into more dimensions.
        self.in_channels = sum(
            cat_embed_dims[k] if isinstance(v, Cat) else self.continuous_input_dims[k]
            for (k, v) in input_schema.items()
        )

        def _not_implemented(*args, **kwargs):
            """Placeholder for neural_distribution until a subclass provides one.

            :raises NotImplementedError: Always, since subclasses must implement
                neural_distribution.
            """
            raise NotImplementedError("Subclasses must implement neural_distribution")

        self.neural_distribution = _not_implemented

    def forward(self, **kwargs):
        """
        :param kwargs: Input features as keyword arguments matching input_schema keys
        :returns: Sampled values from the conditional distribution
        """
        embedded_x = self.data_embedder(**kwargs)
        return self.neural_distribution(embedded_x)


class CausalKernelConditionalNormal(CausalKernelConditional):
    """
    Conditional kernel for normal (Gaussian) outcomes.

    :ivar neural_distribution: The conditional normal distribution module
    """

    def __init__(
        self,
        *args,
        loc_neural_net: torch.nn.Module | None = None,
        scale_neural_net: torch.nn.Module | None = None,
        **kwargs,
    ):
        """
        :param args: Positional arguments passed to parent class
        :param loc_neural_net: Optional custom neural net for the location parameter
        :param scale_neural_net: Optional custom neural net for the scale parameter
        :param kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

        self.neural_distribution = NeuralDistributionConditionalNormal(
            outcome_name=self.outcome_name,
            in_channels=self.in_channels,
            event_dim=self.event_dim,
            loc_neural_net=loc_neural_net,
            scale_neural_net=scale_neural_net,
        )


class CausalKernelConditionalCategorical(CausalKernelConditional):
    """
    Conditional kernel for categorical outcomes.

    :ivar neural_distribution: The conditional categorical distribution module
    """

    def __init__(self, *args, neural_net: torch.nn.Module | None = None, **kwargs):
        """
        :param args: Positional arguments passed to parent class
        :param neural_net: Optional custom neural net for the logits
        :param kwargs: Keyword arguments passed to parent class
        :raises ValueError: If outcome schema is not categorical
        """
        super().__init__(*args, **kwargs)

        outcome_schema_elem = list(self.outcome_schema.values())[0]
        if not isinstance(outcome_schema_elem, Cat):
            raise ValueError("Expected categorical output", self.outcome_schema)

        self.neural_distribution = NeuralDistributionConditionalCategorical(
            outcome_name=self.outcome_name,
            num_outcome_cat=outcome_schema_elem.n_cat,
            in_channels=self.in_channels,
            neural_net=neural_net,
        )


class StructuredModel(pyro.nn.PyroModule):
    """
    Structured model composed of multiple causal kernels.
    """

    def _get_kernels(self) -> Mapping[str, "CausalKernel"]:
        """Return the kernel mapping, preferring kernel_modules (new arch).

        # TODO: remove kernels fallback when legacy upstream models go away
        """
        return getattr(self, "kernel_modules", None) or getattr(self, "kernels", {})

    @property
    def device(self):
        """The device on which the model's parameters reside.

        :returns: The device of the first model parameter.
        """
        return next(self.parameters()).device

    def __init__(self, causal_kernels: Collection[CausalKernel]):
        """
        :param causal_kernels: Collection of causal kernels defining the model
        """
        super().__init__()

        kernel_dict = {k.outcome_name: k for k in causal_kernels}

        graph = {
            k: (
                list(v.input_schema.keys())
                if isinstance(v, CausalKernelConditional)
                else []
            )
            for k, v in kernel_dict.items()
        }

        topo_order = list(graphlib.TopologicalSorter(graph).static_order())
        ordered_kernels = {k: kernel_dict[k] for k in topo_order if k in kernel_dict}

        self.graph = graph
        self.topo_ordering = topo_order
        self.kernel_modules = torch.nn.ModuleDict(ordered_kernels)

    def forward(
        self, n: int | None = None, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        This method performs a forward pass through all kernels in topological
        order. For each kernel:

        1. Conditional kernels receive their required parent variables
        2. Unconditional kernels are sampled independently
        3. Flow kernels register component features as separate variables

        The method ensures all dependencies are satisfied before sampling
        each variable.

        :param n: Number of samples to generate. Required for unconditional
            kernels, ignored for conditional kernels
        :param kwargs: Pre-specified values for certain variables (for interventions
            or conditioning). These values will not be sampled
        :returns: Dictionary mapping variable names to their sampled values
        :raises ValueError: If a kernel output is passed as input, or if
            required dependencies are missing
        :raises TypeError: If an unknown kernel type is encountered
        """
        _kernels = self._get_kernels()

        # check that no kernel output was passed in as input
        unexpected_key = next((k for k in kwargs if k in _kernels), None)
        if unexpected_key is not None:
            raise ValueError("Passed kernel output as input parameter", unexpected_key)

        # will need to add flow components later
        outcome_keys = list(_kernels.keys())

        for outcome_name, kernel in _kernels.items():
            # Conditional kernels: use input_schema to gather arguments
            if isinstance(kernel, CausalKernelConditional):
                kernel_kwargs = {k: kwargs[k] for k in kernel.input_schema.keys()}
                outcome = kernel(**kernel_kwargs)

            elif isinstance(kernel, CausalKernelUnconditionalFlow):
                outcome = kernel(n=n)
                for i, name in enumerate(kernel.component_feature_names):
                    if name in kwargs:
                        raise ValueError(
                            "Passed kernel output as input parameter", name
                        )
                    kwargs[name] = outcome[..., i].unsqueeze(-1)

                    outcome_keys.append(name)

            # Unconditional kernels: ignore input_schema, use n
            elif isinstance(kernel, CausalKernelUnconditional):
                outcome = kernel(n=n)

            else:
                raise TypeError(
                    f"Unknown kernel type for {outcome_name}: {type(kernel)}"
                )

            kwargs[outcome_name] = outcome

        return {name: kwargs[name] for name in outcome_keys}
