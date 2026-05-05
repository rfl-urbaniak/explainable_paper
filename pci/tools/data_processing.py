import graphlib
import logging
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONT_DISTS_1D = ["normal", "mixnormal"]
CAT_DISTS_1D = ["categorical"]
CONT_DISTS_ND = ["multivariate_normal", "flow"]

# Specification suffixes are slightly different from the ones in processed data
SUFFIX_MAPPING = MappingProxyType(
    {
        "log-epsilon": "log_epsilon",
        "minmax": "minmax",
        "standardize": "std",
        "reset_index": "reind",
    }
)


@dataclass
class TransformParams:
    """Base class for all transformation parameters."""

    pass


@dataclass
class StandardizeParams(TransformParams):
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class MinMaxParams(TransformParams):
    min_val: torch.Tensor
    range_val: torch.Tensor
    max_val: torch.Tensor


@dataclass
class LogEpsilonParams(TransformParams):
    epsilon: float
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class IdentityParams(TransformParams):
    """Parameters for the identity (no-op) transformation.  No fields needed."""

    pass


@dataclass
class CategoricalParams(TransformParams):
    mapping: dict[Any, int]
    n_cat: int


def strip_suffix(x: str) -> str:
    """Helper function to strip data preprocessing suffixes from a string.

    >>> strip_suffix("height_log_epsilon")
    'height'
    >>> strip_suffix("height_minmax")
    'height'
    >>> strip_suffix("height_std")
    'height'
    >>> strip_suffix("height_reind")
    'height'

    :param x: String to strip suffixes from
    :returns: String with suffixes stripped
    """
    for suffix in SUFFIX_MAPPING.values():
        if x.endswith(suffix):
            return x[: -(len(suffix) + 1)]
    return x


# Standardize


def standardize_tensor(
    tensor: torch.Tensor,
    epsilon: float = 1e-6,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    target_device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    if mean is None:
        mean = tensor.mean(dim=0)
    if std is None:
        std = tensor.std(dim=0)
    mean = mean.to(target_device)
    std = std.to(target_device)
    std = torch.maximum(std, std.new_tensor(epsilon))
    standardized_tensor = (tensor.to(target_device) - mean) / std
    return standardized_tensor, {"mean": mean, "std": std}


# Destandardize


def destandardize_tensor(
    tensor: torch.Tensor,
    params: dict | StandardizeParams | LogEpsilonParams,
    target_device: torch.device | str = "cpu",
) -> torch.Tensor:
    mean = params["mean"] if isinstance(params, dict) else params.mean
    std = params["std"] if isinstance(params, dict) else params.std
    mean_tensor = (
        tensor.new_tensor(mean)
        if not isinstance(mean, torch.Tensor)
        else mean.to(target_device)
    )
    std_tensor = (
        tensor.new_tensor(std)
        if not isinstance(std, torch.Tensor)
        else std.to(target_device)
    )
    destandardized_tensor = tensor.to(target_device) * std_tensor + mean_tensor
    return destandardized_tensor


# Minmax


def min_max_scale_tensor(
    tensor: torch.Tensor,
    epsilon: float = 1e-6,
    min_val: torch.Tensor | None = None,
    max_val: torch.Tensor | None = None,
    target_device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    if min_val is None:
        min_val = tensor.min(dim=0, keepdim=True)[0]
    if max_val is None:
        max_val = tensor.max(dim=0, keepdim=True)[0]
    min_val = min_val.to(target_device)
    max_val = max_val.to(target_device)
    range_val = max_val - min_val
    range_val = torch.maximum(range_val, range_val.new_tensor(epsilon))
    scaled_tensor = (tensor.to(target_device) - min_val) / range_val
    return scaled_tensor, {
        "min_val": min_val,
        "range_val": range_val,
        "max_val": max_val,
    }


def descale_tensor(
    tensor: torch.Tensor,
    params: dict | MinMaxParams,
    target_device: torch.device | str = "cpu",
) -> torch.Tensor:
    min_val = params["min_val"] if isinstance(params, dict) else params.min_val
    range_val = params["range_val"] if isinstance(params, dict) else params.range_val
    min_val = (
        min_val.to(target_device) if isinstance(min_val, torch.Tensor) else min_val
    )
    range_val = (
        range_val.to(target_device)
        if isinstance(range_val, torch.Tensor)
        else range_val
    )
    descaled_tensor = tensor.to(target_device) * range_val + min_val
    return descaled_tensor


def transform_log_epsilon(
    tensor: torch.Tensor,
    epsilon: float = 1e-6,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    target_device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Applies a log(epsilon) transformation to a scalar tensor, then standardizes the result,
    i.e., x -> y = log(x + epsilon) -> (y - mean_y) / std_y.

    :param tensor: Tensor to transform
    :param epsilon: Minimum value to add to the tensor. Defaults to 1e-6
    :param mean: Mean to use for standardization. Defaults to None
    :param std: Std to use for standardization. Defaults to None
    :param target_device: Device to use for computation. Defaults to module-level device
    :returns: Transformed tensor and parameters
    """
    # assert (tensor + epsilon) is positive
    if (tensor + epsilon).min() <= 0:
        raise ValueError(
            "Tensor values must be greater than -epsilon to apply log transformation."
        )
    log_transformed = torch.log(tensor + epsilon)
    standardized, standardization_params = standardize_tensor(
        log_transformed, mean=mean, std=std, target_device=target_device
    )
    return standardized, {"epsilon": epsilon, **standardization_params}


def inverse_transform_log_epsilon(
    tensor: torch.Tensor,
    params: dict | LogEpsilonParams,
    target_device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Reverses the log(epsilon) transformation and standardization.

    :param tensor: Tensor to inverse transform
    :param params: Parameters from transform_log_epsilon
    :param target_device: Device to use for computation. Defaults to module-level device
    :returns: Inverse transformed tensor
    """
    destandardized_tensor = destandardize_tensor(
        tensor, params, target_device=target_device
    )

    epsilon = params["epsilon"] if isinstance(params, dict) else params.epsilon
    unstandardized = destandardized_tensor
    return torch.exp(unstandardized) - epsilon


def forward_transform(values: torch.Tensor, params: TransformParams) -> torch.Tensor:
    """Apply a fitted :class:`TransformParams` to a tensor of raw values.

    This is the canonical way to apply an already-fitted transform to new data.
    Returns a ``float32`` tensor for continuous params and an ``int64`` tensor
    for :class:`CategoricalParams`.  Unknown keys in :class:`CategoricalParams`
    map to ``-1`` so callers can detect and handle out-of-vocabulary values.

    :param values: Tensor of raw values (any shape).
    :param params: Fitted transform parameters.
    :returns: Transformed tensor on the same device as *values*.
    :raises TypeError: If *params* is not a recognised :class:`TransformParams` subclass.

    >>> import torch
    >>> from pci.tools.data_processing import forward_transform, StandardizeParams
    >>> p = StandardizeParams(mean=torch.tensor(3.0), std=torch.tensor(2.0))
    >>> forward_transform(torch.tensor([1.0, 3.0, 5.0]), p)
    tensor([-1.,  0.,  1.])
    """
    dev = values.device

    if isinstance(params, LogEpsilonParams):
        log_vals = torch.log(values.double() + params.epsilon)
        return (
            (log_vals - params.mean.to(dev).double()) / params.std.to(dev).double()
        ).float()

    if isinstance(params, StandardizeParams):
        return (
            (values.double() - params.mean.to(dev).double())
            / params.std.to(dev).double()
        ).float()

    if isinstance(params, MinMaxParams):
        return (
            (values.double() - params.min_val.to(dev).double())
            / params.range_val.to(dev).double()
        ).float()

    if isinstance(params, IdentityParams):
        return values.float()

    if isinstance(params, CategoricalParams):
        # Dict lookup on CPU; result moved back to original device.
        arr = np.array(
            [params.mapping.get(v, -1) for v in values.cpu().numpy().flat],
            dtype=np.int64,
        ).reshape(values.shape)
        return torch.from_numpy(arr).to(dev)

    raise TypeError(
        f"forward_transform: unsupported params type '{type(params).__name__}'. "
        "Expected one of: LogEpsilonParams, StandardizeParams, MinMaxParams, "
        "IdentityParams, CategoricalParams."
    )


def get_inverse_transform(transform_type: str | TransformParams):
    """Return the inverse-transform callable for a given continuous transform type.

    Parameters
    ----------
    transform_type:
        Either the string key used in ``continuous_transformations`` dicts
        (e.g. ``"log-epsilon"``) or a :class:`TransformParams` instance whose
        concrete type determines the transform (e.g. ``LogEpsilonParams``).

    Returns
    -------
    Callable ``(tensor, params) -> tensor`` that reverses the transform.

    Raises
    ------
    ValueError
        If *transform_type* is not recognised.
    """
    _params_to_key: dict[type, str] = {
        IdentityParams: "identity",
        LogEpsilonParams: "log-epsilon",
        StandardizeParams: "standardize",
        MinMaxParams: "minmax",
    }
    _registry = {
        "identity": lambda v, _: v,
        "log-epsilon": inverse_transform_log_epsilon,
        "standardize": destandardize_tensor,
        "minmax": descale_tensor,
    }
    if isinstance(transform_type, TransformParams):
        key = _params_to_key.get(type(transform_type))
        if key is None:
            raise ValueError(
                f"No inverse transform registered for {type(transform_type).__name__}."
            )
        return _registry[key]
    if transform_type not in _registry:
        raise ValueError(
            f"Unknown transform type {transform_type!r}. "
            f"Known types: {sorted(_registry)}"
        )
    return _registry[transform_type]


def reset_categorical_index(
    tensor: torch.Tensor,
    mapping: dict | None = None,
    other_values: list | None = None,
    other_token: str = "__OTHER__",
) -> tuple[torch.Tensor, dict]:
    """Reindex a categorical tensor to contiguous integers starting at 0, with support for an 'OTHER' category.

    Previously unseen values encountered in `tensor` are handled as follows:
    - If `other_values` is empty or None, all unseen values are mapped to `other_token`.
    - If `other_values` is non-empty, all unseen values and all values listed in `other_values` are mapped to `other_token`.
    - If `mapping` is missing `other_token`, a KeyError is raised for unseen values.

    :param tensor: Tensor containing categorical values.
    :param mapping: Optional existing mapping from category to index. If None, a new mapping is created.
    :param other_values: Optional list of values to force into OTHER. Defaults to None.
    :param other_token: The token used for the OTHER category. Defaults to "__OTHER__".
    :returns: A tuple containing
        - the reindexed tensor with the same shape as `tensor`, on the same device as the input
        - the mapping used (dict of original values and `other_token` to indices)
    :raises KeyError: If an unseen category is encountered in `tensor` and cannot be mapped to `other_token`.
    """
    original_shape = tensor.shape
    out_device = tensor.device
    # Always process on CPU — the LUT can be huge (e.g. FIPS code range) and
    # must not be allocated on GPU. The result is moved back to out_device at the end.
    tensor = tensor.long().flatten().cpu()
    other_values_set = set(other_values or [])

    # ---- Setup mapping ----
    if mapping is None:
        # Identify categories excluding forced OTHER values
        unique_vals = sorted(set(tensor.tolist()) - other_values_set)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}

        # Add OTHER token
        mapping[other_token] = len(mapping)
        other_index = mapping[other_token]
    else:
        if other_token in mapping:
            other_index = mapping[other_token]
        else:
            other_index = None
            logger.warning(
                f"Mapping does not include '{other_token}'. "
                "The mapping will be unable to handle unseen values."
            )

    if other_index is None:
        other_index = -1

    min_cat = min(set(mapping.keys()) - {other_token})
    max_cat = max(set(mapping.keys()) - {other_token})
    lut = torch.full((max_cat - min_cat + 1,), other_index, device=torch.device("cpu"))
    for orig, new in mapping.items():
        if orig == other_token:
            continue
        lut[orig - min_cat] = new

    out_of_range_mask = torch.logical_or(tensor < min_cat, tensor > max_cat)
    reindexed = tensor.clone().detach()
    reindexed[out_of_range_mask] = min_cat
    reindexed = lut[reindexed - min_cat]
    reindexed[out_of_range_mask] = other_index
    reindexed = reindexed.reshape(original_shape).to(out_device)
    if (reindexed < 0).any():
        raise ValueError("Unexpected out of range value in categorical tensor")

    return reindexed, mapping


def unmap_categorical_index(tensor: torch.Tensor, mapping: dict) -> torch.Tensor:
    """
    Reverses the reindexing of a categorical tensor to its original values.

    :param tensor: Reindexed tensor with contiguous integers starting from 0.
    :param mapping: Dictionary mapping original values to indices used in
                    `reset_categorical_index`.
    :return: Tensor with original categorical values restored.

    Example
    -------
    >>> import torch
    >>> t = torch.tensor([0, 1, 0])
    >>> mapping = {100: 0, 200: 1}
    >>> unmap_categorical_index(t, mapping)
    tensor([100, 200, 100])

    .. note::
        The function assumes that the mapping dictionary might contain a special key
        ``__OTHER__`` for out-of-vocabulary values, which will be mapped to
        ``max_original_value + 1``.
    """
    numeric_keys = [k for k in mapping.keys() if k != "__OTHER__"]

    max_numeric_key = max(
        [k for k in numeric_keys if isinstance(k, (int, np.integer))], default=-1
    )

    max_val = max(mapping.values())
    lookup = np.full(max_val + 1, fill_value=max_numeric_key + 1, dtype=np.int64)

    for k, v in mapping.items():
        if k == "__OTHER__":
            logger.warning(
                f"Categorical unmapping '__OTHER__' to {max_numeric_key + 1} instead of {v}."
            )
        else:
            lookup[v] = k

    raw_tensor = torch.as_tensor(lookup, device=tensor.device)
    return raw_tensor[tensor.long()]


def invert_all_features(
    feature_dict: dict[str, torch.Tensor],
    transformation_params: dict[str, dict],
    continuous_transformations: dict[str, str | TransformParams],
    clip_raw_data_dict: dict[str, tuple[float, float]] | None = None,
) -> dict[str, torch.Tensor]:
    raw_dict = {}
    for key in feature_dict.keys():
        transformed_value = feature_dict[key]
        transformation_type = (
            continuous_transformations[key]
            if key in continuous_transformations
            else "reset_index"
        )
        params = transformation_params[f"{key}_transformed"]
        if transformation_type == "reset_index":
            mapping = (
                params.mapping if hasattr(params, "mapping") else params["mapping"]
            )
            raw_tensor = unmap_categorical_index(transformed_value, mapping)
        else:
            raw_tensor = get_inverse_transform(transformation_type)(
                transformed_value, params
            )

        raw_dict[key] = raw_tensor

        if clip_raw_data_dict and (key in continuous_transformations.keys()):
            min_val = clip_raw_data_dict[key][0]
            max_val = clip_raw_data_dict[key][1]
            min_val_tensor = (
                torch.tensor(min_val, dtype=raw_tensor.dtype, device=raw_tensor.device)
                if not isinstance(min_val, torch.Tensor)
                else min_val.to(raw_tensor.device)
            )
            max_val_tensor = (
                torch.tensor(max_val, dtype=raw_tensor.dtype, device=raw_tensor.device)
                if not isinstance(max_val, torch.Tensor)
                else max_val.to(raw_tensor.device)
            )
            raw_dict[key] = torch.clamp(
                raw_tensor, min=min_val_tensor, max=max_val_tensor
            )

    return raw_dict


def data_to_transformed_tensors(
    inference_df: pd.DataFrame,
    continuous_features: dict[str, str],
    categorical_features: dict[str, str],
    transformation_params: dict[str, TransformParams] | None = None,
    ensure_shapes: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Transforms a DataFrame into tensors and keeps track of transformation parameters
    using structured dataclasses.
    """
    data_dict: dict[str, Any] = {}

    # Use provided params or initialize empty
    data_dict["transformation_params"] = transformation_params or {}
    data_dict["continuous_features"] = continuous_features
    data_dict["categorical_features"] = categorical_features

    cols = inference_df.columns.tolist()

    # Convert raw columns to tensors — always on CPU during transformation;
    # moved to the target device at the end of this function.
    data_dict["continuous"] = {
        col: torch.tensor(inference_df[col].to_numpy()).float()
        for col in continuous_features
        if col in cols
    }
    data_dict["categorical"] = {
        col: torch.tensor(inference_df[col].to_numpy()).long()
        for col in categorical_features
        if col in cols
    }

    data_dict["transformation_params"] = {
        f"{k}_transformed": data_dict["transformation_params"][f"{k}_transformed"]
        for k in cols
        if f"{k}_transformed" in data_dict["transformation_params"]
    }
    data_dict["continuous_features"] = {
        k: v for k, v in continuous_features.items() if k in cols
    }
    data_dict["categorical_features"] = {
        k: v for k, v in categorical_features.items() if k in cols
    }

    # --- Continuous features ---
    for column, method in data_dict["continuous_features"].items():
        new_col = f"{column}_transformed"

        if method == "standardize":
            existing_params_standardize: StandardizeParams | None = data_dict[
                "transformation_params"
            ].get(new_col)
            if existing_params_standardize:
                transformed = forward_transform(
                    data_dict["continuous"][column], existing_params_standardize
                )
                params_standardize = existing_params_standardize
            else:
                transformed, params_dict = standardize_tensor(
                    data_dict["continuous"][column],
                    target_device=torch.device("cpu"),
                )
                params_standardize = StandardizeParams(**params_dict)

            data_dict["continuous"][new_col] = transformed
            data_dict["transformation_params"][new_col] = params_standardize
            logger.debug(
                f"Standardized {column} with mean={params_standardize.mean}, std={params_standardize.std}"
            )

        elif method == "minmax":
            existing_params_minmax: MinMaxParams | None = data_dict[
                "transformation_params"
            ].get(new_col)
            if existing_params_minmax:
                transformed = forward_transform(
                    data_dict["continuous"][column], existing_params_minmax
                )
                params_minmax = existing_params_minmax
            else:
                transformed, params_dict = min_max_scale_tensor(
                    data_dict["continuous"][column],
                    target_device=torch.device("cpu"),
                )
                params_minmax = MinMaxParams(**params_dict)

            data_dict["continuous"][new_col] = transformed
            data_dict["transformation_params"][new_col] = params_minmax
            logger.debug(
                f"MinMax scaled {column} with min={params_minmax.min_val}, range={params_minmax.range_val}"
            )

        elif method == "log-epsilon":
            existing_params_log_epsilon: LogEpsilonParams | None = data_dict[
                "transformation_params"
            ].get(new_col)
            if existing_params_log_epsilon:
                transformed = forward_transform(
                    data_dict["continuous"][column], existing_params_log_epsilon
                )
                params_log_epsilon = existing_params_log_epsilon
            else:
                transformed, params_dict = transform_log_epsilon(
                    data_dict["continuous"][column],
                    target_device=torch.device("cpu"),
                )
                params_log_epsilon = LogEpsilonParams(**params_dict)

            data_dict["continuous"][new_col] = transformed
            data_dict["transformation_params"][new_col] = params_log_epsilon
            logger.debug(f"Applied log-epsilon to {column}")

        elif method == "identity":
            data_dict["continuous"][new_col] = data_dict["continuous"][column]
            data_dict["transformation_params"][new_col] = IdentityParams()
            logger.debug(f"Applied identity (no-op) to {column}")

        else:
            raise ValueError(
                f"Unknown continuous transformation '{method}' for {column}"
            )

    # --- Categorical features ---
    for column, method in data_dict["categorical_features"].items():
        new_col = f"{column}_transformed"

        if method == "reset_index":
            existing_params_categorical: CategoricalParams | None = data_dict[
                "transformation_params"
            ].get(new_col)
            if existing_params_categorical:
                data_dict["categorical"][new_col], _mapping = reset_categorical_index(
                    data_dict["categorical"][column],
                    mapping=existing_params_categorical.mapping,
                )
                params_categorical = existing_params_categorical
            else:
                data_dict["categorical"][new_col], _mapping = reset_categorical_index(
                    data_dict["categorical"][column],
                )
                params_categorical = CategoricalParams(
                    mapping=_mapping, n_cat=len(_mapping)
                )

            data_dict["transformation_params"][new_col] = params_categorical
            logger.debug(
                f"Reindexed {column} with mapping={params_categorical.mapping}"
            )

        else:
            raise ValueError(
                f"Unknown categorical transformation '{method}' for {column}"
            )

        if ensure_shapes:
            # make (batch, 1, 1) shape

            for key, tensor in data_dict["categorical"].items():
                data_dict["categorical"][key] = tensor.view(-1, 1, 1)
            for key, tensor in data_dict["continuous"].items():
                data_dict["continuous"][key] = tensor.view(-1, 1, 1)

    # Move all tensors to the target device only after all CPU transformations are done.
    out_device = torch.device(device)
    for key in data_dict["continuous"]:
        data_dict["continuous"][key] = data_dict["continuous"][key].to(out_device)
    for key in data_dict["categorical"]:
        data_dict["categorical"][key] = data_dict["categorical"][key].to(out_device)

    return data_dict


def enrich_data_dict_with_nd_tensors(
    data_dict: dict, input_feature_map: dict, device: torch.device | str = "cpu"
):
    for key in input_feature_map:
        feature_list = input_feature_map[key]
        tensors = [
            data_dict["continuous"][f"{feature}_transformed"]
            for feature in feature_list
        ]
        data_dict["continuous"][f"{key}_transformed"] = torch.cat(tensors, dim=-1)

    return data_dict


def produce_dictionary_for_inference(
    inference_df: pd.DataFrame,
    continuous_features: dict,
    categorical_features: dict,
    transformation_params: dict | None = None,
) -> dict:
    """
    Transforms a pd.DataFrame into a dictionary of dictionaries tensors for model inference,
    applying specified transformations to continuous and categorical features, keeping these groups separate.

    :param inference_df: The input DataFrame containing feature columns
    :param continuous_features: A dictionary mapping continuous feature names to transformation methods. Supported methods include:

        - "standardize": standardizes values (zero mean, unit std)
        - "minmax": scales values to [0, 1]
        - "log-epsilon": applies log(x + epsilon) followed by standardization

    :param categorical_features: A dictionary mapping categorical feature names to transformation methods. Currently supported:

        - "reset_index": maps unique values to contiguous integers starting from 0

    :param transformation_params: Parameters for transformations
    :returns: A dictionary with the following structure:

    .. code-block::

        {
            "continuous": {
                <original_feature_name>: torch.Tensor,
                <transformed_feature_name>: torch.Tensor,
                ...
            },
            "categorical": {
                <original_feature_name>: torch.Tensor,
                <reindexed_feature_name>: torch.Tensor,
                ...
            },
            "transformation_params": {
                <feature_name>: {
                    ...  # parameters needed to reverse the transformation
                },
                ...
            }
        }

    Notes:
        - Transformed features are added to the corresponding sub-dictionaries under new post-fixed keys (e.g., "height_std", "weight_log_epsilon")
        - All tensors are of type torch.float (for continuous) or torch.long (for categorical)
        - The transformation parameters (e.g., mean, std, min, range, mappings) are stored under the "transformation_params" key to support inverse transformations
        - Original tensors are preserved for easier debugging or visualization
    """

    warnings.warn(
        "produce_dictionary_for_inference is deprecated. Use data_to_transformed_tensors instead.",
        DeprecationWarning,
    )

    def _cpu(d: dict) -> dict:
        """Normalise tensor values to CPU for device-agnostic dict comparison."""
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    data_dict: dict[str, dict] = {}

    if transformation_params is None:
        data_dict["transformation_params"] = {}
    else:
        data_dict["transformation_params"] = transformation_params

    data_dict["continuous_features"] = continuous_features
    data_dict["categorical_features"] = categorical_features

    data_dict["continuous"] = {
        column: torch.tensor(inference_df[column].to_numpy()).float()
        for column in continuous_features.keys()
    }
    data_dict["categorical"] = {
        column: torch.tensor(inference_df[column].to_numpy()).long()
        for column in categorical_features.keys()
    }

    for column, method in continuous_features.items():
        new_col = f"{column}_{SUFFIX_MAPPING.get(method, method)}"

        if method == "standardize":
            if existing_params := data_dict["transformation_params"].get(new_col):
                transformed, params = standardize_tensor(
                    data_dict["continuous"][column],
                    mean=existing_params["mean"],
                    std=existing_params["std"],
                )
                assert _cpu(existing_params) == _cpu(params), (
                    "Existing parameters do not match new parameters"
                )
            else:
                transformed, params = standardize_tensor(
                    data_dict["continuous"][column]
                )

            data_dict["continuous"][new_col] = transformed
            data_dict["transformation_params"][new_col] = params
            logger.debug(
                f"Standardized {column} with mean: {params['mean']}, std: {params['std']}"
            )
            assert transformed.shape == data_dict["continuous"][column].shape

        elif method == "minmax":
            if existing_params := data_dict["transformation_params"].get(new_col):
                transformed, params = min_max_scale_tensor(
                    data_dict["continuous"][column],
                    min_val=existing_params["min_val"],
                    max_val=existing_params["max_val"],
                )
                assert _cpu(existing_params) == _cpu(params), (
                    "Existing parameters do not match new parameters"
                )
            else:
                transformed, params = min_max_scale_tensor(
                    data_dict["continuous"][column]
                )

            data_dict["continuous"][new_col] = transformed
            data_dict["transformation_params"][new_col] = params
            logger.debug(
                f"MinMax scaled {column} with min: {params['min_val']}, range: {params['range_val']}"
            )
            assert transformed.shape == data_dict["continuous"][column].shape

        elif method == "log-epsilon":
            if existing_params := data_dict["transformation_params"].get(new_col):
                transformed, params = transform_log_epsilon(
                    data_dict["continuous"][column],
                    epsilon=existing_params["epsilon"],
                    mean=existing_params["mean"],
                    std=existing_params["std"],
                )
                assert _cpu(existing_params) == _cpu(params), (
                    "Existing parameters do not match new parameters"
                )
            else:
                transformed, params = transform_log_epsilon(
                    data_dict["continuous"][column]
                )

            data_dict["continuous"][new_col] = transformed
            data_dict["transformation_params"][new_col] = params
            logger.debug(f"Applied log(epsilon) transformation to {column}")
            assert transformed.shape == data_dict["continuous"][column].shape

        else:
            raise ValueError(
                f"Unknown transformation method '{method}' for column '{column}'"
            )

    for column, method in categorical_features.items():
        new_col = f"{column}_{SUFFIX_MAPPING.get(method, method)}"

        if method == "reset_index":
            if existing_params := data_dict["transformation_params"].get(new_col):
                data_dict["categorical"][new_col], _mapping = reset_categorical_index(
                    data_dict["categorical"][column],
                    mapping=existing_params["mapping"],
                )
                assert existing_params["mapping"] == _mapping, (
                    "Existing parameters do not match new parameters"
                )
                assert existing_params["n_cat"] == len(_mapping), (
                    "Existing parameters do not match new parameters"
                )
            else:
                data_dict["categorical"][new_col], _mapping = reset_categorical_index(
                    data_dict["categorical"][column]
                )

            n_cat = len(_mapping)
            logger.debug(f"Reindexed {column} with mapping: {_mapping}")

            assert (
                data_dict["categorical"][new_col].shape
                == data_dict["categorical"][column].shape
            )
            data_dict["transformation_params"][new_col] = {
                "mapping": _mapping,
                "n_cat": n_cat,
            }
        else:
            raise ValueError(
                f"Unknown transformation method '{method}' for column '{column}'"
            )

    return data_dict


def prepare_features(
    data: pd.DataFrame,
    categorical_var_spec: dict[str, str],
    continuous_var_spec: dict[str, str],
    outcome_name: str = "y",
    transformation_params: dict | None = None,
) -> dict:
    """Prepares features from a DataFrame for model inference. This is a helper function that calls :func:`produce_dictionary_for_inference`, and prepares PyTorch tensors. The outputs are general-purpose, but particularly well-suited for use with :class:`pci.gp.categorical.EmbeddedCategoricalGP`.

    Example Usage:

    .. code-block:: python

        data = pd.DataFrame(
            {
                "CAT_FEATURE_1": [0, 1, 1],
                "CAT_FEATURE_2": [1, 0, 0],
                "CONT_FEATURE_1": [1.0, 2.0, 3.5],
                "CONT_FEATURE_2": [4.0, 5.2, 6.3],
            }
        )
        categorical_var_spec = {"CAT_FEATURE_1": "reset_index", "CAT_FEATURE_2": "reset_index"}
        continuous_var_spec = {"CONT_FEATURE_1": "log-epsilon", "CONT_FEATURE_2": "minmax"}
        outcome_name = "CONT_FEATURE_1"
        features = prepare_features(
            data, categorical_var_spec, continuous_var_spec, outcome_name
        )

    To use precomputed transformation parameters (e.g., at test time), pass them in as a dictionary.

    .. code-block:: python

        transformation_params = {"CAT_FEATURE_1": {"mapping": {0: 0, 1: 1}, "n_cat": 2},
                                 "CAT_FEATURE_2": {"mapping": {0: 0, 1: 1}, "n_cat": 2},
                                 "CONT_FEATURE_1": {"mean": 0, "std": 1},
                                 "CONT_FEATURE_2": {"min_val": 0, "range_val": 1, "max_val": 1},
                                 }
        features = prepare_features(
            data, categorical_var_spec, continuous_var_spec, outcome_name, transformation_params
        )

    :param data: Dataframe to prepare features from
    :param categorical_var_spec: Specification for categorical variables. Should be a dictionary with mappings {column_name: transformation_method}. See :func:`produce_dictionary_for_inference` for supported methods
    :param continuous_var_spec: Specification for continuous variables. Should be a dictionary with mappings {column_name: transformation_method}. See :func:`produce_dictionary_for_inference` for supported methods
    :param outcome_name: Name of the outcome variable. Defaults to "y"
    :param transformation_params: Transformation parameters for each variable, if available. Defaults to None, which computes transformation parameters from the data
    :returns: A dictionary with the following keys:

        - "x_categorical": Categorical features tensor
        - "x_continuous": Continuous features tensor
        - "x": Combined features tensor
        - "y": Outcome tensor
        - "num_embeddings_list": List of number of unique values for each categorical variable
        - "transformation_params": Transformation parameters for each variable
        - "continuous_keys": List of continuous variable names
        - "categorical_keys": List of categorical variable names
    """

    #### Perform transformations ####
    data_dict = produce_dictionary_for_inference(
        data,
        continuous_features=continuous_var_spec,
        categorical_features=categorical_var_spec,
        transformation_params=transformation_params,
    )

    if outcome_name in continuous_var_spec:
        transform = continuous_var_spec[outcome_name]
        source = "continuous"
    elif outcome_name in categorical_var_spec:
        transform = categorical_var_spec[outcome_name]
        source = "categorical"
    else:
        raise ValueError(
            f"{outcome_name} must be in either continuous or categorical var spec"
        )

    if transform not in SUFFIX_MAPPING:
        raise ValueError(
            f"Unknown transformation: method '{transform}' for column '{outcome_name}'"
        )

    outcome_suffix = SUFFIX_MAPPING[transform]
    y_key = f"{outcome_name}_{outcome_suffix}"
    y = data_dict[source][y_key]

    assert y.dim() == 1, "Outcome tensor must be 1-dimensional"
    assert y.shape[0] == data.shape[0], "Outcome tensor length must match data length"

    #### Prepare features ####
    continuous_keys = [
        f"{key}_{SUFFIX_MAPPING[continuous_var_spec[key]]}"
        for key in continuous_var_spec
        if key != outcome_name
    ]

    x_continuous = (
        torch.stack([data_dict["continuous"][key] for key in continuous_keys], dim=1)
        if continuous_keys
        else torch.empty((len(data), 0))
    )

    assert x_continuous.dim() == 2, "Continuous features tensor must be 2-dimensional"
    assert x_continuous.shape[0] == data.shape[0], (
        "Continuous features tensor length must match data length"
    )

    categorical_keys = [
        f"{key}_{SUFFIX_MAPPING[categorical_var_spec[key]]}"
        for key in categorical_var_spec
        if key != outcome_name
    ]

    x_categorical = (
        torch.stack(
            [
                data_dict["categorical"][
                    key
                ]  # directly use keys with suffixes formed earlier
                for key in categorical_keys
            ],
            dim=1,
        )
        if categorical_keys
        else torch.empty((len(data), 0), dtype=torch.long)
    )

    assert x_categorical.dim() == 2, "Categorical features tensor must be 2-dimensional"
    assert x_categorical.shape[0] == data.shape[0], (
        "Categorical features tensor length must match data length"
    )

    x = torch.cat([x_categorical, x_continuous.to(x_categorical.device)], dim=1)
    logger.info(f"Combined features tensor shape: {x.shape}")

    num_embeddings_list = [
        len(
            data_dict["categorical"][
                f"{key}_{SUFFIX_MAPPING[categorical_var_spec[key]]}"
            ].unique()
        )
        for key in categorical_var_spec
        if key != outcome_name
    ]

    return {
        "x_categorical": x_categorical.double(),
        "x_continuous": x_continuous.double(),
        "x": x,
        "y": y.double(),
        "num_embeddings_list": num_embeddings_list,
        "transformation_params": data_dict["transformation_params"],
        "continuous_keys": continuous_keys,
        "categorical_keys": categorical_keys,
        "data_dict": data_dict,
    }


def prepare_conditional_features(
    data: pd.DataFrame,
    categorical_var_spec: dict[str, str],
    continuous_var_spec: dict[str, str],
    likelihood_spec: dict[str, dict[str, str | dict | list]],
    outcome_name: str = "y",
) -> dict:
    """Given the raw data, generate a processed dictionary of the dataset given specifications of conditioning.

    Note: For now, it is assumed that the outcome variable is a terminal node in the graph.

    Example Usage:

    .. code-block:: python

        data = pd.DataFrame(
            {
                "CAT_FEATURE_1": [0, 1, 1],
                "CAT_FEATURE_2": [1, 0, 0],
                "CONT_FEATURE_1": [1.0, 2.0, 3.5],
                "CONT_FEATURE_2": [4.0, 5.2, 6.3],
            }
        )
        categorical_var_spec = {"CAT_FEATURE_1": "reset_index", "CAT_FEATURE_2": "reset_index"}
        continuous_var_spec = {"CONT_FEATURE_1": "log-epsilon", "CONT_FEATURE_2": "minmax"}
        likelihood_spec = {"CONT_FEATURE_1": {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "conditions": ["CONT_FEATURE_2"]}},
                           "CONT_FEATURE_2": {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "conditions": ["CAT_FEATURE_1"]},
                           "CAT_FEATURE_1": {"distribution": "categorical", "parameters": {}, "conditions": []},
        outcome_name = "CAT_FEATURE_2"
        conditional_features = prepare_conditional_features(
            data, categorical_var_spec, continuous_var_spec, likelihood_spec, outcome_name
        )

    :param data: the dataframe of the dataset
    :param categorical_var_spec: transformation of the categorical variables
    :param continuous_var_spec: transformation of the continuous variables
    :param conditions: the conditional dependencies of each variable
    :param likelihood_spec: likelihood function of each variable and its parameters
    :param outcome_name: the name of the outcome variable, which will be ignored in the upstream model
    :returns: A dictionary of the processed dataset, including the following keys:

        - "transformation_params": transformation parameters for each variable
        - "continuous_keys": list of continuous variable names
        - "categorical_keys": list of categorical variable names
        - "conditional_spec": conditional specification for each variable
        - "data_dict": the processed dataset
        - "topo_ordering": topological ordering of the variables
    """

    data_dict = produce_dictionary_for_inference(
        data,
        continuous_features=continuous_var_spec,
        categorical_features=categorical_var_spec,
    )

    # add individual feature dimension
    for setting in data_dict:
        for key in data_dict[setting]:
            if isinstance(data_dict[setting][key], torch.Tensor):
                data_dict[setting][key] = data_dict[setting][key].unsqueeze(-1)

    continuous_keys = [
        f"{key}_{SUFFIX_MAPPING[continuous_var_spec[key]]}"
        for key in continuous_var_spec
        if key != outcome_name
    ]

    categorical_keys = [
        f"{key}_{SUFFIX_MAPPING[categorical_var_spec[key]]}"
        for key in categorical_var_spec
        if key != outcome_name
    ]

    nd_keys = [
        key
        for key in likelihood_spec
        if likelihood_spec[key]["distribution"] in CONT_DISTS_ND
    ]

    # default to same no of dimensions for each categorical variable
    # embedding_dim_list = [embedding_dim] * len(num_embeddings_list)

    def split_categorical_continuous(variables: set):
        # process a list of variable names into a list of categorical variables and a list of continuous variables
        # the number of categories for each categorical variable will be collected in GenerativeConditionalModel
        categorical = []
        continuous = []

        for variable in variables:
            if variable in categorical_var_spec:
                cond_key = (
                    f"{variable}_{SUFFIX_MAPPING[categorical_var_spec[variable]]}"
                )
                categorical.append(cond_key)
            elif variable in continuous_var_spec:
                continuous.append(
                    f"{variable}_{SUFFIX_MAPPING[continuous_var_spec[variable]]}"
                )
            elif variable in nd_keys:
                continuous.append(variable)
            else:
                raise ValueError(f"Unknown variable {variable}")
        return (
            categorical,
            continuous,
        )

    # split out and remove the conditions from the likelihood specification
    conditions = {}
    for key in likelihood_spec.keys():
        conditions[key] = set(likelihood_spec[key]["conditions"])
        likelihood_spec[key].pop("conditions")

    conditional_spec = {}
    conditions.pop(outcome_name)
    for key in conditions:
        cat, con = split_categorical_continuous(conditions[key])
        if likelihood_spec[key]["distribution"] in CAT_DISTS_1D:
            obs_key = f"{key}_{SUFFIX_MAPPING[categorical_var_spec[key]]}"
        elif likelihood_spec[key]["distribution"] in CONT_DISTS_1D:
            obs_key = f"{key}_{SUFFIX_MAPPING[continuous_var_spec[key]]}"
        elif likelihood_spec[key]["distribution"] in CONT_DISTS_ND:
            obs_key = key
        conditional_spec[obs_key] = {
            "categorical_cond": cat,
            "continuous_cond": con,
            "likelihood_dist": likelihood_spec[key]["distribution"],
            "parameters": likelihood_spec[key]["parameters"],
        }
        if likelihood_spec[key]["distribution"] in ["categorical"]:
            conditional_spec[obs_key]["parameters"]["obs_categories"] = len(
                data_dict["categorical"][obs_key].unique()
            )
        if likelihood_spec[key]["distribution"] in CONT_DISTS_ND:
            feature_list = likelihood_spec[key]["feature_list"]
            transformed_feature_list = [
                f"{key}_{SUFFIX_MAPPING[continuous_var_spec[key]]}"
                for key in feature_list
            ]
            conditional_spec[obs_key]["feature_list"] = transformed_feature_list
            datas = [data_dict["continuous"][key] for key in transformed_feature_list]
            data_dict["continuous"][key] = torch.concat(datas, dim=1)

    # Check the topological ordering and define the topological structure
    topo_ordering = graphlib.TopologicalSorter(conditions).static_order()
    topo_ordering = [
        f"{key}_{SUFFIX_MAPPING[continuous_var_spec[key]]}"
        if key in continuous_var_spec
        else f"{key}_{SUFFIX_MAPPING[categorical_var_spec[key]]}"
        if key in categorical_var_spec
        else key
        for key in topo_ordering
    ]

    return {
        "transformation_params": data_dict["transformation_params"],
        "continuous_keys": continuous_keys,
        "categorical_keys": categorical_keys,
        "conditional_spec": conditional_spec,
        "data_dict": data_dict,
        "topo_ordering": topo_ordering,
    }


def get_variable_from_structured_data(
    structured_data: dict,
    variable_name: str,
    aliases: dict | None = None,
    canonical_form=False,
):
    """
    Get a variable from a structured data dictionary that may contain
    multiple sub-dictionaries (e.g., "categorical", "continuous", etc.).

    :param structured_data: A dictionary of dictionaries.
    :param variable_name: The key to look for.
    :param aliases: Optional dict mapping variable names to alternative names (or lists of names).
    :return: The value associated with `variable_name` or one of its aliases.
    :raises ValueError: If `variable_name` (or its aliases) is not found.
    """
    # Build list of names to check: the variable itself + its aliases
    names_to_check = [variable_name]
    if aliases and variable_name in aliases:
        if isinstance(aliases[variable_name], list | tuple):
            names_to_check.extend(aliases[variable_name])
        else:
            names_to_check.append(aliases[variable_name])

    if canonical_form:
        subdicts_to_search = {
            k: structured_data[k]
            for k in ["continuous", "categorical"]
            if k in structured_data
        }
    else:
        subdicts_to_search = structured_data

    # Search all sub-dicts
    for subdict in subdicts_to_search.values():
        for name in names_to_check:
            if name in subdict:
                return subdict[name]

    raise ValueError(
        f"Variable {variable_name} not found in structured data (checked aliases: {names_to_check})."
    )
