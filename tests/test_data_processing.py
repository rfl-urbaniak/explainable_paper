import numpy as np
import pandas as pd
import pytest
import torch

from pci.tools.data_processing import (
    CategoricalParams,
    IdentityParams,
    LogEpsilonParams,
    MinMaxParams,
    StandardizeParams,
    descale_tensor,
    destandardize_tensor,
    forward_transform,
    inverse_transform_log_epsilon,
    invert_all_features,
    min_max_scale_tensor,
    prepare_conditional_features,
    prepare_features,
    produce_dictionary_for_inference,
    reset_categorical_index,
    standardize_tensor,
    strip_suffix,
    transform_log_epsilon,
    unmap_categorical_index,
)


def test_unmap_categorical_index():
    """Test basic unmapping of categorical indices."""
    tensor = torch.tensor([0, 1, 0, 2])
    mapping = {100: 0, 200: 1, 300: 2}

    result = unmap_categorical_index(tensor, mapping)

    assert torch.equal(result, torch.tensor([100, 200, 100, 300]))


def test_unmap_categorical_index_with_other():
    """Test unmapping when __OTHER__ is in the mapping."""
    tensor = torch.tensor([0, 1, 2, 3])
    mapping = {100: 0, 200: 1, 300: 2, "__OTHER__": 3}

    result = unmap_categorical_index(tensor, mapping)

    # __OTHER__ should map to max_numeric_key + 1 = 300 + 1 = 301
    assert torch.equal(result, torch.tensor([100, 200, 300, 301]))


def test_unmap_categorical_index_preserves_device():
    """Test that unmapping preserves tensor device."""
    tensor = torch.tensor([0, 1, 0])
    mapping = {100: 0, 200: 1}

    result = unmap_categorical_index(tensor, mapping)

    assert result.device == tensor.device


def test_invert_all_features_standardize():
    """Test inverting standardized continuous features."""
    feature_dict = {"height": torch.tensor([[0.0], [1.0], [-1.0]])}
    transformation_params = {
        "height_transformed": {
            "mean": torch.tensor(170.0),
            "std": torch.tensor(10.0),
        }
    }
    continuous_transformations = {"height": "standardize"}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    expected = torch.tensor([[170.0], [180.0], [160.0]])
    assert torch.allclose(result["height"], expected, atol=1e-5)


def test_invert_all_features_minmax():
    """Test inverting minmax scaled continuous features."""
    feature_dict = {"weight": torch.tensor([[0.0], [0.5], [1.0]])}
    transformation_params = {
        "weight_transformed": {
            "min_val": torch.tensor(50.0),
            "range_val": torch.tensor(50.0),
            "max_val": torch.tensor(100.0),
        }
    }
    continuous_transformations = {"weight": "minmax"}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    expected = torch.tensor([[50.0], [75.0], [100.0]])
    assert torch.allclose(result["weight"], expected, atol=1e-5)


def test_invert_all_features_log_epsilon():
    """Test inverting log-epsilon transformed continuous features."""
    feature_dict = {"price": torch.tensor([[0.0], [1.0], [-1.0]])}
    transformation_params = {
        "price_transformed": {
            "mean": torch.tensor(2.0),
            "std": torch.tensor(1.0),
            "epsilon": 1e-6,
        }
    }
    continuous_transformations = {"price": "log-epsilon"}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    # Values should be positive after inverse transform
    assert torch.all(result["price"] > 0)


def test_invert_all_features_categorical():
    """Test inverting categorical features."""
    feature_dict = {"city_id": torch.tensor([[0], [1], [2]])}
    transformation_params = {
        "city_id_transformed": {"mapping": {100: 0, 200: 1, 300: 2}, "n_cat": 3}
    }
    continuous_transformations = {}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    expected = torch.tensor([[100], [200], [300]])
    assert torch.equal(result["city_id"], expected)


def test_invert_all_features_categorical_with_other():
    """Test inverting categorical features with __OTHER__ values."""
    feature_dict = {"city_id": torch.tensor([[0], [1], [3]])}
    transformation_params = {
        "city_id_transformed": {
            "mapping": {100: 0, 200: 1, 300: 2, "__OTHER__": 3},
            "n_cat": 4,
        }
    }
    continuous_transformations = {}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    # __OTHER__ should map to max_numeric_key + 1 = 301
    expected = torch.tensor([[100], [200], [301]])
    assert torch.equal(result["city_id"], expected)


def test_invert_all_features_mixed():
    """Test inverting mixed continuous and categorical features."""
    feature_dict = {
        "height": torch.tensor([[0.0], [1.0]]),
        "city_id": torch.tensor([[0], [1]]),
    }
    transformation_params = {
        "height_transformed": {"mean": torch.tensor(170.0), "std": torch.tensor(10.0)},
        "city_id_transformed": {"mapping": {100: 0, 200: 1}, "n_cat": 2},
    }
    continuous_transformations = {"height": "standardize"}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    assert torch.allclose(result["height"], torch.tensor([[170.0], [180.0]]), atol=1e-5)
    assert torch.equal(result["city_id"], torch.tensor([[100], [200]]))


def test_invert_all_features_with_clipping():
    """Test inverting features with clipping applied."""
    feature_dict = {"height": torch.tensor([[0.0], [5.0], [-5.0]])}
    transformation_params = {
        "height_transformed": {"mean": torch.tensor(170.0), "std": torch.tensor(10.0)}
    }
    continuous_transformations = {"height": "standardize"}
    clip_raw_data_dict = {"height": (160.0, 180.0)}

    result = invert_all_features(
        feature_dict,
        transformation_params,
        continuous_transformations,
        clip_raw_data_dict=clip_raw_data_dict,
    )

    # Values outside [160, 180] should be clipped
    expected = torch.tensor([[170.0], [180.0], [160.0]])
    assert torch.allclose(result["height"], expected, atol=1e-5)


def test_invert_all_features_with_clipping_tensor_bounds():
    """Test inverting features with tensor-based clipping bounds."""
    feature_dict = {"height": torch.tensor([[0.0], [5.0], [-5.0]])}
    transformation_params = {
        "height_transformed": {"mean": torch.tensor(170.0), "std": torch.tensor(10.0)}
    }
    continuous_transformations = {"height": "standardize"}
    clip_raw_data_dict = {"height": (torch.tensor(160.0), torch.tensor(180.0))}

    result = invert_all_features(
        feature_dict,
        transformation_params,
        continuous_transformations,
        clip_raw_data_dict=clip_raw_data_dict,
    )

    expected = torch.tensor([[170.0], [180.0], [160.0]])
    assert torch.allclose(result["height"], expected, atol=1e-5)


def test_invert_all_features_clipping_only_continuous():
    """Test that clipping is only applied to continuous features."""
    feature_dict = {
        "height": torch.tensor([[5.0]]),
        "city_id": torch.tensor([[0]]),
    }
    transformation_params = {
        "height_transformed": {"mean": torch.tensor(170.0), "std": torch.tensor(10.0)},
        "city_id_transformed": {"mapping": {100: 0}, "n_cat": 1},
    }
    continuous_transformations = {"height": "standardize"}
    clip_raw_data_dict = {"height": (160.0, 180.0), "city_id": (50, 150)}

    result = invert_all_features(
        feature_dict,
        transformation_params,
        continuous_transformations,
        clip_raw_data_dict=clip_raw_data_dict,
    )

    # Height should be clipped
    assert torch.allclose(result["height"], torch.tensor([[180.0]]), atol=1e-5)
    # city_id should NOT be clipped (remains 100, not clipped to [50, 150])
    assert torch.equal(result["city_id"], torch.tensor([[100]]))


def test_invert_all_features_unknown_transformation():
    """Test that unknown transformation types raise an error."""
    feature_dict = {"feature": torch.tensor([[0.0]])}
    transformation_params = {"feature_transformed": {}}
    continuous_transformations = {"feature": "unknown_method"}

    with pytest.raises(ValueError, match="Unknown transform type 'unknown_method'"):
        invert_all_features(
            feature_dict, transformation_params, continuous_transformations
        )


def test_invert_all_features_empty():
    """Test inverting an empty feature dictionary."""
    feature_dict = {}
    transformation_params = {}
    continuous_transformations = {}

    result = invert_all_features(
        feature_dict, transformation_params, continuous_transformations
    )

    assert result == {}


def test_invert_all_features_params_object_vs_dict():
    """Test that transformation params work as both objects and dicts."""
    feature_dict = {"city_id": torch.tensor([[0], [1]])}

    # Test with dict
    transformation_params_dict = {
        "city_id_transformed": {"mapping": {100: 0, 200: 1}, "n_cat": 2}
    }
    continuous_transformations = {}

    result_dict = invert_all_features(
        feature_dict, transformation_params_dict, continuous_transformations
    )

    # Test with object (mock object with mapping attribute)
    class ParamsObject:
        def __init__(self, mapping):
            self.mapping = mapping

    transformation_params_obj = {
        "city_id_transformed": ParamsObject(mapping={100: 0, 200: 1})
    }

    result_obj = invert_all_features(
        feature_dict, transformation_params_obj, continuous_transformations
    )

    assert torch.equal(result_dict["city_id"], result_obj["city_id"])


def test_strip_suffix():
    assert strip_suffix("height_log_epsilon") == "height"
    assert strip_suffix("height_minmax") == "height"
    assert strip_suffix("height_std") == "height"
    assert strip_suffix("height_reind") == "height"


def test_standardize_and_destandardize_tensor():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    standardized_tensor, params = standardize_tensor(tensor)

    mean = standardized_tensor.mean(dim=0)
    std = standardized_tensor.std(dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-6)

    recovered_tensor = destandardize_tensor(standardized_tensor, params)
    assert torch.allclose(recovered_tensor, tensor, atol=1e-6)


def test_standardize_and_destandardize_tensor_with_correct_params():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = torch.tensor([3.0, 4.0])
    std = torch.tensor([2.0, 2.0])
    standardized_tensor, params = standardize_tensor(tensor, mean=mean, std=std)
    assert torch.allclose(
        standardized_tensor.mean(dim=0), torch.zeros_like(mean), atol=1e-6
    )
    assert torch.allclose(
        standardized_tensor.std(dim=0), torch.ones_like(std), atol=1e-6
    )

    recovered_tensor = destandardize_tensor(standardized_tensor, params)
    assert torch.allclose(recovered_tensor, tensor, atol=1e-6)

    tranformed_no_params, _ = standardize_tensor(tensor)
    assert torch.allclose(tranformed_no_params, standardized_tensor, atol=1e-6)


def test_standardize_and_destandardize_tensor_with_incorrect_params():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = torch.tensor([5.0, 5.0])
    std = torch.tensor([5.0, 5.0])
    standardized_tensor, params = standardize_tensor(tensor, mean=mean, std=std)
    no_params_tensor, _ = standardize_tensor(tensor)
    assert torch.logical_not(
        torch.isclose(standardized_tensor, no_params_tensor, atol=1e-6)
    ).any()


def test_min_max_scale_and_descale_tensor():
    tensor = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    scaled_tensor, params = min_max_scale_tensor(tensor)

    assert torch.all(scaled_tensor >= 0.0)
    assert torch.all(scaled_tensor <= 1.0)

    recovered_tensor = descale_tensor(scaled_tensor, params)

    assert torch.allclose(recovered_tensor, tensor, atol=1e-6)


def test_min_max_scale_and_descale_tensor_with_correct_params():
    tensor = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    min_val = torch.tensor([1.0, 4.0])
    max_val = torch.tensor([3.0, 6.0])
    scaled_tensor, params = min_max_scale_tensor(
        tensor, min_val=min_val, max_val=max_val
    )
    assert torch.allclose(
        scaled_tensor.min(dim=0)[0], torch.zeros_like(min_val), atol=1e-6
    )
    assert torch.allclose(
        scaled_tensor.max(dim=0)[0], torch.ones_like(max_val), atol=1e-6
    )

    recovered_tensor = descale_tensor(scaled_tensor, params)
    assert torch.allclose(recovered_tensor, tensor, atol=1e-6)

    tranformed_no_params, _ = min_max_scale_tensor(tensor)
    assert torch.allclose(tranformed_no_params, scaled_tensor, atol=1e-6)


def test_min_max_scale_and_descale_tensor_with_incorrect_params():
    tensor = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    min_val = torch.tensor([0.0, 5.0])
    max_val = torch.tensor([1.0, 5.0])
    scaled_tensor, params = min_max_scale_tensor(
        tensor, min_val=min_val, max_val=max_val
    )
    no_params_tensor, _ = min_max_scale_tensor(tensor)
    assert torch.logical_not(
        torch.isclose(scaled_tensor, no_params_tensor, atol=1e-6)
    ).any()


def test_transform_and_inverse_log_epsilon():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    transformed_tensor, params = transform_log_epsilon(tensor)

    mean = transformed_tensor.mean(dim=0)
    std = transformed_tensor.std(dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-6)

    recovered_tensor = inverse_transform_log_epsilon(transformed_tensor, params)

    assert torch.allclose(recovered_tensor, tensor, atol=1e-5)


def test_transform_and_inverse_log_epsilon_with_correct_params():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = torch.tensor([0.90268391, 1.29040062])
    std = torch.tensor([0.82241291, 0.55554819])
    transformed_tensor, params = transform_log_epsilon(tensor, mean=mean, std=std)
    assert torch.allclose(
        transformed_tensor.mean(dim=0), torch.zeros_like(mean), atol=1e-6
    ), f"Mean: {transformed_tensor.mean(dim=0)}"
    assert torch.allclose(
        transformed_tensor.std(dim=0), torch.ones_like(std), atol=1e-6
    ), f"Std: {transformed_tensor.std(dim=0)}"

    recovered_tensor = inverse_transform_log_epsilon(transformed_tensor, params)
    assert torch.allclose(recovered_tensor, tensor, atol=1e-5)


def test_transform_and_inverse_log_epsilon_with_incorrect_params():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = torch.tensor([0.1, 1.1])
    std = torch.tensor([0.1, 0.1])
    transformed_tensor, params = transform_log_epsilon(tensor, mean=mean, std=std)
    no_params_tensor, _ = transform_log_epsilon(tensor)
    assert torch.logical_not(
        torch.isclose(transformed_tensor, no_params_tensor, atol=1e-6)
    ).any()


def test_reset_categorical_index():
    tensor = torch.tensor([10, 30, 10, 20, 30, 20, 10])

    reindexed_tensor, mapping = reset_categorical_index(tensor)

    assert torch.equal(
        reindexed_tensor.unique(sorted=True), torch.arange(len(mapping) - 1)
    )

    inverse_mapping = {v: k for k, v in mapping.items()}
    recovered_tensor = torch.tensor(
        [inverse_mapping[int(val.item())] for val in reindexed_tensor]
    )
    assert torch.equal(recovered_tensor, tensor)


def test_reset_categorical_index_with_correct_params():
    tensor = torch.tensor([10, 30, 10, 20, 30, 20, 10])
    mapping = {10: 0, 20: 1, 30: 2}
    reindexed_tensor, mapping = reset_categorical_index(tensor, mapping=mapping)
    assert torch.equal(reindexed_tensor.unique(sorted=True), torch.arange(len(mapping)))
    assert torch.equal(reindexed_tensor, torch.tensor([0, 2, 0, 1, 2, 1, 0]))

    tranformed_no_params, _ = reset_categorical_index(tensor)
    assert torch.equal(tranformed_no_params, reindexed_tensor)


def test_reset_categorical_index_with_incorrect_params():
    tensor = torch.tensor([10, 30, 10, 20, 30, 20, 10])
    mapping = {10: 0, 20: 1, 30: 3}
    reindexed_tensor, mapping = reset_categorical_index(tensor, mapping=mapping)
    assert torch.equal(reindexed_tensor, torch.tensor([0, 3, 0, 1, 3, 1, 0]))

    no_params_tensor, _ = reset_categorical_index(tensor)
    assert not torch.equal(no_params_tensor, reindexed_tensor)


def test_reset_categorical_index_old_mapping_no_other():
    tensor = torch.tensor([10, 20, 10])
    mapping = {10: 0, 20: 1}  # old-style mapping, no __OTHER__

    reindexed, out_mapping = reset_categorical_index(
        tensor,
        mapping=mapping,
    )

    # Values map correctly
    assert torch.equal(reindexed, torch.tensor([0, 1, 0]))

    # Mapping is unchanged
    assert out_mapping is mapping
    assert "__OTHER__" not in out_mapping


def test_reset_categorical_index_creates_contiguous_mapping_with_other():
    x = torch.tensor([10, 20, 10, 30])

    y, mapping = reset_categorical_index(x)

    assert torch.equal(y, torch.tensor([0, 1, 0, 2]))
    assert mapping == {
        10: 0,
        20: 1,
        30: 2,
        "__OTHER__": 3,
    }


def test_reset_categorical_index_collapses_other_values():
    x = torch.tensor([1, 2, 3, 99, 100, 2, 100])

    y, mapping = reset_categorical_index(
        x,
        other_values=[99, 100],
    )

    assert torch.equal(y, torch.tensor([0, 1, 2, 3, 3, 1, 3]))
    assert mapping == {
        1: 0,
        2: 1,
        3: 2,
        "__OTHER__": 3,
    }


def test_reset_categorical_index_maps_unseen_to_other():
    x_train = torch.tensor([1, 2, 3])
    _, mapping = reset_categorical_index(x_train)

    x_test = torch.tensor([1, 4])

    y_test, _ = reset_categorical_index(
        x_test,
        mapping=mapping,
    )

    assert mapping["__OTHER__"] == len(x_train)
    assert torch.equal(y_test, torch.tensor([0, mapping["__OTHER__"]]))


def test_reset_categorical_index_preserves_known_values():
    x_train = torch.tensor([5, 6])
    _, mapping = reset_categorical_index(x_train)

    x_test = torch.tensor([6, 5, 6])

    y_test, _ = reset_categorical_index(
        x_test,
        mapping=mapping,
    )

    assert torch.equal(y_test, torch.tensor([1, 0, 1]))


def test_reset_categorical_index_preserves_device():
    x = torch.tensor([1, 2, 3])

    y, _ = reset_categorical_index(x)

    assert y.device == x.device


def test_produce_dictionary_for_inference():
    # Create a dummy DataFrame
    df = pd.DataFrame(
        {
            "height": [1.60, 1.70, 1.65, 1.80],
            "weight": [60, 70, 65, 80],
            "city_id": [100, 200, 100, 300],
        }
    )

    continuous_features = {
        "height": "standardize",
        "weight": "log-epsilon",
    }

    categorical_features = {"city_id": "reset_index"}

    data_dict = produce_dictionary_for_inference(
        df, continuous_features, categorical_features
    )

    assert "continuous" in data_dict
    assert "categorical" in data_dict
    assert "transformation_params" in data_dict

    assert "height_std" in data_dict["continuous"]
    assert "weight_log_epsilon" in data_dict["continuous"]
    assert "city_id_reind" in data_dict["categorical"]

    for key in ["height_std", "weight_log_epsilon"]:
        assert (
            data_dict["continuous"][key].shape
            == torch.tensor(df[key.split("_")[0]]).shape
        )

    assert (
        data_dict["categorical"]["city_id_reind"].shape
        == torch.tensor(df["city_id"]).shape
    )

    for key in ["height_std", "weight_log_epsilon", "city_id_reind"]:
        assert key in data_dict["transformation_params"]
        assert isinstance(data_dict["transformation_params"][key], dict)

    continuous_features = {
        "height": "standardize",
        "weight": "unknown_method",  # This should raise an error
    }

    with pytest.raises(
        ValueError,
        match="Unknown transformation method 'unknown_method' for column 'weight'",
    ):
        produce_dictionary_for_inference(df, continuous_features, categorical_features)


def test_produce_dictionary_for_inference_with_correct_params():
    # Create a dummy DataFrame
    df = pd.DataFrame(
        {
            "height": [1.60, 1.70, 1.65, 1.80],
            "weight": [60, 70, 65, 80],
            "city_id": [100, 200, 100, 300],
        }
    )

    continuous_features = {
        "height": "standardize",
        "weight": "log-epsilon",
    }

    categorical_features = {"city_id": "reset_index"}

    transformation_params = {
        "height_std": {"mean": torch.tensor(1.6875), "std": torch.tensor(0.08539123)},
        "weight_log_epsilon": {
            "mean": torch.tensor(4.2248134613),
            "std": torch.tensor(0.1222589016),
            "epsilon": 1e-6,
        },
        "city_id_reind": {"mapping": {100: 0, 200: 1, 300: 2}, "n_cat": 3},
    }

    data_dict = produce_dictionary_for_inference(
        df,
        continuous_features,
        categorical_features,
        transformation_params=transformation_params,
    )

    assert "continuous" in data_dict
    assert "categorical" in data_dict
    assert "transformation_params" in data_dict

    assert "height_std" in data_dict["continuous"]
    assert "weight_log_epsilon" in data_dict["continuous"]
    assert "city_id_reind" in data_dict["categorical"]

    data_dict_no_params = produce_dictionary_for_inference(
        df, continuous_features, categorical_features
    )

    assert data_dict.keys() == data_dict_no_params.keys()

    for key in data_dict["transformation_params"].keys():
        for param_key in data_dict["transformation_params"][key].keys():
            if isinstance(
                data_dict["transformation_params"][key][param_key], torch.Tensor
            ):
                assert torch.allclose(
                    data_dict["transformation_params"][key][param_key],
                    data_dict_no_params["transformation_params"][key][param_key],
                )
            else:
                left = data_dict["transformation_params"][key][param_key]

                if isinstance(left, dict):
                    left["__OTHER__"] = 3

                if param_key == "n_cat":
                    left = left + 1

                assert (
                    left == data_dict_no_params["transformation_params"][key][param_key]
                )

    for key in data_dict["continuous"].keys():
        assert torch.allclose(
            data_dict["continuous"][key], data_dict_no_params["continuous"][key]
        )

    for key in data_dict["categorical"].keys():
        assert torch.allclose(
            data_dict["categorical"][key], data_dict_no_params["categorical"][key]
        )


def test_prepare_features():
    data = pd.DataFrame(
        {
            "height": [1.60, 1.70, 1.65, 1.80],
            "weight": [60, 70, 65, 80],
            "city_id": [100, 200, 100, 300],
        }
    )
    categorical_var_spec = {"city_id": "reset_index"}
    continuous_var_spec = {"height": "standardize", "weight": "log-epsilon"}
    outcome_name = "weight"
    results = prepare_features(
        data, categorical_var_spec, continuous_var_spec, outcome_name
    )

    assert set(results.keys()) == {
        "x_categorical",
        "x_continuous",
        "x",
        "y",
        "num_embeddings_list",
        "transformation_params",
        "continuous_keys",
        "categorical_keys",
        "data_dict",
    }

    precomputed_params = {
        "height_std": {"mean": torch.tensor(1.6875), "std": torch.tensor(0.08539123)},
        "weight_log_epsilon": {
            "mean": torch.tensor(4.2248134613),
            "std": torch.tensor(0.1222589016),
            "epsilon": 1e-6,
        },
        "city_id_reind": {
            "mapping": {100: 0, 200: 1, 300: 2, "__OTHER__": 3},
            "n_cat": 4,
        },
    }
    assert results["x_categorical"].shape == (4, 1)
    assert results["x_continuous"].shape == (4, 1)
    assert results["x"].shape == (4, 2)
    assert results["y"].shape == (4,)
    assert results["num_embeddings_list"] == [3]
    assert results["continuous_keys"] == ["height_std"]
    assert results["categorical_keys"] == ["city_id_reind"]

    for key in results["transformation_params"].keys():
        for param_key in results["transformation_params"][key].keys():
            if isinstance(
                results["transformation_params"][key][param_key], torch.Tensor
            ):
                assert torch.allclose(
                    results["transformation_params"][key][param_key],
                    precomputed_params[key][param_key],
                )
            else:
                assert (
                    results["transformation_params"][key][param_key]
                    == precomputed_params[key][param_key]
                )


def test_prepare_conditional_features():
    df = pd.DataFrame(
        {
            "height": [1.60, 1.70, 1.65, 1.80],
            "weight": [60, 70, 65, 80],
            "city_id": [100, 200, 100, 300],
        }
    )

    categorical_var_spec = {"city_id": "reset_index"}
    continuous_var_spec = {"height": "standardize", "weight": "log-epsilon"}
    likelihood_spec = {
        "height": {"distribution": "normal", "parameters": {}, "conditions": []},
        "weight": {
            "distribution": "mixnormal",
            "parameters": {"n_components": 2},
            "conditions": [],
        },
        "city_id": {
            "distribution": "categorical",
            "parameters": {"obs_categories": 3},
            "conditions": ["height"],
        },
    }
    outcome_name = "weight"
    conditional_features = prepare_conditional_features(
        df,
        categorical_var_spec,
        continuous_var_spec,
        likelihood_spec,
        outcome_name,
    )

    assert set(conditional_features.keys()) == {
        "transformation_params",
        "continuous_keys",
        "categorical_keys",
        "conditional_spec",
        "data_dict",
        "topo_ordering",
    }

    assert set(conditional_features["data_dict"]["continuous"].keys()) == {
        "height",
        "weight",
        "height_std",
        "weight_log_epsilon",
    }
    assert set(conditional_features["data_dict"]["categorical"].keys()) == {
        "city_id",
        "city_id_reind",
    }

    # We now have a feature dimension after data processing
    assert conditional_features["data_dict"]["continuous"]["height"].shape == (4, 1)
    assert conditional_features["data_dict"]["continuous"]["weight"].shape == (4, 1)
    assert conditional_features["data_dict"]["continuous"]["height_std"].shape == (4, 1)
    assert conditional_features["data_dict"]["continuous"][
        "weight_log_epsilon"
    ].shape == (4, 1)
    assert conditional_features["data_dict"]["categorical"]["city_id"].shape == (4, 1)
    assert conditional_features["data_dict"]["categorical"]["city_id_reind"].shape == (
        4,
        1,
    )

    precomputed_params = {
        "height_std": {"mean": torch.tensor(1.6875), "std": torch.tensor(0.08539123)},
        "weight_log_epsilon": {
            "mean": torch.tensor(4.2248134613),
            "std": torch.tensor(0.1222589016),
            "epsilon": 1e-6,
        },
        "city_id_reind": {
            "mapping": {100: 0, 200: 1, 300: 2, "__OTHER__": 3},
            "n_cat": 4,
        },
    }

    for key in conditional_features["transformation_params"].keys():
        for param_key in conditional_features["transformation_params"][key].keys():
            if isinstance(
                conditional_features["transformation_params"][key][param_key],
                torch.Tensor,
            ):
                assert torch.allclose(
                    conditional_features["transformation_params"][key][param_key],
                    precomputed_params[key][param_key],
                )
            else:
                assert (
                    conditional_features["transformation_params"][key][param_key]
                    == precomputed_params[key][param_key]
                )

    assert conditional_features["continuous_keys"] == ["height_std"]
    assert conditional_features["categorical_keys"] == ["city_id_reind"]
    assert conditional_features["conditional_spec"]["city_id_reind"]
    assert conditional_features["topo_ordering"] == ["height_std", "city_id_reind"]
    assert (
        conditional_features["conditional_spec"]["city_id_reind"]["categorical_cond"]
        == []
    )
    assert conditional_features["conditional_spec"]["city_id_reind"][
        "continuous_cond"
    ] == ["height_std"]
    assert (
        conditional_features["conditional_spec"]["city_id_reind"]["likelihood_dist"]
        == "categorical"
    )
    assert (
        conditional_features["conditional_spec"]["height_std"]["categorical_cond"] == []
    )
    assert (
        conditional_features["conditional_spec"]["height_std"]["continuous_cond"] == []
    )
    assert (
        conditional_features["conditional_spec"]["height_std"]["likelihood_dist"]
        == "normal"
    )


def test_prepare_conditional_features_nd():
    df = pd.DataFrame(
        {
            "height1": [1.60, 1.70, 1.65, 1.80],
            "height2": [1.65, 1.75, 1.70, 1.85],
            "weight": [60, 70, 65, 80],
            "city_id": [100, 200, 100, 300],
            "color1": [0.1, 0.2, 0.3, 0.4],
            "color2": [0.9, 0.8, 0.7, 0.6],
        }
    )
    categorical_var_spec = {"city_id": "reset_index"}
    continuous_var_spec = {
        "height1": "standardize",
        "height2": "standardize",
        "weight": "log-epsilon",
        "color1": "minmax",
        "color2": "minmax",
    }
    likelihood_spec = {
        "height": {
            "distribution": "multivariate_normal",
            "parameters": {"size": 2},
            "feature_list": ["height1", "height2"],
            "conditions": [],
        },
        "weight": {
            "distribution": "mixnormal",
            "parameters": {"n_components": 2},
            "conditions": [],
        },
        "color": {
            "distribution": "multivariate_normal",
            "parameters": {"size": 2},
            "feature_list": ["color1", "color2"],
            "conditions": ["city_id", "height"],
        },
        "city_id": {
            "distribution": "categorical",
            "parameters": {"obs_categories": 3},
            "conditions": ["height"],
        },
    }
    outcome_name = "weight"
    conditional_features = prepare_conditional_features(
        df,
        categorical_var_spec,
        continuous_var_spec,
        likelihood_spec,
        outcome_name,
    )

    assert set(conditional_features.keys()) == {
        "transformation_params",
        "continuous_keys",
        "categorical_keys",
        "conditional_spec",
        "data_dict",
        "topo_ordering",
    }

    assert set(conditional_features["data_dict"]["continuous"].keys()) == {
        "height1",
        "height2",
        "weight",
        "color1",
        "color2",
        "height1_std",
        "height2_std",
        "weight_log_epsilon",
        "color1_minmax",
        "color2_minmax",
        "height",
        "color",
    }
    assert set(conditional_features["data_dict"]["categorical"].keys()) == {
        "city_id",
        "city_id_reind",
    }

    assert conditional_features["data_dict"]["continuous"]["height"].shape == (4, 2)
    assert conditional_features["data_dict"]["continuous"]["height1_std"].shape == (
        4,
        1,
    )
    assert conditional_features["data_dict"]["continuous"]["height2_std"].shape == (
        4,
        1,
    )
    assert torch.all(
        conditional_features["data_dict"]["continuous"]["height"][..., 0]
        == conditional_features["data_dict"]["continuous"]["height1_std"].squeeze(-1),
    )
    assert torch.all(
        conditional_features["data_dict"]["continuous"]["height"][..., 1]
        == conditional_features["data_dict"]["continuous"]["height2_std"].squeeze(-1),
    )

    assert conditional_features["data_dict"]["continuous"]["color"].shape == (4, 2)
    assert conditional_features["data_dict"]["continuous"]["color1_minmax"].shape == (
        4,
        1,
    )
    assert conditional_features["data_dict"]["continuous"]["color2_minmax"].shape == (
        4,
        1,
    )
    assert torch.all(
        conditional_features["data_dict"]["continuous"]["color"][..., 0]
        == conditional_features["data_dict"]["continuous"]["color1_minmax"].squeeze(-1),
    )
    assert torch.all(
        conditional_features["data_dict"]["continuous"]["color"][..., 1]
        == conditional_features["data_dict"]["continuous"]["color2_minmax"].squeeze(-1),
    )

    precomputed_params = {
        "height1_std": {"mean": torch.tensor(1.6875), "std": torch.tensor(0.08539123)},
        "height2_std": {"mean": torch.tensor(1.7375), "std": torch.tensor(0.08539123)},
        "weight_log_epsilon": {
            "mean": torch.tensor(4.2248134613),
            "std": torch.tensor(0.1222589016),
            "epsilon": 1e-6,
        },
        "color1_minmax": {
            "min_val": torch.tensor(0.1),
            "range_val": torch.tensor(0.3),
            "max_val": torch.tensor(0.4),
        },
        "color2_minmax": {
            "min_val": torch.tensor(0.6),
            "range_val": torch.tensor(0.3),
            "max_val": torch.tensor(0.9),
        },
        "city_id_reind": {
            "mapping": {100: 0, 200: 1, 300: 2, "__OTHER__": 3},
            "n_cat": 4,
        },
    }

    for key in conditional_features["transformation_params"].keys():
        for param_key in conditional_features["transformation_params"][key].keys():
            if isinstance(
                conditional_features["transformation_params"][key][param_key],
                torch.Tensor,
            ):
                assert torch.allclose(
                    conditional_features["transformation_params"][key][param_key],
                    precomputed_params[key][param_key],
                )
            else:
                assert (
                    conditional_features["transformation_params"][key][param_key]
                    == precomputed_params[key][param_key]
                )

    assert set(conditional_features["continuous_keys"]) == {
        "height1_std",
        "height2_std",
        "color1_minmax",
        "color2_minmax",
    }

    assert set(conditional_features["categorical_keys"]) == {
        "city_id_reind",
    }
    assert conditional_features["topo_ordering"] == ["height", "city_id_reind", "color"]
    assert (
        conditional_features["conditional_spec"]["city_id_reind"]["categorical_cond"]
        == []
    )
    assert conditional_features["conditional_spec"]["city_id_reind"][
        "continuous_cond"
    ] == ["height"]
    assert conditional_features["conditional_spec"]["height"]["continuous_cond"] == []
    assert conditional_features["conditional_spec"]["height"]["categorical_cond"] == []
    assert (
        conditional_features["conditional_spec"]["height"]["likelihood_dist"]
        == "multivariate_normal"
    )
    assert conditional_features["conditional_spec"]["height"]["parameters"] == {
        "size": 2
    }
    assert conditional_features["conditional_spec"]["height"]["feature_list"] == [
        "height1_std",
        "height2_std",
    ]
    assert conditional_features["conditional_spec"]["color"]["continuous_cond"] == [
        "height"
    ]
    assert conditional_features["conditional_spec"]["color"]["categorical_cond"] == [
        "city_id_reind"
    ]
    assert (
        conditional_features["conditional_spec"]["color"]["likelihood_dist"]
        == "multivariate_normal"
    )
    assert conditional_features["conditional_spec"]["color"]["parameters"] == {
        "size": 2
    }
    assert conditional_features["conditional_spec"]["color"]["feature_list"] == [
        "color1_minmax",
        "color2_minmax",
    ]


# ── forward_transform tests ───────────────────────────────────────────────────


def test_forward_transform_log_epsilon():
    params = LogEpsilonParams(
        epsilon=1e-6,
        mean=torch.tensor(5.0),
        std=torch.tensor(2.0),
    )
    # log(e^5 - 1e-6 + 1e-6) = 5; (5 - 5) / 2 = 0
    # log(e^7 - 1e-6 + 1e-6) = 7; (7 - 5) / 2 = 1
    raw = torch.tensor([np.e**5 - 1e-6, np.e**7 - 1e-6])
    out = forward_transform(raw, params)
    assert out.dtype == torch.float32
    assert abs(out[0].item() - 0.0) < 1e-4
    assert abs(out[1].item() - 1.0) < 1e-4


def test_forward_transform_standardize():
    params = StandardizeParams(mean=torch.tensor(3.0), std=torch.tensor(2.0))
    raw = torch.tensor([1.0, 3.0, 5.0])
    out = forward_transform(raw, params)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor([-1.0, 0.0, 1.0]))


def test_forward_transform_minmax():
    params = MinMaxParams(
        min_val=torch.tensor(0.0),
        range_val=torch.tensor(10.0),
        max_val=torch.tensor(10.0),
    )
    raw = torch.tensor([0.0, 5.0, 10.0])
    out = forward_transform(raw, params)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 1.0]))


def test_forward_transform_identity():
    params = IdentityParams()
    raw = torch.tensor([7.0, -3.5, 0.0])
    out = forward_transform(raw, params)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, raw.float())


def test_forward_transform_categorical_known_keys():
    params = CategoricalParams(mapping={10: 0, 20: 1, 30: 2}, n_cat=3)
    raw = torch.tensor([10, 20, 30, 10])
    out = forward_transform(raw, params)
    assert out.dtype == torch.int64
    assert out.tolist() == [0, 1, 2, 0]


def test_forward_transform_categorical_oov_returns_minus_one():
    params = CategoricalParams(mapping={10: 0, 20: 1}, n_cat=2)
    raw = torch.tensor([10, 99, 20])  # 99 is OOV
    out = forward_transform(raw, params)
    assert out.tolist() == [0, -1, 1]


def test_forward_transform_unsupported_params_raises():
    from pci.tools.data_processing import TransformParams

    class FakeParams(TransformParams):
        pass

    with pytest.raises(TypeError, match="unsupported params type"):
        forward_transform(torch.tensor([1.0]), FakeParams())


def test_forward_transform_preserves_device():
    params = StandardizeParams(mean=torch.tensor(0.0), std=torch.tensor(1.0))
    raw = torch.tensor([1.0, 2.0])
    out = forward_transform(raw, params)
    assert out.device == raw.device


def test_forward_transform_multidim_tensor():
    params = MinMaxParams(
        min_val=torch.tensor(0.0),
        range_val=torch.tensor(10.0),
        max_val=torch.tensor(10.0),
    )
    raw = torch.tensor([[0.0, 5.0], [10.0, 2.5]])
    out = forward_transform(raw, params)
    assert out.shape == raw.shape
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.5], [1.0, 0.25]]))


def test_forward_transform_roundtrip_standardize():
    """forward_transform then destandardize should recover original values."""
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    _, params_dict = standardize_tensor(raw)
    params = StandardizeParams(**params_dict)
    transformed = forward_transform(raw, params)
    recovered = destandardize_tensor(transformed, params)
    torch.testing.assert_close(recovered, raw, atol=1e-5, rtol=0)


def test_forward_transform_roundtrip_minmax():
    raw = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
    _, params_dict = min_max_scale_tensor(raw)
    params = MinMaxParams(**params_dict)
    transformed = forward_transform(raw, params)
    recovered = descale_tensor(transformed, params)
    torch.testing.assert_close(recovered, raw, atol=1e-5, rtol=0)


def test_forward_transform_roundtrip_log_epsilon():
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    _, params_dict = transform_log_epsilon(raw)
    params = LogEpsilonParams(**params_dict)
    transformed = forward_transform(raw, params)
    recovered = inverse_transform_log_epsilon(transformed, params)
    torch.testing.assert_close(recovered, raw, atol=1e-5, rtol=0)
