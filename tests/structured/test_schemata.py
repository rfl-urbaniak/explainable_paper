import pandas as pd
import pytest

from pci.structured.schemata import Cat, Flow, Norm, Schema
from pci.tools.data_processing import (
    CategoricalParams,
    LogEpsilonParams,
    MinMaxParams,
    StandardizeParams,
    data_to_transformed_tensors,
)


def get_setup():
    df = pd.DataFrame(
        {
            "age": [25, 40, 35],
            "height": [5.5, 6.0, 5.8],
            "zip": [1001, 1002, 1001],
            "income": [50000, 80000, 60000],
            "lon": [-71.0589, -74.0060, -73.9352],
            "lat": [42.3601, 40.7128, 40.7306],
        }
    )

    continuous_features = {
        "age": "standardize",
        "income": "log-epsilon",
        "lon": "standardize",
        "lat": "standardize",
        "height": "minmax",
    }

    categorical_features = {
        "zip": "reset_index",
    }

    data_dict = data_to_transformed_tensors(
        inference_df=df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )

    return df, continuous_features, categorical_features, data_dict


@pytest.fixture
def setup():
    return get_setup()


def test_schemata_construction(setup):
    df, continuous_features, categorical_features, data_dict = setup

    distribution_map = {
        "zip": Cat,
        "age": Norm,
        "height_income": Norm,  # this will be 2d Norm
        "lon_lat": Flow,  # combine lon + lat here, pass to Flow
    }

    feature_names = [
        "zip",
        "age",
        "weight",  # not in elem map, will default to Norm
        "height_income",
        "lon_lat",
    ]

    feature_map = {
        "height_income": ["height", "income"],  # One Norm will bundle these two
        "lon_lat": ["lon", "lat"],  # Flow will bundle these two
    }

    schema = Schema(
        feature_names=feature_names,
        params=data_dict["transformation_params"],  # still per original feature
        distribution_map=distribution_map,
        feature_map=feature_map,
    )

    assert schema["zip"].n_cat == 3  # two unique zip codes in data plus one for unseen
    assert isinstance(schema["age"], Norm)
    assert isinstance(schema["height_income"], Norm)
    assert isinstance(schema["lon_lat"], Flow)
    assert isinstance(schema["weight"], Norm)  # defaulted
    for name in feature_names:
        assert schema[name].feature_name == name
    for name in ["zip", "age", "weight"]:
        assert schema[name].component_feature_names == [name]

    assert schema["lon_lat"].component_feature_names == ["lon", "lat"]
    assert schema["height_income"].component_feature_names == ["height", "income"]


# need to test the data_to_transformed_tensors function
# as it now involves transformation param dataclasses


def test_continuous_transformed_columns_exist(setup):
    df, continuous_features, categorical_features, data_dict = setup

    for col in continuous_features.keys():
        transformed_col = f"{col}_transformed"
        assert transformed_col in data_dict["continuous"], f"{transformed_col} missing"
        assert (
            data_dict["continuous"][transformed_col].shape
            == data_dict["continuous"][col].shape
        )


def test_categorical_transformed_columns_exist(setup):
    df, continuous_features, categorical_features, data_dict = setup

    for col in categorical_features.keys():
        transformed_col = f"{col}_transformed"
        assert transformed_col in data_dict["categorical"], f"{transformed_col} missing"
        assert (
            data_dict["categorical"][transformed_col].shape
            == data_dict["categorical"][col].shape
        )


def test_transformation_params_types(setup):
    df, continuous_features, categorical_features, data_dict = setup

    for col in continuous_features.keys():
        transformed_col = f"{col}_transformed"
        params = data_dict["transformation_params"][transformed_col]
        if continuous_features[col] == "standardize":
            assert isinstance(params, StandardizeParams)
        elif continuous_features[col] == "log-epsilon":
            assert isinstance(params, LogEpsilonParams)
        elif continuous_features[col] == "minmax":
            assert isinstance(params, MinMaxParams)

    for col in categorical_features.keys():
        transformed_col = f"{col}_transformed"
        params = data_dict["transformation_params"][transformed_col]
        assert isinstance(params, CategoricalParams)
        # Check that mapping length matches n_cat
        assert len(params.mapping) == params.n_cat


def test_categorical_mapping_values(setup):
    df, continuous_features, categorical_features, data_dict = setup

    # For zip column
    transformed_col = "zip_transformed"
    cat_params = data_dict["transformation_params"][transformed_col]
    # The mapping should contain all unique values
    unique_values = df["zip"].unique()
    for val in unique_values:
        assert val in cat_params.mapping
        assert 0 <= cat_params.mapping[val] < cat_params.n_cat


def test_reuse_transformation_params_identity(setup):
    df, continuous_features, categorical_features, data_dict = setup

    # Keep original params dict for comparison
    original_params = data_dict["transformation_params"]

    # Run again with same params
    reused_dict = data_to_transformed_tensors(
        inference_df=df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        transformation_params=original_params,
    )

    # The output params for each feature should be the **same objects**
    # (i.e., the function reused the provided dataclass instances)
    for col in continuous_features.keys() | categorical_features.keys():
        transformed_col = f"{col}_transformed"
        assert (
            original_params[transformed_col]
            == reused_dict["transformation_params"][transformed_col]
        )
