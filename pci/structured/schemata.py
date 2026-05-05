from collections import UserDict
from collections.abc import Collection


class SchemaElem:
    def __init__(
        self,
        feature_name: str,
        params: dict,
        component_feature_names: Collection[str] | None = None,
    ):
        self.feature_name = feature_name
        self.component_feature_names = (
            list(component_feature_names)
            if component_feature_names is not None
            else [feature_name]
        )


class Cat(SchemaElem):
    n_cat: int

    def __init__(
        self,
        feature_name: str,
        params: dict,
        component_feature_names: Collection[str] | None = None,
    ):
        if component_feature_names is not None:
            raise ValueError("Cat does not support multiple component features")

        super().__init__(feature_name, params)

        xform_name = f"{feature_name}_transformed"
        assert xform_name in params, (
            f"Categorical feature '{feature_name}' missing transformation params"
        )

        cat_params = params[xform_name]
        self.n_cat = cat_params.n_cat


class Norm(SchemaElem):
    def __init__(
        self,
        feature_name: str,
        params: dict,
        component_feature_names: Collection[str] | None = None,
    ):
        super().__init__(feature_name, params, component_feature_names)


class Flow(SchemaElem):
    pass


class Schema(UserDict[str, "SchemaElem"]):
    def __init__(
        self,
        params: dict,
        feature_names: Collection[str],
        distribution_map: dict[str, type["SchemaElem"]] | None = None,
        feature_map: dict[str, list[str]] | None = None,
    ):
        """
        feature_names: schema keys
        params: transformation params
        distribution_map: schema key -> SchemaElem class (default: Norm)
        feature_map: schema key -> component feature(s)
        """
        distribution_map = distribution_map or {}
        feature_map = feature_map or {}

        result: dict[str, SchemaElem] = {}

        for name in feature_names:
            distribution_schema = distribution_map.get(name, Norm)
            mapped = feature_map.get(name)

            result[name] = distribution_schema(
                feature_name=name,
                params=params,
                component_feature_names=mapped,
            )

        super().__init__(result)
