from collections import UserDict
from collections.abc import Collection


class SchemaElem:
    """Base descriptor for a single sample-site distribution in a schema.

    Associates a feature with the distribution used at its sample site and with
    the underlying component features it draws on. Subclasses specialise the
    distribution family (categorical, normal, normalizing flow).
    """

    def __init__(
        self,
        feature_name: str,
        params: dict,
        component_feature_names: Collection[str] | None = None,
    ):
        """Initialise the schema element.

        :param feature_name: Name of the feature this element describes.
        :param params: Transformation parameters keyed by feature.
        :param component_feature_names: Underlying component feature(s) backing
            this element; defaults to a single component equal to ``feature_name``.
        """
        self.feature_name = feature_name
        self.component_feature_names = (
            list(component_feature_names)
            if component_feature_names is not None
            else [feature_name]
        )


class Cat(SchemaElem):
    """Schema element for a categorical sample site.

    Reads the number of categories from the feature's transformation parameters.
    Single-component only; combining multiple component features is not supported.
    """

    n_cat: int

    def __init__(
        self,
        feature_name: str,
        params: dict,
        component_feature_names: Collection[str] | None = None,
    ):
        """Initialise the categorical schema element.

        :param feature_name: Name of the categorical feature.
        :param params: Transformation parameters; must contain an entry for
            ``"{feature_name}_transformed"`` carrying the category count.
        :param component_feature_names: Must be omitted; categorical elements
            do not support multiple component features.
        :raises ValueError: If component feature names are supplied.
        :raises AssertionError: If the feature's transformation parameters are missing.
        """
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
    """Schema element for a normally distributed sample site."""

    def __init__(
        self,
        feature_name: str,
        params: dict,
        component_feature_names: Collection[str] | None = None,
    ):
        """Initialise the normal schema element.

        :param feature_name: Name of the feature this element describes.
        :param params: Transformation parameters keyed by feature.
        :param component_feature_names: Underlying component feature(s) backing
            this element; defaults to a single component equal to ``feature_name``.
        """
        super().__init__(feature_name, params, component_feature_names)


class Flow(SchemaElem):
    """Schema element for a sample site modelled by a normalizing flow.

    Inherits the base element behaviour without specialisation, marking the site
    as flow-distributed.
    """

    pass


class Schema(UserDict[str, "SchemaElem"]):
    """Mapping from feature name to its sample-site schema element.

    Aggregates one schema element per feature, selecting the distribution family
    per feature (defaulting to normal) and wiring each element to its component
    features. Behaves as a dict keyed by feature name.
    """

    def __init__(
        self,
        params: dict,
        feature_names: Collection[str],
        distribution_map: dict[str, type["SchemaElem"]] | None = None,
        feature_map: dict[str, list[str]] | None = None,
    ):
        """Build a schema, one element per feature.

        :param params: Transformation parameters keyed by feature.
        :param feature_names: Feature names that become the schema keys.
        :param distribution_map: Per-feature override mapping a feature name to
            the distribution family to use; features absent from the map default
            to normal.
        :param feature_map: Per-feature override mapping a feature name to its
            underlying component feature(s).
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
