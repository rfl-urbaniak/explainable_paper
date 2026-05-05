import pyro
import pyro.distributions as dist
import pytest
import torch

from pci.explanation.excised import excise, sample_alternatives


@pytest.fixture
def factual_and_epsilon():
    factual = torch.tensor([[[1.5]], [[1.0]]])
    epsilon = torch.tensor(0.5)
    return factual, epsilon


@pytest.fixture
def true_parameters():
    mean = torch.tensor([[[1.0]], [[3.0]]])
    stddev = torch.tensor([[[2.0]], [[3.0]]])
    return mean, stddev


@pytest.fixture
def probs():
    return torch.tensor([0.2, 0.5, 0.3])


@pytest.mark.parametrize("interval_type", ["shaped", "shaped_multiple"])
@pytest.mark.parametrize(
    "dist_type,n_samples", [("normal", 3000), ("categorical", 4000)]
)
def test_excise_sampling(interval_type, dist_type, n_samples):
    key = "x"

    if dist_type == "normal":
        # continuous intervals for normal
        if interval_type == "shaped":
            intervals = [(torch.tensor(-1.0), torch.tensor(1.0))]
        else:  # shaped_multiple
            intervals = [
                (torch.tensor(-2.0), torch.tensor(-1.0)),
                (torch.tensor(1.0), torch.tensor(2.0)),
            ]
    else:
        # Categorical: avoid covering all indices, but still test interval avoidance
        if interval_type == "shaped":
            intervals = [
                (torch.tensor(0.5), torch.tensor(1.5))
            ]  # covers part of category 1
        else:  # shaped_multiple
            intervals = [
                (torch.tensor(-0.5), torch.tensor(0.5)),
                (torch.tensor(1.5), torch.tensor(2.5)),
            ]

    with excise(intervals={key: intervals}):
        with pyro.plate("sample", n_samples):
            if dist_type == "categorical":
                x = pyro.sample(
                    key, dist.Categorical(torch.tensor([1, 2, 3, 4, 5, 6, 4, 3, 2, 1]))
                )
            else:
                x = pyro.sample(key, dist.Normal(0, 1))

    assert x.shape[0] == n_samples

    for low, high in intervals:
        mask = (x < low) | (x > high)
        assert torch.all(mask), (
            f"Some samples fall inside excised interval {low}-{high}"
        )

    assert torch.unique(x).numel() > 1, "All samples are identical"


@pytest.mark.parametrize(
    "batch_size,dist_type",
    [
        (4000, "normal"),
        (4000, "categorical"),
    ],
)
def test_sample_alternatives_avoid_intervals(
    batch_size, dist_type, factual_and_epsilon, true_parameters
):
    factual, epsilon = factual_and_epsilon
    true_mean, true_stddev = true_parameters

    if dist_type == "categorical":
        probs = torch.tensor([[[[0.2, 0.5, 0.3]]], [[[0.1, 0.7, 0.2]]]])
    else:
        probs = None

    with pyro.plate("sample", batch_size, dim=-4):
        with sample_alternatives(
            factuals={"X": factual, "Y": factual}, epsilon=epsilon
        ):
            if dist_type == "normal":
                x = pyro.sample("X", dist.Normal(true_mean, true_stddev))
                y = pyro.sample("Y", dist.Normal(true_mean, true_stddev))
            else:
                x = pyro.sample("X", dist.Categorical(probs=probs))
                y = pyro.sample("Y", dist.Categorical(probs=probs))

    assert x.shape[0] == batch_size
    assert y.shape[0] == batch_size
    assert x.shape[-3:] == factual.shape
    assert y.shape[-3:] == factual.shape

    low = factual - epsilon
    high = factual + epsilon

    for var, name in [(x, "X"), (y, "Y")]:
        low_b = low.expand_as(var)
        high_b = high.expand_as(var)
        mask = (var < low_b) | (var > high_b)
        assert torch.all(mask), (
            f"Some {name} samples fall inside factual ± epsilon interval"
        )


def test_distribution_type_selector_for_normal():
    key = "x"
    intervals = [(torch.tensor(-0.5), torch.tensor(0.5))]

    base = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
    mask = torch.tensor([True, False, True])  # arbitrary mask
    distr = base.mask(mask)

    assert isinstance(distr, dist.MaskedDistribution)

    # first w/o selector, ensure raises NotImplementedError

    with pytest.raises(NotImplementedError):
        with pyro.plate("N", 1000):
            with excise(intervals={key: intervals}):
                pyro.sample(key, distr)

    with pyro.plate("N", 1000, dim=-2):
        with excise(
            intervals={key: intervals},
            distribution_type_selectors={key: "normal"},  # force interpretation
        ):
            x = pyro.sample(key, distr)

    assert torch.all((x < -0.5) | (x > 0.5))


def test_distribution_type_selector_for_categorical():
    key = "y"
    intervals = [(torch.tensor(0.5), torch.tensor(1.5))]

    base = dist.Categorical(logits=torch.tensor([1.0, 2.0, 3.0]))
    mask = torch.tensor([True, False, True])  # arbitrary mask
    distr = base.mask(mask)

    assert isinstance(distr, dist.MaskedDistribution)

    with pytest.raises(NotImplementedError):
        with pyro.plate("N", 1000):
            with excise(intervals={key: intervals}):
                pyro.sample(key, distr)

    # --- now with selector, should succeed ---
    with pyro.plate("N", 1000, dim=-2):
        with excise(
            intervals={key: intervals},
            distribution_type_selectors={key: "categorical"},  # force interpretation
        ):
            y = pyro.sample(key, distr)

    assert torch.all((y < 0.5) | (y > 1.5))


def test_sample_alternatives_with_type_selector_for_normal():
    key = "x"
    factual = torch.tensor([[[0.0]]])
    epsilon = torch.tensor(0.5)

    base = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
    mask = torch.tensor([True, False, True])  # arbitrary mask
    distr = base.mask(mask)

    assert isinstance(distr, dist.MaskedDistribution)

    # Without selector → should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        with pyro.plate("N", 500):
            with sample_alternatives(factuals={key: factual}, epsilon=epsilon):
                pyro.sample(key, distr)

    # With selector → should succeed
    with pyro.plate("N", 500, dim=-2):
        with sample_alternatives(
            factuals={key: factual},
            epsilon=epsilon,
            distribution_type_selectors={key: "normal"},  # force interpretation
        ):
            x = pyro.sample(key, distr)

    # Ensure sampled values avoid the factual ± epsilon interval
    low = factual - epsilon
    high = factual + epsilon
    low_b = low.expand_as(x)
    high_b = high.expand_as(x)
    mask_result = (x < low_b) | (x > high_b)
    assert torch.all(mask_result), "Some samples fall inside factual ± epsilon interval"


def test_sample_alternatives_with_type_selector_for_categorical(factual_and_epsilon):
    factual, epsilon = factual_and_epsilon

    key = "Y"
    # batch/event shape matches factual
    probs = torch.tensor([[[[0.2, 0.5, 0.3]]], [[[0.1, 0.7, 0.2]]]])  # shape [2,1,1,3]

    base = dist.Categorical(probs=probs)
    mask = torch.tensor([True, False])  # arbitrary mask along batch dim
    distr = base.mask(mask)

    assert isinstance(distr, dist.MaskedDistribution)

    # Without selector → should raise
    with pytest.raises(NotImplementedError):
        with pyro.plate("N", 500, dim=-4):
            with sample_alternatives(factuals={key: factual}, epsilon=epsilon):
                pyro.sample(key, distr)

    # With selector → should succeed
    with pyro.plate("N", 500, dim=-4):
        with sample_alternatives(
            factuals={key: factual},
            epsilon=epsilon,
            distribution_type_selectors={key: "categorical"},
        ):
            y = pyro.sample(key, distr)

    # Ensure sampled values avoid factual ± epsilon
    low_b = (factual - epsilon).expand_as(y)
    high_b = (factual + epsilon).expand_as(y)
    mask_result = (y < low_b) | (y > high_b)
    assert torch.all(mask_result), "Some samples fall inside factual ± epsilon interval"
