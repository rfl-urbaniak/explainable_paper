import pyro
import pyro.distributions as dist
import pytest
import torch

from pci.explanation.utils import (
    broadcast_mask,
    get_alternative_sample,
    sample_k_indices,
)


def toy_factory(sampling_dim):
    def toy_model(
        kwargs_iterable=[
            {"observations_dict": None, "n_size": 1},
            dict(),
            dict(),
        ],
    ):
        # for compatibility with a structured model, we assume it takes kwargs_iterable,
        # normally expected to be passed to a model built using ModelComposer
        batch_size = kwargs_iterable[0]["n_size"]
        with pyro.plate("sample", size=batch_size, dim=sampling_dim):
            x = pyro.sample(
                "x",
                dist.Normal(0, 1),
            )

            y = pyro.sample(
                "y",
                dist.Categorical(logits=torch.ones(2)),
            )

        return {"x": x, "y": y}

    return toy_model


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("sampling_dim", [-1, -2, -3])
@pytest.mark.parametrize("equality_epsilon", [0.01, 0.05])
def test_alternative_samples(batch_size, sampling_dim, equality_epsilon):
    toy_model = toy_factory(sampling_dim)
    kwargs_iterable = kwargs_iterable = [
        {"observations_dict": None, "n_size": batch_size},
        dict(),
        dict(),
    ]

    factual_dict = toy_model(kwargs_iterable)

    active_antecedents = ["x", "y"]

    samples = get_alternative_sample(
        structured_model=toy_model,
        n_size=batch_size,
        active_antecedents=active_antecedents,
        factual_dictionary=factual_dict,
        batch_dim=sampling_dim,
        equality_epsilon=equality_epsilon,
    )

    for key in ["x", "y"]:
        assert samples[key].shape == factual_dict[key].shape
        assert not torch.allclose(
            samples[key], factual_dict[key], atol=equality_epsilon
        ), "Sample matches factual at some point!"


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("sampling_dim", [-1, -2, -3])
@pytest.mark.parametrize("equality_epsilon", [0.01, 0.05])
def test_alternative_samples_conditioned(batch_size, sampling_dim, equality_epsilon):
    toy_model = toy_factory(sampling_dim)
    kwargs_iterable = kwargs_iterable = [
        {"observations_dict": None, "n_size": batch_size},
        dict(),
        dict(),
    ]

    factual_dict = toy_model(kwargs_iterable)

    observations_dict = {
        "continuous": {"x": torch.ones_like(factual_dict["x"])},
        "categorical": {},
    }

    new_kwargs_iterable = [
        {"observations_dict": observations_dict, "n_size": batch_size},
        dict(),
        dict(),
    ]

    active_antecedents = ["y"]

    samples_conditioned = get_alternative_sample(
        structured_model=toy_model,
        n_size=batch_size,
        active_antecedents=active_antecedents,
        factual_dictionary=factual_dict,
        batch_dim=sampling_dim,
        equality_epsilon=equality_epsilon,
        kwargs_iterable=new_kwargs_iterable,
    )

    assert samples_conditioned["y"].shape == factual_dict["y"].shape
    assert not torch.allclose(
        samples_conditioned["y"], factual_dict["y"], atol=equality_epsilon
    ), "Sample matches factual at some point!"


def test_broadcast_mask_basic():
    mask = torch.tensor([1, 0, 1])
    target = torch.zeros(3, 4)
    result = broadcast_mask(mask, target)
    expected = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    assert torch.equal(result, expected)


def test_broadcast_mask_different_axis():
    mask = torch.tensor([1, 0])
    target = torch.zeros(4, 2, 3)
    result = broadcast_mask(mask, target)
    expected = mask.view(1, 2, 1).expand(4, 2, 3)
    assert torch.equal(result, expected)


def test_broadcast_mask_multiple_dims_error():
    mask = torch.tensor([2, 3])
    target = torch.zeros(2, 2, 3)
    with pytest.raises(ValueError):
        broadcast_mask(mask, target)


def test_broadcast_mask_no_matching_dim_error():
    mask = torch.tensor([1, 2, 3])
    target = torch.zeros(4, 5)
    with pytest.raises(ValueError):
        broadcast_mask(mask, target)


def test_broadcast_mask_singleton_target():
    mask = torch.tensor([1])
    target = torch.zeros(1, 3, 2)
    result = broadcast_mask(mask, target)
    expected = mask.view(1, 1, 1).expand(1, 3, 2)
    assert torch.equal(result, expected)


def test_broadcast_mask_higher_dim():
    mask = torch.tensor([1, 0, 1])
    target = torch.zeros(2, 3, 4)
    result = broadcast_mask(mask, target)
    expected = mask.view(1, 3, 1).expand(2, 3, 4)
    assert torch.equal(result, expected)


def test_sample_k_indices_basic():
    indices = sample_k_indices(k_min=1, k_max=4, n=10, sample_size=5)
    assert len(indices) == 5  # sample_size
    for idx in indices:
        num_selected = idx.sum().item()
        assert 1 <= num_selected <= 4  # k_min..k_max


def test_sample_k_indices_invert():
    indices = sample_k_indices(
        k_min=1, k_max=4, n=10, sample_size=5, invert_selection=True
    )
    assert len(indices) == 5  # sample_size
    for idx in indices:
        num_selected = idx.sum().item()
        assert 6 <= num_selected <= 9  # n - k_max..n - k_min = 10-4..10-1


def test_sample_k_indices_zero_selection():
    # Setting k_min=k_max=0 should produce all-zero masks
    indices = sample_k_indices(k_min=0, k_max=0, n=5, sample_size=3)
    for idx in indices:
        assert idx.sum() == 0
        assert torch.all(idx == 0)


def test_sample_k_indices_large_n():
    n = 100
    indices = sample_k_indices(k_min=5, k_max=10, n=n, sample_size=5)
    for idx in indices:
        assert idx.sum() >= 5 and idx.sum() <= 10
        assert idx.numel() == n


def test_sample_k_indices_single_sample():
    indices = sample_k_indices(k_min=2, k_max=2, n=10, sample_size=1)
    assert len(indices) == 1
    assert indices[0].sum() == 2
    assert indices[0].numel() == 10
