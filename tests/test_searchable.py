import pyro
import pyro.distributions as dist
import torch

from pci.explanation.searchable import SearchableModel
from pci.explanation.thin_search import ThinSearchSampler


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

        return {"categorical": {"y": y}, "continuous": {"x": x}}

    return toy_model


def test_toy_searchable():
    batch_size = 3
    num_samples = 2
    sampling_dim = -3

    toy_model = toy_factory(sampling_dim)

    kwargs_iterable = kwargs_iterable = [
        {"observations_dict": None, "n_size": batch_size},
        dict(),
        dict(),
    ]

    factual_dict = toy_model(kwargs_iterable)

    toy_searchable = SearchableModel(
        structured_model=toy_model,
        sites_of_interest=["x", "y"],
        suspects=["x"],
    )

    for conditioned_alternative in [True, False]:
        for factual_exclusion in [True, False]:
            search_sampler = ThinSearchSampler(
                structured_model=toy_searchable,
                conditioned_alternatives=conditioned_alternative,
                factual_exclusion=factual_exclusion,
            )

            results = search_sampler.sample(
                factual_dict,
                num_samples=num_samples,
            )

            assert results["sufficiency"]["y"].shape == (num_samples, batch_size, 1, 1)
