import numpy as np
import pyro
import pyro.distributions as dist
import pyro.nn
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition
from torch.optim import Adam

from pci.structured.distribution_neural import (
    DataEmbedder,
    NeuralDistributionConditionalCategorical,
    NeuralDistributionConditionalNormal,
    NeuralDistributionUnconditionalCategorical,
    NeuralDistributionUnconditionalNormal,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 50
pyro.set_rng_seed(42)
torch.manual_seed(42)
np.random.seed(42)


def make_synthetic_data(n, device):
    cat1 = torch.randint(0, 3, (n, 1), device=device)
    cat2 = torch.randint(0, 5, (n, 1), device=device)

    cont1 = torch.randn(n, 2, device=device)
    cont2 = torch.randn(n, 1, device=device)

    outcome_mean = (
        0.5 * cat1.float()
        - 0.3 * cat2.float()
        + torch.sin(cont1[:, [0]]) * 0.8
        + cont2 * 1.2
        + 0.1 * torch.randn(n, 1, device=device)
    )

    outcome_sampled = dist.Normal(outcome_mean, 0.1).sample()

    outcome_cat_logits = torch.cat(
        [
            0.4 * cat1.float() + 0.2 * cont1[:, [0]],  # category 0 logits
            -0.3 * cat2.float() + 0.1 * cont1[:, [1]],  # category 1 logits
            0.2 * cont2,  # category 2 logits
            -0.1 * cat1.float() + 0.3 * cont2,  # category 3 logits
        ],
        dim=1,
    )

    outcome_cat = dist.Categorical(logits=outcome_cat_logits).sample()

    data = {
        "categorical": {
            "cat1": cat1.unsqueeze(-1),
            "cat2": cat2.unsqueeze(-2),
            "outcome_cat": outcome_cat.unsqueeze(-1).unsqueeze(-1),
        },
        "continuous": {
            "cont1": cont1.unsqueeze(-2),
            "cont2": cont2.unsqueeze(-2),
            "outcome": outcome_sampled.unsqueeze(-2),
        },
    }

    return data


def quick_training_unconditional_loop(neural_unconditional, condition_dict):
    optimizer = Adam(neural_unconditional.parameters(), lr=0.05)
    epochs = 2
    for epoch in range(epochs):
        optimizer.zero_grad()
        with pyro.poutine.trace() as tr:
            with condition(data=condition_dict):
                neural_unconditional()
        tr.trace.compute_log_prob()
        loss = -tr.trace.log_prob_sum() / n
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()


class MiniModel(pyro.nn.PyroModule):
    def __init__(
        self,
        input_cat_cardinalities: dict[str, int],
        cat_embed_dims: dict[str, int],
        in_channels: int = 5,
        event_dim: int = 1,
    ):
        super().__init__()

        self.data_embedder = DataEmbedder(
            input_cat_cardinalities=input_cat_cardinalities,
            cat_embed_dims=cat_embed_dims,
        ).to(device)

        self.neural_normal = NeuralDistributionConditionalNormal(
            outcome_name="outcome",
            in_channels=in_channels,
            event_dim=event_dim,
        ).to(device)

        self.neural_categorical = NeuralDistributionConditionalCategorical(
            outcome_name="outcome_cat",
            num_outcome_cat=4,
            in_channels=in_channels,
        ).to(device)

        self.event_dim = event_dim

    def forward(self, **kwargs):
        for key in kwargs:
            kwargs[key] = pyro.deterministic(key, kwargs[key])

        embedded_x = self.data_embedder(**kwargs)

        sampled = self.neural_normal(embedded_x)
        sampled_cat = self.neural_categorical(embedded_x)

        return sampled, sampled_cat


def test_unconditional_neural_categorical_training():
    synthetic_logits = torch.tensor([0.2, 0.5, 0.3]).view(1, 1, 3).to(device)

    with pyro.plate("data", size=n, dim=-2):
        y_sampled = pyro.sample(
            "y_sampled",
            dist.Categorical(logits=synthetic_logits).to_event(1),
        )

    neural_unconditional = NeuralDistributionUnconditionalCategorical(
        outcome_name="y",
        num_outcome_cat=3,
        event_dim=1,
    ).to(device)

    # unconditoned forward pass
    with pyro.plate("data", size=n, dim=-2):
        with pyro.poutine.trace() as tr_unconditional:
            y_unconditioned = neural_unconditional()

    assert y_unconditioned.shape == (n, 1, 1)
    assert tr_unconditional.trace.nodes["y"]["value"].shape == (n, 1, 1)

    condition_dict = {"y": y_sampled}

    with condition(data=condition_dict):
        with pyro.poutine.trace() as tr_conditioned:
            y_conditioned = neural_unconditional()

    assert y_conditioned.shape == (n, 1, 1)
    assert tr_conditioned.trace.nodes["y"]["value"].shape == (n, 1, 1)
    assert torch.equal(tr_conditioned.trace.nodes["y"]["value"], y_sampled)

    with MultiWorldCounterfactual(first_available_dim=-3):
        with do(actions=condition_dict):
            with pyro.plate("data", size=n, dim=-2):
                with pyro.poutine.trace() as tr_mwc:
                    y_mwc = neural_unconditional()

    assert y_mwc.shape == (2, n, 1, 1)
    assert tr_mwc.trace.nodes["y"]["value"].shape == (2, n, 1, 1)
    assert torch.equal(y_mwc[1, ...], y_sampled)

    quick_training_unconditional_loop(
        neural_unconditional=neural_unconditional, condition_dict=condition_dict
    )

    with pyro.plate("data", size=n, dim=-2):
        with pyro.poutine.trace() as tr_after_training:
            y_from_posterior = neural_unconditional()

    assert y_from_posterior.shape == (n, 1, 1)
    assert tr_after_training.trace.nodes["y"]["value"].shape == (n, 1, 1)

    # smoke test: just verify forward pass runs after training
    assert y_from_posterior.shape == (n, 1, 1)


def test_unconditional_neural_normal_training():
    neural_unconditional = NeuralDistributionUnconditionalNormal(
        outcome_name="y",
        event_dim=1,
    ).to(device)

    with pyro.plate("data", size=n, dim=-2):
        y_sampled = pyro.sample(
            "y_sampled",
            dist.Normal(
                torch.tensor([1.0]).view(1, 1, 1).to(device),
                torch.tensor([1.5]).view(1, 1, 1).to(device),
            ).to_event(1),
        )

    with pyro.plate("data", size=n, dim=-2):
        with pyro.poutine.trace() as tr_unconditional:
            y_unconditional = neural_unconditional()

    assert y_unconditional.shape == (n, 1, 1)
    assert tr_unconditional.trace.nodes["y"]["value"].shape == (n, 1, 1)
    assert not torch.equal(tr_unconditional.trace.nodes["y"]["value"], y_sampled)

    condition_dict = {"y": y_sampled}

    with condition(data=condition_dict):
        with pyro.poutine.trace() as tr_conditioned:
            y_conditioned = neural_unconditional()

    assert y_conditioned.shape == (n, 1, 1)
    assert tr_conditioned.trace.nodes["y"]["value"].shape == (n, 1, 1)
    assert torch.equal(tr_conditioned.trace.nodes["y"]["value"], y_sampled)

    with MultiWorldCounterfactual(first_available_dim=-3):
        with do(actions=condition_dict):
            with pyro.plate("data", size=n, dim=-2):
                with pyro.poutine.trace() as tr_mwc:
                    y_mwc = neural_unconditional()

    assert y_mwc.shape == (2, n, 1, 1)
    assert tr_mwc.trace.nodes["y"]["value"].shape == (2, n, 1, 1)
    assert torch.equal(y_mwc[1, ...], y_sampled)

    quick_training_unconditional_loop(neural_unconditional, condition_dict)

    with pyro.plate("data", size=n, dim=-2):
        with pyro.poutine.trace() as tr_trained:
            y_from_posterior = neural_unconditional()

    assert y_from_posterior.shape == (n, 1, 1)
    assert tr_trained.trace.nodes["y"]["value"].shape == (n, 1, 1)

    # smoke test: verify shape after training
    assert tr_trained.trace.nodes["y"]["value"].shape == (n, 1, 1)


def test_conditional_neural_training():
    data = make_synthetic_data(n=n, device=device)

    model = MiniModel(
        input_cat_cardinalities={"cat1": 3, "cat2": 5},
        cat_embed_dims={
            "cat1": 1,
            "cat2": 1,
        },
        in_channels=5,
    ).to(device)

    sampled_x, sampled_x_cat = model(
        cat1=data["categorical"]["cat1"],
        cat2=data["categorical"]["cat2"],
        cont1=data["continuous"]["cont1"],
        cont2=data["continuous"]["cont2"],
    )

    assert sampled_x.shape == (n, 1, 1)
    assert sampled_x_cat.shape == (n, 1, 1)

    with pyro.poutine.trace() as tr:
        model(
            cat1=data["categorical"]["cat1"],
            cat2=data["categorical"]["cat2"],
            cont1=data["continuous"]["cont1"],
            cont2=data["continuous"]["cont2"],
        )

    assert not torch.equal(
        tr.trace.nodes["outcome"]["value"], data["continuous"]["outcome"]
    )

    assert not torch.equal(
        tr.trace.nodes["outcome_cat"]["value"], data["categorical"]["outcome_cat"]
    )

    with pyro.poutine.trace() as tr_obs:
        with condition(
            data={
                "outcome": data["continuous"]["outcome"],
                "outcome_cat": data["categorical"]["outcome_cat"],
            }
        ):
            model(
                cat1=data["categorical"]["cat1"],
                cat2=data["categorical"]["cat2"],
                cont1=data["continuous"]["cont1"],
                cont2=data["continuous"]["cont2"],
            )

    assert tr_obs.trace.nodes["outcome"]["value"].shape == (n, 1, 1)
    assert tr_obs.trace.nodes["outcome_cat"]["value"].shape == (n, 1, 1)

    assert torch.equal(
        tr_obs.trace.nodes["outcome"]["value"], data["continuous"]["outcome"]
    )

    assert torch.equal(
        tr_obs.trace.nodes["outcome_cat"]["value"], data["categorical"]["outcome_cat"]
    )

    with MultiWorldCounterfactual(first_available_dim=-3):
        with do(
            actions={
                "outcome": data["continuous"]["outcome"],
                "outcome_cat": data["categorical"]["outcome_cat"],
            }
        ):
            with pyro.poutine.trace() as tr_mwc:
                model(
                    cat1=data["categorical"]["cat1"],
                    cat2=data["categorical"]["cat2"],
                    cont1=data["continuous"]["cont1"],
                    cont2=data["continuous"]["cont2"],
                )

    assert tr_mwc.trace.nodes["outcome"]["value"].shape == (2, n, 1, 1)
    assert tr_mwc.trace.nodes["outcome_cat"]["value"].shape == (
        2,
        1,
        n,
        1,
        1,
    )  # two intervened sites
    assert torch.equal(
        tr_mwc.trace.nodes["outcome"]["value"][1, ...], data["continuous"]["outcome"]
    )

    def quick_training_loop():
        optimizer = Adam(model.parameters(), lr=0.001)
        epochs = 2
        for epoch in range(epochs):
            optimizer.zero_grad()
            with pyro.poutine.trace() as tr:
                with condition(
                    data={
                        "outcome": data["continuous"]["outcome"],
                        "outcome_cat": data["categorical"]["outcome_cat"],
                    }
                ):
                    model(
                        cat1=data["categorical"]["cat1"],
                        cat2=data["categorical"]["cat2"],
                        cont1=data["continuous"]["cont1"],
                        cont2=data["continuous"]["cont2"],
                    )

            tr.trace.compute_log_prob()

            loss = -tr.trace.log_prob_sum() / 10_000

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                print(
                    "Mean categorical loss:",
                    tr.trace.nodes["outcome_cat"]["log_prob"].mean().item(),
                )

                # assert absolute value is not larger than 10
                # to catch potential issues with `to_event` handling
                assert abs(tr.trace.nodes["outcome_cat"]["log_prob"].mean().item()) < 10

                print(
                    "Mean continuous loss:",
                    tr.trace.nodes["outcome"]["log_prob"].mean().item(),
                )

            loss.backward()
            optimizer.step()

    quick_training_loop()

    test_data = make_synthetic_data(n=1000, device=device)
    predicted, predicted_cat = model(
        cat1=test_data["categorical"]["cat1"],
        cat2=test_data["categorical"]["cat2"],
        cont1=test_data["continuous"]["cont1"],
        cont2=test_data["continuous"]["cont2"],
    )

    assert predicted.shape == (1000, 1, 1)
    assert predicted_cat.shape == (1000, 1, 1)


def test_conditional_normal_shapes():
    data = make_synthetic_data(n=n, device=device)

    model = MiniModel(
        input_cat_cardinalities={"cat1": 3, "cat2": 5},
        cat_embed_dims={
            "cat1": 1,
            "cat2": 1,
        },
        in_channels=5,
    ).to(device)

    int_cat1 = torch.ones(n, 1, 1).to(device)
    int_cat1 = int_cat1.long()

    int_con = torch.randn(n, 1, 1).to(device)

    with (
        MultiWorldCounterfactual(first_available_dim=-2),
        do(actions={"cat1": int_cat1, "cont2": int_con}),
        pyro.poutine.trace() as tr_intervened,
    ):
        model(
            cat1=data["categorical"]["cat1"],
            cat2=data["categorical"]["cat2"],
            cont1=data["continuous"]["cont1"],
            cont2=data["continuous"]["cont2"],
        )

    assert tr_intervened.trace.nodes["cat1"]["value"].shape == (2, 1, n, 1, 1)
    assert torch.allclose(
        data["categorical"]["cat1"], tr_intervened.trace.nodes["cat1"]["value"][0, ...]
    )
    assert torch.allclose(int_cat1, tr_intervened.trace.nodes["cat1"]["value"][1, ...])

    assert tr_intervened.trace.nodes["cont2"]["value"].shape == (
        2,
        1,
        1,  # note extra dim here (two intervened sites)
        n,
        1,
        1,
    )

    assert torch.allclose(
        data["continuous"]["cont2"], tr_intervened.trace.nodes["cont2"]["value"][0, ...]
    )
    assert torch.allclose(int_con, tr_intervened.trace.nodes["cont2"]["value"][1, ...])

    assert tr_intervened.trace.nodes["outcome"]["value"].shape == (2, 2, 1, n, 1, 1)
    assert not torch.allclose(
        tr_intervened.trace.nodes["outcome"]["value"][0, ...],
        tr_intervened.trace.nodes["outcome"]["value"][1, ...],
    )

    with (
        MultiWorldCounterfactual(first_available_dim=-3),
        condition(data={"outcome": data["continuous"]["outcome"]}),
        do(actions={"cat1": int_cat1}),
        pyro.poutine.trace() as tr_intervened_conditioned,
    ):
        model(
            cat1=data["categorical"]["cat1"],
            cat2=data["categorical"]["cat2"],
            cont1=data["continuous"]["cont1"],
            cont2=data["continuous"]["cont2"],
        )
    assert tr_intervened_conditioned.trace.nodes["cat1"]["value"].shape == (
        2,
        1,
        1,
        n,
        1,
        1,
    )
    assert torch.allclose(
        data["categorical"]["cat1"],
        tr_intervened_conditioned.trace.nodes["cat1"]["value"][0, ...],
    )
    assert torch.allclose(
        int_cat1, tr_intervened_conditioned.trace.nodes["cat1"]["value"][1, ...]
    )
    assert tr_intervened_conditioned.trace.nodes["outcome"]["value"].shape == (n, 1, 1)

    assert torch.allclose(
        data["continuous"]["outcome"],
        tr_intervened_conditioned.trace.nodes["outcome"]["value"],
    )


# # LEAVE THESE HANGING FOR NOW UNTIL RSAMPLE IS ADDED

# # TODO add base noise test for the rsampled version when rsample lands

# #     n = normal_model

# #     # base noise should be shared across possible worlds
# #     with (
# #         MultiWorldCounterfactual(first_available_dim=-3),
# #         do(actions={"training_x1": torch.ones(1, 1)}),
# #         pyro.poutine.trace() as tr_obs,
# #     ):
# #         with pyro.plate("setup_samples", size=4, dim=-2):
# #             training_x, training_y = setup_regression_problem(n)

# #         n.sample(training_x, training_y)

# #     with (
# #         MultiWorldCounterfactual(first_available_dim=-3),
# #         do(actions={"training_x1": torch.ones(1, 1)}),
# #         pyro.poutine.trace() as tr_nobs,
# #     ):
# #         with pyro.plate("setup_samples", size=4, dim=-2):
# #             training_x, _ = setup_regression_problem(n)
# #         n.sample(training_x)

# #     tr_obs.trace.compute_log_prob()
# #     tr_nobs.trace.compute_log_prob()

# #     nodes_obs = tr_obs.trace.nodes
# #     nodes_nobs = tr_nobs.trace.nodes

# #     assert (
# #         nodes_obs["y_base_noise"]["value"].shape
# #         == nodes_nobs["y_base_noise"]["value"].shape
# #     )
# #     assert nodes_obs["y_base_noise"]["value"].shape == (4, 1)

# #     assert nodes_obs["training_x1"]["value"].shape == (2, 4, 1, 1)
# #     assert (
# #         nodes_nobs["training_x1"]["value"].shape
# #         == nodes_obs["training_x1"]["value"].shape
# #     )


# # TODO add logits tests and logprob tests for the rsampled version

# # def test_conditional_categorical_shapes(categorical_model):
# #     c = categorical_model
# #     with (
# #         MultiWorldCounterfactual(first_available_dim=-3),
# #         do(actions={"training_x1": torch.ones(1, 1)}),
# #         pyro.poutine.trace() as tr_cat_nobs,
# #     ):
# #         with pyro.plate("setup_samples", size=6, dim=-2):
# #             training_x, training_y = setup_categorical_problem(c)
# #         c.sample(training_x)
# #         logits = c.forward(training_x)
# #         pyro.sample(
# #             "y_from_logits", dist.Categorical(logits=logits.unsqueeze(-2)).to_event(1)
# #         )

# # def test_conditional_normal_log_prob(normal_model):
# #     n = normal_model
# #     # we want log probs to be consistent with usual computation thereof
# #     with pyro.poutine.trace() as tr_lp:
# #         with pyro.plate("setup_samples", size=5000, dim=-2):
# #             training_x, training_y = setup_regression_problem(n)
# #         n.sample(training_x, training_y)
# #         loc, scale = n.forward(training_x)
# #         pyro.sample("y_resampled", dist.Normal(loc, scale).to_event(1), obs=training_y)

# #     tr_lp.trace.compute_log_prob()
# #     nodes_lp = tr_lp.trace.nodes

# #     raw_y_lp = nodes_lp["y"]["log_prob"]
# #     assert torch.allclose(raw_y_lp, torch.tensor(0.0))

# #     base_noise_lp = nodes_lp["y_base_noise"]["log_prob"]
# #     resampled_lp = nodes_lp["y_resampled"]["log_prob"]
# #     original_lp = nodes_lp["y_logp"]["log_prob"]

# #     assert torch.allclose(resampled_lp, original_lp + base_noise_lp)
