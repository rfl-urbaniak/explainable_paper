import pyro
import pyro.distributions as dist
import pytest
import scipy.stats
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.handlers import do
from chirho.observational.handlers.condition import condition
from pyro.distributions.transforms import AffineTransform

from pci.tools.rsample import (
    Exogenate,
    RSampleNormalSites,
    make_analytic_log_prob_and_inv,
    rsample,
)


@pytest.fixture(scope="function", autouse=True)
def set_rng_seed():
    """Set random seed before each test."""
    pyro.set_rng_seed(12)
    yield
    # Cleanup if needed


def get_var(name, tr):
    """Helper function to extract variable value from trace."""
    return tr.trace.nodes[name]["value"]


def get_lp(name, tr):
    """Helper function to extract log probability from trace."""
    return tr.trace.nodes[name]["log_prob"]


def stdf(x):
    """Helper function for scale transformation."""
    return torch.tensor(1.0) / (1.0 + torch.exp(torch.tensor(-0.3) * x)) + 0.1


def model(rsample_mode="auto_rsample", event_shape=None):
    """Test model that uses both rsample (y) and regular sample (yp) for comparison.

    Args:
        rsample_mode: One of "auto_rsample", "analytic_rsample", or "handler_rsample"
            - "auto_rsample": Uses rsample directly (default)
            - "analytic_rsample": Uses rsample with analytic function
            - "handler_rsample": Uses regular pyro.sample for y, which will be converted
              to rsample by RSampleNormalSites handler
        event_shape: Event shape tuple for x, y, and yp distributions (e.g., () for scalar, (3,) for event dim of size 3)
    """
    if event_shape is None:
        raise ValueError("event_shape must be provided")
    # Define x distribution
    x_dist = dist.Normal(10.0, 1.0).expand(event_shape).to_event(len(event_shape))
    x = pyro.sample("x", x_dist)

    scale = stdf(x)

    if rsample_mode == "analytic_rsample":
        analytic_fn = make_analytic_log_prob_and_inv(x, scale, event_shape)
    else:
        analytic_fn = None

    if rsample_mode == "handler_rsample":
        # Use regular sample - will be converted by RSampleNormalSites handler
        y_dist = dist.Normal(x, scale).to_event(len(event_shape))
        y = pyro.sample("y", y_dist)
    else:
        # Use rsample directly
        base_dist = dist.Normal(0.0, 1.0).expand(event_shape).to_event(len(event_shape))
        y = rsample(
            "y",
            base_dist=base_dist,
            transforms=[AffineTransform(loc=x, scale=scale)],
            analytic_log_prob_and_inv=analytic_fn,
        )

    # Define yp distribution
    yp_dist = dist.Normal(x, scale).to_event(len(event_shape))
    pyro.sample("yp", yp_dist)
    return y, analytic_fn


def model_with_verification(rsample_mode="auto_rsample", event_shape=None):
    """Wrapper around model that verifies analytic function usage and handles context wrapping."""
    # Wrap model execution with RSampleNormalSites handler if needed
    if rsample_mode == "handler_rsample":
        with RSampleNormalSites("y"):
            y, analytic_fn = model(rsample_mode=rsample_mode, event_shape=event_shape)
    else:
        y, analytic_fn = model(rsample_mode=rsample_mode, event_shape=event_shape)

    if rsample_mode == "analytic_rsample":
        assert analytic_fn is not None, (
            "Analytic function should be created when rsample_mode='analytic_rsample'"
        )
        assert analytic_fn.call_count[0] > 0, (
            "Analytic function should be called when rsample_mode='analytic_rsample'"
        )
    else:
        assert analytic_fn is None, (
            "Analytic function should not be created when rsample_mode != 'analytic_rsample'"
        )

    return y


def _baseline_trace(rsample_mode="auto_rsample", event_shape=None):
    """Create baseline trace for comparison tests."""
    with pyro.plate("p0", size=50000):
        with pyro.poutine.trace() as tr0:
            with Exogenate():
                model_with_verification(
                    rsample_mode=rsample_mode, event_shape=event_shape
                )

    tr0.trace.compute_log_prob()

    return tr0


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_1_distributional_equivalence(rsample_mode, event_shape):
    """
    Test 1: Verify that rsample produces distributionally equivalent samples
    to regular sample.
    """
    baseline_trace = _baseline_trace(rsample_mode=rsample_mode, event_shape=event_shape)
    kstatistic = scipy.stats.ks_2samp(
        get_var("y", baseline_trace).flatten(),
        get_var("yp", baseline_trace).flatten(),
    )
    assert kstatistic[0] < 0.1


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_2_log_probability_consistency(rsample_mode, event_shape):
    """
    Test 2: Verify that unconditioned log likelihoods are consistent between
    regular sample and rsample (where log prob contributions are separated
    between y and the base noise).
    """
    baseline_trace = _baseline_trace(rsample_mode=rsample_mode, event_shape=event_shape)
    logprob_kstatistic = scipy.stats.ks_2samp(
        get_lp("yp", baseline_trace),
        # Here the log prob contributions are separated between y and the base noise
        get_lp("y", baseline_trace) + get_lp("y_u", baseline_trace),
    )
    assert logprob_kstatistic[0] < 0.1


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_3_counterfactual_intervention(rsample_mode, event_shape):
    """
    Test 3: Verify that when intervening on x, potential outcomes for y differ,
    but use the same base noise sample.
    """
    # Compute intervention value based on event_shape
    x_intervention = torch.full(event_shape, -10.0)
    with Exogenate():
        with MultiWorldCounterfactual(first_available_dim=-3) as mwf:
            with do(actions=dict(x=x_intervention)):
                with pyro.poutine.trace() as tr1:
                    model_with_verification(
                        rsample_mode=rsample_mode, event_shape=event_shape
                    )

    with mwf:
        event_dim = len(event_shape)
        assert indices_of(get_var("y_u", tr1), event_dim=event_dim) == IndexSet()
        assert (
            indices_of(get_var("x", tr1), event_dim=event_dim)
            == indices_of(get_var("y", tr1), event_dim=event_dim)
            == IndexSet(x={0, 1})
        )

    event_size = 1 if not event_shape else torch.Size(event_shape).numel()
    assert get_var("y_u", tr1).numel() == event_size

    # Batch dimensions are always (2, 1, 1): counterfactual worlds, some dimension, separator
    # event_shape is appended to batch_shape (empty tuple for scalar case)
    expected_shape = (2, 1, 1) + event_shape
    assert get_var("x", tr1).shape == expected_shape
    assert get_var("y", tr1).shape == expected_shape


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_4_plating_independence(rsample_mode, event_shape):
    """
    Test 4: Verify that when plating the model, samples along that dimension
    are independent and use different base noise.
    """
    # Compute intervention value based on event_shape
    x_intervention = torch.full(event_shape, -10.0)
    with pyro.plate("p2", size=4, dim=-2):
        with Exogenate():
            with MultiWorldCounterfactual(first_available_dim=-3) as mwf:
                with do(actions=dict(x=x_intervention)):
                    with pyro.poutine.trace() as tr2:
                        model_with_verification(
                            rsample_mode=rsample_mode, event_shape=event_shape
                        )

    with mwf:
        event_dim = len(event_shape)
        assert indices_of(get_var("y_u", tr2), event_dim=event_dim) == IndexSet()
        assert (
            indices_of(get_var("x", tr2), event_dim=event_dim)
            == indices_of(get_var("y", tr2), event_dim=event_dim)
            == IndexSet(x={0, 1})
        )

    event_size = 1 if not event_shape else torch.Size(event_shape).numel()
    assert get_var("y_u", tr2).numel() == 4 * event_size
    # Assert that y_u values are different along the plate dimension
    y_u_values = get_var("y_u", tr2)
    assert torch.unique(y_u_values).numel() == 4 * event_size, (
        "y_u values should be different along the plate dimension, "
        "indicating independent base noise samples"
    )

    # Batch dimensions are always (2, 4, 1): counterfactual worlds, plate, separator
    # event_shape is appended to batch_shape (empty tuple for scalar case)
    expected_shape = (2, 4, 1) + event_shape
    assert get_var("x", tr2).shape == expected_shape
    assert get_var("y", tr2).shape == expected_shape


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_5_conditioning_with_explicit_noise(rsample_mode, event_shape):
    """
    Test 5: Verify that conditioning preserves posterior log likelihoods in
    factual world from equivalent, regular sample, when conditioning on y_u explicitly.
    """
    baseline_trace = _baseline_trace(rsample_mode=rsample_mode, event_shape=event_shape)
    with Exogenate():
        with condition(
            data=dict(
                y=get_var("y", baseline_trace),
                yp=get_var("y", baseline_trace),  # Condition yp on same values as y
                y_u=get_var("y_u", baseline_trace),
                x=get_var("x", baseline_trace),
            )
        ):
            with pyro.poutine.trace() as trobs1:
                with pyro.plate("p5", size=50000):
                    model_with_verification(
                        rsample_mode=rsample_mode, event_shape=event_shape
                    )

    trobs1.trace.compute_log_prob()

    yp_lp = get_lp("yp", trobs1)
    y_lp = get_lp("y", trobs1)
    y_u_lp = get_lp("y_u", trobs1)
    assert torch.allclose(
        yp_lp,
        y_lp + y_u_lp,
    ), (
        f"Log prob values don't match: "
        f"yp_lp.shape={yp_lp.shape}, y_lp.shape={y_lp.shape}, y_u_lp.shape={y_u_lp.shape}, "
        f"max_diff={(yp_lp - (y_lp + y_u_lp)).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_6_conditioning_without_explicit_noise(rsample_mode, event_shape):
    """
    Test 6: Verify that conditioning inverts to the exogenous noise and results
    in the right log probs without conditioning on y_u explicitly.
    """
    baseline_trace = _baseline_trace(rsample_mode=rsample_mode, event_shape=event_shape)
    with Exogenate():
        with condition(
            data=dict(
                y=get_var("y", baseline_trace),
                yp=get_var("y", baseline_trace),  # Condition yp on same values as y
                x=get_var("x", baseline_trace),
            )
        ):
            with pyro.poutine.trace() as trobs2:
                with pyro.plate("p5", size=50000):
                    model_with_verification(
                        rsample_mode=rsample_mode, event_shape=event_shape
                    )

    trobs2.trace.compute_log_prob()

    yp_lp = get_lp("yp", trobs2)
    y_lp = get_lp("y", trobs2)
    y_u_lp = get_lp("y_u", trobs2)
    assert torch.allclose(
        yp_lp,
        y_lp + y_u_lp,
    ), (
        f"Log prob values don't match: "
        f"yp_lp.shape={yp_lp.shape}, y_lp.shape={y_lp.shape}, y_u_lp.shape={y_u_lp.shape}, "
        f"max_diff={(yp_lp - (y_lp + y_u_lp)).abs().max().item()}"
    )

    y_u_trobs2 = get_var("y_u", trobs2)
    y_u_baseline = get_var("y_u", baseline_trace)
    assert torch.allclose(y_u_trobs2, y_u_baseline, atol=1e-6)


@pytest.mark.parametrize(
    "rsample_mode", ["auto_rsample", "analytic_rsample", "handler_rsample"]
)
@pytest.mark.parametrize("event_shape", [(), (3,), (3, 4)])
def test_7_conditioning_with_counterfactuals(rsample_mode, event_shape):
    """
    Test 7: Verify that conditioning inverts to exogenous noise and then
    propagates that noise to counterfactuals.
    """
    baseline_trace = _baseline_trace(rsample_mode=rsample_mode, event_shape=event_shape)
    # Compute intervention value based on event_shape
    x_intervention = torch.full(event_shape, -10.0)
    with Exogenate():
        with MultiWorldCounterfactual(first_available_dim=-3) as mwf:
            with condition(
                data=dict(
                    y=get_var("y", baseline_trace),
                    yp=get_var("y", baseline_trace),  # Condition yp on same values as y
                    x=get_var("x", baseline_trace),
                )
            ):
                with do(actions=dict(x=x_intervention)):
                    with pyro.poutine.trace() as trobs3:
                        with pyro.plate("p7", size=50000):
                            model_with_verification(
                                rsample_mode=rsample_mode, event_shape=event_shape
                            )

    y_u_trobs3 = get_var("y_u", trobs3)
    y_u_baseline = get_var("y_u", baseline_trace)
    assert torch.allclose(y_u_trobs3, y_u_baseline, atol=1e-6)

    with mwf:
        event_dim = len(event_shape)
        f_x = gather(get_var("x", trobs3), IndexSet(x={0}), event_dim=event_dim)
        # Random, centered at 10.
        assert torch.isclose(f_x.mean(), torch.tensor(10.0), atol=0.5)
        assert f_x.std() > torch.tensor(0.0)

        # Not random (intervened) at -10.
        cf_x = gather(get_var("x", trobs3), IndexSet(x={1}), event_dim=event_dim)
        if event_shape:
            assert torch.allclose(cf_x, torch.full(event_shape, -10.0))
        else:
            assert torch.allclose(cf_x, torch.tensor(-10.0))
        assert torch.isclose(cf_x.std(), torch.tensor(0.0))

        # Random, centered at f_x, which 10.
        f_y = gather(get_var("y", trobs3), IndexSet(x={0}), event_dim=event_dim)
        assert torch.isclose(f_y.mean(), f_x.mean(), atol=0.5)
        assert f_y.std() > torch.tensor(0.0)

        # Random, centered at cf_x, which is -10.
        cf_y = gather(get_var("y", trobs3), IndexSet(x={1}), event_dim=event_dim)
        assert torch.isclose(cf_y.mean(), cf_x.mean(), atol=0.5)
        assert cf_y.std() > torch.tensor(0.0)

        # Under intervention, y should equal the affine transform applied to the inferred noise.
        y_u = get_var("y_u", trobs3)
        cf_scale = stdf(cf_x)
        cf_y_expected = cf_scale * y_u + cf_x
        assert torch.allclose(
            cf_y,
            cf_y_expected,
            atol=1e-5,
            rtol=1e-5,
        ), "Counterfactual y must match affine transform of the shared exogenous noise"

        trobs3.trace.compute_log_prob()

        # Require that the counterfactual log probs are zero
        assert torch.allclose(
            gather(get_lp("y", trobs3), IndexSet(x={1})), torch.tensor(0.0)
        )

        yp_factual_lp = gather(get_lp("yp_factual", trobs3), IndexSet(x={0}))
        y_lp_gathered = gather(get_lp("y", trobs3), IndexSet(x={0}))
        y_u_lp_gathered = gather(get_lp("y_u", trobs3), IndexSet(x={0}))
        assert torch.allclose(yp_factual_lp, y_lp_gathered + y_u_lp_gathered), (
            f"Log prob values don't match: "
            f"yp_factual_lp.shape={yp_factual_lp.shape}, y_lp_gathered.shape={y_lp_gathered.shape}, "
            f"y_u_lp_gathered.shape={y_u_lp_gathered.shape}, "
            f"max_diff={(yp_factual_lp - (y_lp_gathered + y_u_lp_gathered)).abs().max().item()}"
        )


@pytest.mark.skip(
    # To see this, breakpoint the final assertion of test_7_conditioning_with_counterfactuals
    #  and inspect trobs3.
    "Exogenattion currently leaves hanging y_fatual and y_counterfactual"
    "sites with -inf log likelihood and non-constant, non-zero likelihood respectively."
)
def test_ambiguous_conditioning():
    """
    This test needs to ensure the following issue is resolved:
    ```
    # Ambiguous conditioning zeros log probs at the original site, and adds
    #  two sites with log probs as if the conditioning were applied to the
    #  factual xor counterfactual worlds only.
    get_lp("yp", trobs3)
    tensor([[[0., 0., 0.,  ..., 0., 0., 0.]],

            [[0., 0., 0.,  ..., 0., 0., 0.]]])
    get_lp("yp_factual", trobs3)
    tensor([[[-2.4314, -1.7922, -1.0594,  ..., -1.0343, -3.0283, -1.3167]],

            [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])
    get_lp("yp_counterfactual", trobs3)
    tensor([[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],

            [[0.9519, 0.3107, 0.7129,  ..., 0.5355, 0.7287, 0.2241]]])

    # Exogenate does not currently match this behavior.
    get_lp("y", trobs3)
    tensor([[[-0.0495, -0.0496, -0.0555,  ..., -0.0639, -0.0258, -0.0431]],

            [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])
    get_lp("y_factual", trobs3)
    tensor([[[-inf, -inf, -inf,  ..., -inf, -inf, -inf]],

            [[0., 0., 0.,  ..., 0., 0., 0.]]])
    get_lp("y_counterfactual", trobs3)
    tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

            [[-0.0495, -0.0496, -0.0555,  ..., -0.0639, -0.0258, -0.0431]]])python
    ```

    # The branch az-ru-exog-factualconditioning starts a refactor that relies on the observe effect
    #  directly. This is probably the right way to go.
    """
    raise NotImplementedError()


def test_base_dist_with_counterfactual_indices_raises():
    """Test that rsample raises an error if base_dist has counterfactual indices."""

    def bad_model():
        x = pyro.sample("x", dist.Normal(10.0, 1.0))
        scale = stdf(x)

        # This should raise an error because x has counterfactual dimensions
        # when we intervene on it, and we're using it as a parameter to base_dist
        y = rsample(
            "y",
            base_dist=dist.Normal(
                x, 1.0
            ),  # x will have counterfactual dims after intervention
            transforms=[AffineTransform(loc=0.0, scale=scale)],
        )
        return y

    with Exogenate():
        with MultiWorldCounterfactual(first_available_dim=-3):
            with do(actions=dict(x=-10.0)):
                with pytest.raises(
                    ValueError, match="Base distribution has counterfactual indices"
                ):
                    with pyro.poutine.trace():
                        bad_model()


@pytest.mark.parametrize("event_shape", [(), (3,)])
def test_8_two_rsample_sites_upstream_intervention(event_shape):
    """
    Model: x -> y -> z, where y and z are both rsampled.
    Intervene on x and verify both downstream counterfactual splits are correct.
    """

    def two_site_model():
        x = pyro.sample(
            "x", dist.Normal(10.0, 1.0).expand(event_shape).to_event(len(event_shape))
        )
        y = pyro.sample("y", dist.Normal(x, stdf(x)).to_event(len(event_shape)))
        z = pyro.sample("z", dist.Normal(y, stdf(y)).to_event(len(event_shape)))
        return x, y, z

    x_intervention = torch.full(event_shape, -10.0)

    with Exogenate():
        with MultiWorldCounterfactual(first_available_dim=-3) as mwf:
            with do(actions=dict(x=x_intervention)):
                with pyro.poutine.trace() as tr:
                    with RSampleNormalSites("y", "z"):
                        two_site_model()

    event_dim = len(event_shape)
    with mwf:
        # Base noise is shared across worlds for both sites
        assert indices_of(get_var("y_u", tr), event_dim=event_dim) == IndexSet()
        assert indices_of(get_var("z_u", tr), event_dim=event_dim) == IndexSet()

        # x, y, z all split across counterfactual worlds
        assert indices_of(get_var("x", tr), event_dim=event_dim) == IndexSet(x={0, 1})
        assert indices_of(get_var("y", tr), event_dim=event_dim) == IndexSet(x={0, 1})
        assert indices_of(get_var("z", tr), event_dim=event_dim) == IndexSet(x={0, 1})

        y_u = get_var("y_u", tr)
        z_u = get_var("z_u", tr)

        f_x = gather(get_var("x", tr), IndexSet(x={0}), event_dim=event_dim)
        cf_x = gather(get_var("x", tr), IndexSet(x={1}), event_dim=event_dim)
        f_y = gather(get_var("y", tr), IndexSet(x={0}), event_dim=event_dim)
        cf_y = gather(get_var("y", tr), IndexSet(x={1}), event_dim=event_dim)
        f_z = gather(get_var("z", tr), IndexSet(x={0}), event_dim=event_dim)
        cf_z = gather(get_var("z", tr), IndexSet(x={1}), event_dim=event_dim)

        # Counterfactual x is the intervention
        assert torch.allclose(cf_x.squeeze(), x_intervention)

        # y = stdf(x) * y_u + x in both worlds
        assert torch.allclose(f_y, stdf(f_x) * y_u + f_x, atol=1e-5)
        assert torch.allclose(cf_y, stdf(cf_x) * y_u + cf_x, atol=1e-5)

        # z = stdf(y) * z_u + y in both worlds
        assert torch.allclose(f_z, stdf(f_y) * z_u + f_y, atol=1e-5)
        assert torch.allclose(cf_z, stdf(cf_y) * z_u + cf_y, atol=1e-5)

        # Factual and counterfactual should differ
        assert not torch.allclose(f_y, cf_y)
        assert not torch.allclose(f_z, cf_z)


# TODO tests with weird event dims that differ across transforms and on base dist.
#  i.e. have transforms expand and remove event dims.


if __name__ == "__main__":
    # Debug failing test_7 case with event_shape=(3,)
    test_7_conditioning_with_counterfactuals("auto_rsample", (3,))
