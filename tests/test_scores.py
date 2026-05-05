import torch

from pci.explanation.scores import abs_diff_score, delta_to_reference_group_score


def test_delta_to_reference_group_score():
    torch.manual_seed(0)

    factual = torch.tensor([0.1, 0.4, 0.6, 0.8]).view(4, 1, 1)
    suff = torch.tensor([0.2, 0.3, 0.8, 0.9]).view(4, 1, 1)
    nec = torch.tensor([0.4, 0.7, 0.1, 1.1]).view(4, 1, 1)

    abs_diff_result = abs_diff_score(factual, suff, nec)
    delta_result = delta_to_reference_group_score(
        factual_outcomes=factual, suff_outcomes=suff, nec_outcomes=nec
    )

    type_keys = {"total", "suff", "nec"}

    for key in type_keys:
        assert key in abs_diff_result
        assert key in delta_result

        assert abs_diff_result[key].shape == (4, 1, 1)
        assert delta_result[key].shape == (4, 1, 1)

    assert torch.allclose(
        abs_diff_result["suff"],
        torch.tensor([-0.1, -0.1, -0.2, -0.1]).view(4, 1, 1),
        atol=1e-6,
    )
    assert torch.allclose(
        abs_diff_result["nec"],
        torch.tensor([0.3, 0.3, 0.5, 0.3]).view(4, 1, 1),
        atol=1e-6,
    )
    assert torch.allclose(
        abs_diff_result["total"],
        torch.tensor([0.2, 0.2, 0.3, 0.2]).view(4, 1, 1),
        atol=1e-6,
    )

    assert torch.isclose(delta_result["suff"][0, 0, 0], torch.tensor(-0.1), atol=1e-4)
    assert torch.isclose(delta_result["nec"][0, 0, 0], torch.tensor(0.3), atol=1e-4)
    assert torch.isclose(delta_result["total"][0, 0, 0], torch.tensor(0.2), atol=1e-4)


def test_delta_to_reference_group_score_with_sampling_dim():
    torch.manual_seed(0)

    # Add a leading sampling dimension (S=2)
    factual = torch.tensor(
        [[0.1], [0.4], [0.6], [0.8]],
    ).unsqueeze(-1)  # shape (2, 4, 1, 1)

    suff = torch.tensor(
        [[[0.2], [0.3], [0.8], [0.9]], [[0.1], [0.2], [0.4], [0.6]]]
    ).unsqueeze(-1)

    nec = torch.tensor(
        [[[0.4], [0.7], [0.1], [1.1]], [[0.3], [0.0], [0.2], [0.4]]]
    ).unsqueeze(-1)

    abs_diff_result = abs_diff_score(factual, suff, nec)
    delta_result = delta_to_reference_group_score(
        factual_outcomes=factual, suff_outcomes=suff, nec_outcomes=nec
    )

    type_keys = {"total", "suff", "nec"}
    for key in type_keys:
        assert key in abs_diff_result
        assert key in delta_result

        assert abs_diff_result[key].shape == (2, 4, 1, 1)
        assert delta_result[key].shape == (2, 4, 1, 1)

    # Expected values for the first sample
    assert torch.isclose(
        delta_result["suff"][0, 0, 0, 0], torch.tensor(-0.1), atol=1e-4
    )
    assert torch.isclose(delta_result["nec"][0, 0], torch.tensor(0.3), atol=1e-4)
    assert torch.isclose(delta_result["total"][0, 0], torch.tensor(0.2), atol=1e-4)

    # Expected values for the second sample
    assert torch.isclose(delta_result["suff"][1, 0], torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(delta_result["nec"][1, 0], torch.tensor(0.2), atol=1e-4)
    assert torch.isclose(delta_result["total"][1, 0], torch.tensor(0.2), atol=1e-4)


def test_delta_to_reference_group_score_with_last_batch_dim():
    factual = torch.tensor([[[0.1, 0.4, 0.6, 0.8]]])

    suff = torch.tensor(
        [
            [[[0.2, 0.3, 0.8, 0.9]]],
            [[[0.1, 0.2, 0.4, 0.6]]],
        ]
    )

    nec = torch.tensor(
        [
            [[[0.4, 0.7, 0.1, 1.1]]],
            [[[0.3, 0.0, 0.2, 0.4]]],
        ]
    )

    abs_diff_result = abs_diff_score(factual, suff, nec)
    delta_result = delta_to_reference_group_score(
        factual_outcomes=factual, suff_outcomes=suff, nec_outcomes=nec
    )

    for key in {"total", "suff", "nec"}:
        assert key in abs_diff_result
        assert key in delta_result

        assert abs_diff_result[key].shape == (2, 1, 1, 4)
        assert delta_result[key].shape == (2, 1, 1, 4)

    assert torch.isclose(
        delta_result["suff"][0, 0, 0, 0], torch.tensor(-0.1), atol=1e-4
    )
    assert torch.isclose(delta_result["nec"][0, 0, 0, 0], torch.tensor(0.3), atol=1e-4)
    assert torch.isclose(
        delta_result["total"][0, 0, 0, 0], torch.tensor(0.2), atol=1e-4
    )
