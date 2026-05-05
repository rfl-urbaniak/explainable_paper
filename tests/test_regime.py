import pytest
import torch

from pci.explanation.regime import condition_on_interventional_regime


def test_condition_on_interventional_regime_basic():
    results = {
        "sufficiency": {
            "var1": torch.tensor([[1.0], [2.0]]),
            "var2": torch.tensor([[3.0], [4.0]]),
        },
        "necessity": {
            "var1": torch.tensor([[10.0], [20.0]]),
            "var2": torch.tensor([[30.0], [40.0]]),
        },
        "antecedent_indicators": {
            "var1": torch.tensor([True, False]),
            "var2": torch.tensor([False, True]),
        },
        "witness_indicators": {
            "var1": torch.tensor([False, True]),
            "var2": torch.tensor([True, False]),
        },
    }

    # Case 1: No indicators provided → should raise ValueError

    with pytest.raises(
        ValueError,
        match="At least one of antecedent_regimes or witness_regimes must be provided.",
    ):
        condition_on_interventional_regime(results, ["var1"])

    # Case 2: Antecedent indicator filters rows
    out = condition_on_interventional_regime(
        results, reference_variable_names=["var1"], antecedent_regimes={"var1": True}
    )
    # Only first row should remain
    expected_suff = torch.tensor([[1.0], [float("nan")]])
    expected_nec = torch.tensor([[10.0], [float("nan")]])
    assert torch.equal(
        torch.isnan(expected_suff), torch.isnan(out["regime_sufficiency"]["var1"])
    ), "Mismatch in NaN positions"
    assert torch.equal(
        torch.isnan(expected_nec), torch.isnan(out["regime_necessity"]["var1"])
    ), "Mismatch in NaN positions"

    non_nan_mask = ~torch.isnan(expected_suff)
    assert torch.allclose(
        out["regime_sufficiency"]["var1"][non_nan_mask], expected_suff[non_nan_mask]
    ), "Non-NaN values differ"
    assert torch.allclose(
        out["regime_necessity"]["var1"][non_nan_mask], expected_nec[non_nan_mask]
    ), "Non-NaN values differ"
    assert "var1" in out["antecedent_indicators"]

    expected_suff_var2 = torch.tensor([[3.0], [float("nan")]])
    expected_nec_var2 = torch.tensor([[30.0], [float("nan")]])
    assert torch.equal(
        torch.isnan(expected_suff_var2), torch.isnan(out["regime_sufficiency"]["var2"])
    ), "Mismatch in NaN positions for var2"
    assert torch.equal(
        torch.isnan(expected_nec_var2), torch.isnan(out["regime_necessity"]["var2"])
    ), "Mismatch in NaN positions for var2"

    assert torch.allclose(
        out["regime_sufficiency"]["var2"][non_nan_mask],
        expected_suff_var2[non_nan_mask],
    ), "Non-NaN values differ for var2"
    assert torch.allclose(
        out["regime_necessity"]["var2"][non_nan_mask], expected_nec_var2[non_nan_mask]
    ), "Non-NaN values differ for var2"

    # # Case 3: Antecedent indicator inverted (False flag) → keeps rows where original mask is False
    out_false = condition_on_interventional_regime(
        results, reference_variable_names=["var1"], antecedent_regimes={"var1": False}
    )
    expected_suff_false_var1 = torch.tensor([[float("nan")], [2.0]])
    expected_nec_false_var1 = torch.tensor([[float("nan")], [20.0]])

    assert torch.equal(
        torch.isnan(out_false["regime_sufficiency"]["var1"]),
        torch.isnan(expected_suff_false_var1),
    ), "Mismatch in NaN positions for sufficiency var1 (False case)"
    assert torch.equal(
        torch.isnan(out_false["regime_necessity"]["var1"]),
        torch.isnan(expected_nec_false_var1),
    ), "Mismatch in NaN positions for necessity var1 (False case)"

    non_nan_mask_false = ~torch.isnan(expected_suff_false_var1)
    assert torch.allclose(
        out_false["regime_sufficiency"]["var1"][non_nan_mask_false],
        expected_suff_false_var1[non_nan_mask_false],
    ), "Non-NaN sufficiency values differ for var1 (False case)"
    assert torch.allclose(
        out_false["regime_necessity"]["var1"][non_nan_mask_false],
        expected_nec_false_var1[non_nan_mask_false],
    ), "Non-NaN necessity values differ for var1 (False case)"

    # Case 4: Witness indicator True → keep second row (witness_indicators['var1'] = [False, True])
    out_witness = condition_on_interventional_regime(
        results, reference_variable_names=["var1"], witness_regimes={"var1": True}
    )
    # Expected: only second row kept, first row NaN
    expected_suff_witness_var1 = torch.tensor([[float("nan")], [2.0]])
    expected_nec_witness_var1 = torch.tensor([[float("nan")], [20.0]])

    assert (
        out_witness["regime_sufficiency"]["var1"].shape
        == expected_suff_witness_var1.shape
    )
    assert (
        out_witness["regime_necessity"]["var1"].shape == expected_nec_witness_var1.shape
    )

    assert torch.equal(
        torch.isnan(out_witness["regime_sufficiency"]["var1"]),
        torch.isnan(expected_suff_witness_var1),
    ), "Mismatch in NaN positions for sufficiency var1 (witness case)"
    assert torch.equal(
        torch.isnan(out_witness["regime_necessity"]["var1"]),
        torch.isnan(expected_nec_witness_var1),
    ), "Mismatch in NaN positions for necessity var1 (witness case)"

    non_nan_mask_witness = ~torch.isnan(expected_suff_witness_var1)
    assert torch.allclose(
        out_witness["regime_sufficiency"]["var1"][non_nan_mask_witness],
        expected_suff_witness_var1[non_nan_mask_witness],
    ), "Non-NaN sufficiency values differ for var1 (witness case)"
    assert torch.allclose(
        out_witness["regime_necessity"]["var1"][non_nan_mask_witness],
        expected_nec_witness_var1[non_nan_mask_witness],
    ), "Non-NaN necessity values differ for var1 (witness case)"

    # Case 5: Witness True + Antecedent True → keep rows where both masks are True
    out_both = condition_on_interventional_regime(
        results,
        reference_variable_names=["var1"],
        antecedent_regimes={"var1": True},
        witness_regimes={"var1": True},
    )
    # antecedent_indicators['var1'] = [True, False]
    # witness_indicators['var1'] = [False, True]
    # Logical AND = [False, False] → no rows kept, all NaN expected
    expected_suff_both_var1 = torch.tensor([[float("nan")], [float("nan")]])
    expected_nec_both_var1 = torch.tensor([[float("nan")], [float("nan")]])

    assert out_both["regime_sufficiency"]["var1"].shape == expected_suff_both_var1.shape
    assert out_both["regime_necessity"]["var1"].shape == expected_nec_both_var1.shape

    assert torch.equal(
        torch.isnan(out_both["regime_sufficiency"]["var1"]),
        torch.isnan(expected_suff_both_var1),
    ), "Mismatch in NaN positions for sufficiency var1 (both masks case)"
    assert torch.equal(
        torch.isnan(out_both["regime_necessity"]["var1"]),
        torch.isnan(expected_nec_both_var1),
    ), "Mismatch in NaN positions for necessity var1 (both masks case)"

    non_nan_mask_both = ~torch.isnan(expected_suff_both_var1)
    assert out_both["regime_sufficiency"]["var1"][non_nan_mask_both].numel() == 0, (
        "Expected no non-NaN sufficiency values for var1 (both masks case)"
    )
    assert out_both["regime_necessity"]["var1"][non_nan_mask_both].numel() == 0, (
        "Expected no non-NaN necessity values for var1 (both masks case)"
    )

    # Case 6: Multiple vars, mixed masks
    out_partial = condition_on_interventional_regime(
        results,
        reference_variable_names=["var1", "var2"],
        antecedent_regimes={
            "var1": True,
            "var2": False,
        },
        witness_regimes={
            "var1": False,
            "var2": True,
        },
    )

    expected_suff_var1 = torch.tensor([[1.0], [float("nan")]])
    expected_nec_var1 = torch.tensor([[10.0], [float("nan")]])
    expected_suff_var2 = torch.tensor([[3.0], [float("nan")]])
    expected_nec_var2 = torch.tensor([[30.0], [float("nan")]])

    assert out_partial["regime_sufficiency"]["var1"].shape == expected_suff_var1.shape
    assert out_partial["regime_necessity"]["var1"].shape == expected_nec_var1.shape
    assert out_partial["regime_sufficiency"]["var2"].shape == expected_suff_var2.shape
    assert out_partial["regime_necessity"]["var2"].shape == expected_nec_var2.shape

    assert torch.equal(
        torch.isnan(out_partial["regime_sufficiency"]["var1"]),
        torch.isnan(expected_suff_var1),
    ), "Mismatch NaNs sufficiency var1"
    assert torch.equal(
        torch.isnan(out_partial["regime_necessity"]["var1"]),
        torch.isnan(expected_nec_var1),
    ), "Mismatch NaNs necessity var1"
    assert torch.equal(
        torch.isnan(out_partial["regime_sufficiency"]["var2"]),
        torch.isnan(expected_suff_var2),
    ), "Mismatch NaNs sufficiency var2"
    assert torch.equal(
        torch.isnan(out_partial["regime_necessity"]["var2"]),
        torch.isnan(expected_nec_var2),
    ), "Mismatch NaNs necessity var2"

    non_nan_mask = ~torch.isnan(expected_suff_var1)
    assert torch.allclose(
        out_partial["regime_sufficiency"]["var1"][non_nan_mask],
        expected_suff_var1[non_nan_mask],
    ), "Non-NaN sufficiency var1 values differ"
    assert torch.allclose(
        out_partial["regime_necessity"]["var1"][non_nan_mask],
        expected_nec_var1[non_nan_mask],
    ), "Non-NaN necessity var1 values differ"
    assert torch.allclose(
        out_partial["regime_sufficiency"]["var2"][non_nan_mask],
        expected_suff_var2[non_nan_mask],
    ), "Non-NaN sufficiency var2 values differ"
    assert torch.allclose(
        out_partial["regime_necessity"]["var2"][non_nan_mask],
        expected_nec_var2[non_nan_mask],
    ), "Non-NaN necessity var2 values differ"
