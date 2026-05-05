import torch

from pci.explanation.utils import broadcast_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def condition_on_interventional_regime(
    results_dictionary: dict,
    reference_variable_names: list[str],
    antecedent_regimes: dict[str, bool] | None = None,
    witness_regimes: dict[str, bool] | None = None,
):
    """
    Condition sufficiency and necessity tensors on specified interventional regimes.

    This function applies masks derived from antecedent and witness regimes to
    the sufficiency and necessity tensors stored in a results dictionary. It
    returns conditioned tensors along with the variable-specific indicator masks.

    :param results_dictionary: Dictionary containing:
        - "sufficiency": mapping of variable names to their sufficiency tensors.
        - "necessity": mapping of variable names to their necessity tensors.
        - "antecedent_indicators": boolean tensors indicating which entries satisfy the antecedent conditions.
        - "witness_indicators": boolean tensors indicating which entries satisfy the witness conditions.
    :param reference_variable_names: List of variable names to condition on. Each must exist in `results_dictionary["sufficiency"]`.
    :param antecedent_regimes: Optional dictionary specifying which reference variables should follow the antecedent regime (`True`) or its complement (`False`). Default is None.
    :param witness_regimes: Optional dictionary specifying which reference variables should follow the witness regime (`True`) or its complement (`False`). Default is None.

    :return: Dictionary containing:
        - "regime_sufficiency": sufficiency tensors after conditioning on the specified regimes.
        - "regime_necessity": necessity tensors after conditioning on the specified regimes.
        - "antecedent_indicators": variable-specific antecedent masks used in conditioning.
        - "witness_indicators": variable-specific witness masks used in conditioning.

    :raises AssertionError: If any reference variable name is not found in `results_dictionary["sufficiency"]`.
    :raises ValueError: If neither `antecedent_regimes` nor `witness_regimes` is provided, or if no indicators exist for a reference variable.

    :note:
        - The function automatically combines antecedent and witness indicators using logical AND if both are provided for a variable.
        - Conditioning is applied element-wise; entries not satisfying the combined mask are set to NaN.
        - If necessary, the mask is broadcasted to match the shape of sufficiency/necessity tensors.
    """
    for name in reference_variable_names:
        assert name in results_dictionary["sufficiency"], (
            f"Reference variable '{name}' not found in results dictionary."
        )

    if antecedent_regimes is None and witness_regimes is None:
        raise ValueError(
            "At least one of antecedent_regimes or witness_regimes must be provided."
        )

    # --- Build masks and store variable-specific indicators ---

    ant_indicators = {}
    wit_indicators = {}
    combined_indicators = {}

    conditioned_sufficiency = {}
    conditioned_necessity = {}

    for name in reference_variable_names:
        # Antecedent mask
        if antecedent_regimes is not None and name in antecedent_regimes.keys():
            ant_flag = antecedent_regimes[name]
            ant_ind = results_dictionary["antecedent_indicators"][name]
            if not ant_flag:
                ant_ind = ~ant_ind

            ant_indicators[name] = ant_ind
        else:
            ant_ind = None
            ant_indicators[name] = None

        # Witness mask
        if witness_regimes is not None and name in witness_regimes.keys():
            wit_flag = witness_regimes[name]
            wit_ind = results_dictionary["witness_indicators"][name]
            if not wit_flag:
                wit_ind = ~wit_ind
            wit_indicators[name] = wit_ind
        else:
            # If no witness indicator provided, use None or empty tensor
            wit_ind = None
            wit_indicators[name] = None

        # Combine masks if both present, else take whichever exists
        if ant_ind is not None and wit_ind is not None:
            combined_indicators[name] = torch.logical_and(ant_ind, wit_ind)
        elif ant_ind is not None:
            combined_indicators[name] = ant_ind
        elif wit_ind is not None:
            combined_indicators[name] = wit_ind
        else:
            raise ValueError(f"No indicators provided for reference variable '{name}'.")

    # combine mask across variables
    if combined_indicators and next(iter(combined_indicators)) is not None:
        tensors = list(combined_indicators.values())
        combined_mask = tensors[0]
        for t in tensors[1:]:
            combined_mask = combined_mask & t

    for name in results_dictionary["sufficiency"].keys():
        suff_tensor = results_dictionary["sufficiency"][name]
        nec_tensor = results_dictionary["necessity"][name]

        if combined_mask.shape == suff_tensor.shape:
            mask = combined_mask
        elif combined_mask.ndim == suff_tensor.ndim:
            mask = combined_mask.reshape(suff_tensor.shape)
        else:
            mask = broadcast_mask(combined_mask, suff_tensor)
        # mask = combined_mask.reshape(suff_tensor.shape)

        if mask is None:
            # No mask — keep all values unchanged
            conditioned_sufficiency[name] = suff_tensor.detach()
            conditioned_necessity[name] = nec_tensor.detach()
            continue

        else:
            conditioned_sufficiency[name] = torch.where(
                mask.to(suff_tensor.device), suff_tensor, torch.nan
            ).detach()

            conditioned_necessity[name] = torch.where(
                mask.to(nec_tensor.device), nec_tensor, torch.nan
            ).detach()

    return {
        "regime_sufficiency": conditioned_sufficiency,
        "regime_necessity": conditioned_necessity,
        "antecedent_indicators": ant_indicators,
        "witness_indicators": wit_indicators,
    }
