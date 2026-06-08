import torch

from pci.explanation.utils import broadcast_mask
from pci.tools.data_processing import inverse_transform_log_epsilon


def abs_diff_score(
    factual_outcomes: torch.Tensor,
    suff_outcomes: torch.Tensor,
    nec_outcomes: torch.Tensor,
) -> dict:
    """
    Compute sufficiency, necessity, and total scores based on absolute differences from a factual outcome.

    For each entry:
        - Sufficiency: negative absolute difference (smaller difference → higher score)
        - Necessity: positive absolute difference
        - Total: even-weighted sum of sufficiency and necessity scores

    :param factual_outcome: reference factual outcome value
    :param suff_outcomes: model outputs under sufficiency conditions
    :param nec_outcomes: model outputs under necessity conditions
    :return: dictionary containing
        - 'sufficiency': tensor of sufficiency scores
        - 'necessity': tensor of necessity scores
        - 'total': tensor of combined scores (even-weighted sum)
    """

    # Handle scalar tensors (0-dimensional) - they are trivially compatible
    if (
        factual_outcomes.dim() > 0
        and suff_outcomes.dim() > 0
        and nec_outcomes.dim() > 0
    ):
        assert (
            suff_outcomes.shape[-1]
            == nec_outcomes.shape[-1]
            == factual_outcomes.shape[-1]
        ), "All outcome tensors must have the same last dimension size."

    scores_suff = torch.abs(suff_outcomes - factual_outcomes) * -1
    scores_nec = torch.abs(nec_outcomes - factual_outcomes)
    score_total = scores_suff + scores_nec

    return {"suff": scores_suff, "nec": scores_nec, "total": score_total}


def abs_diff_score_inverse_transformed(
    factual_outcomes: torch.Tensor,
    suff_outcomes: torch.Tensor,
    nec_outcomes: torch.Tensor,
    params: dict,
    to_percentile: bool = False,
) -> dict:
    """
    Computes the absolute difference score between factual, sufficiency, and necessity
    outcomes after applying an inverse transformation.

    This function first applies ``inverse_transform_log_epsilon`` to each of the input
    tensors (``factual_outcomes``, ``suff_outcomes``, ``nec_outcomes``) using the provided
    parameters, then computes the absolute difference score using the transformed values.

    :param factual_outcomes: factual outcomes to be compared.
    :param suff_outcomes: sufficiency outcomes to be compared.
    :param nec_outcomes: necessity outcomes to be compared.
    :param params: parameters needed for ``inverse_transform_log_epsilon``.
    :param to_percentile: if true, the output score may be converted to a percentile scale
        (currently not used in this implementation). Defaults to false.
    :returns: the absolute difference scores computed by ``abs_diff_score`` from the
        inverse-transformed outcomes; keys and values are as returned by ``abs_diff_score``.

    .. note::
       The actual score computation is delegated to ``abs_diff_score``.
    """

    inv_factual = inverse_transform_log_epsilon(factual_outcomes, params)
    inv_suff = inverse_transform_log_epsilon(suff_outcomes, params)
    inv_nec = inverse_transform_log_epsilon(nec_outcomes, params)

    return abs_diff_score(
        factual_outcomes=inv_factual,
        suff_outcomes=inv_suff,
        nec_outcomes=inv_nec,
    )


def leave_one_out_nanmeans(values, batch_dim=-2):
    """Compute leave-one-out means along the batch dimension for each entry.

    For each row `r` in the batch:
        - exclude the r-th entry
        - compute the nanmean of the remaining entries (we expect nans due to masking in conditioning)
        - store the result in the corresponding position

    :param values: tensor containing the values to compute leave-one-out means for
    :return: tensor of stacked leave-one-out means
    """

    B = values.shape[batch_dim]
    result = []
    for i in range(B):
        _dist = values.select(-2, i)
        mask = torch.ones(B, dtype=torch.bool, device=values.device)
        mask[i] = False
        mask = broadcast_mask(mask, _dist)
        masked_dist = _dist.clone()
        masked_dist[~mask] = float("nan")
        _nanmean = torch.nanmean(masked_dist, dim=-1)
        result.append(_nanmean)

    if len(result[0].shape) == 0:
        return torch.stack(result)

    return torch.stack(result, dim=1)


def delta_to_reference_group_score(
    factual_outcomes: torch.Tensor,
    suff_outcomes: torch.Tensor,
    nec_outcomes: torch.Tensor,
    params: dict | None = None,
    batch_dim: int = -2,
) -> dict:
    """Compute causal impact score based on absolute difference from other factual outcomes as provided in the inputs.
    If transformation parameters are provided, the outcomes are first inverse transformed using `inverse_transform_log_epsilon`.

    For each row `r`:
        - compute the mean distance from other factual rows in the factual world (leave-one-out)
        - compute the mean distance of the sufficiency outcome for row r from other rows in the *factual* world
        - compute the mean distance of the necessity outcome for row r from other rows in the *factual* world
        - pass those intermediate values to `abs_diff_score` to compute the final scores.

    :param factual_outcomes: tensor of factual outcomes, shape (..., B, ...)
    :param suff_outcomes: tensor of sufficiency outcomes, shape (..., B, ...)
    :param nec_outcomes: tensor of necessity outcomes, shape (..., B, ...)
    :param params: transformation params, defaults to None, in which case no inverse transformation is applied
    :return: point-wise causal impact scores, just as other scores.
    """

    if params is not None:
        inv_factual = inverse_transform_log_epsilon(factual_outcomes, params)
        inv_suff = inverse_transform_log_epsilon(suff_outcomes, params)
        inv_nec = inverse_transform_log_epsilon(nec_outcomes, params)

    else:
        inv_factual = factual_outcomes
        inv_suff = suff_outcomes
        inv_nec = nec_outcomes

    inv_factual = inv_factual.squeeze().unsqueeze(-1)
    inv_suff = inv_suff.squeeze().unsqueeze(-1)
    inv_nec = inv_nec.squeeze().unsqueeze(-1)

    factual_dist = torch.cdist(inv_factual, inv_factual, p=1)
    nec_dist = torch.cdist(inv_nec, inv_factual, p=1)
    suff_dist = torch.cdist(inv_suff, inv_factual, p=1)

    # in distance matrices, the batch dimension
    factual_means = leave_one_out_nanmeans(factual_dist, batch_dim=-2).view_as(
        factual_outcomes
    )
    suff_means = leave_one_out_nanmeans(suff_dist, batch_dim=-2).view_as(suff_outcomes)
    nec_means = leave_one_out_nanmeans(nec_dist, batch_dim=-2).view_as(nec_outcomes)

    scores = abs_diff_score(
        factual_outcomes=factual_means,
        suff_outcomes=suff_means,
        nec_outcomes=nec_means,
    )

    return scores
