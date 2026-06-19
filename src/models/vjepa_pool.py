"""Eval-time pooling of V-JEPA tokens to per-(token)frame features."""

from __future__ import annotations

from torch import Tensor


def frame_mean_pool(tokens: Tensor, grid: tuple[int, int, int]) -> Tensor:
    """(B, N, D) -> (B, gt, D) by averaging the gh*gw spatial tokens per frame.

    Assumes the token order is (gt, gh, gw) row-major (the TubeletEmbed flatten
    order), so frame f occupies the contiguous block ``[f*gh*gw : (f+1)*gh*gw]``.
    """
    b, n, d = tokens.shape
    gt, gh, gw = grid
    assert n == gt * gh * gw, f"{n} != {gt}*{gh}*{gw}"
    return tokens.view(b, gt, gh * gw, d).mean(dim=2)
