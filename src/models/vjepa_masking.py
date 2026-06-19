"""Multi-block 3D token masking for V-JEPA (arXiv:2404.08471, Sec. 3.1).

Per sample, union several space-time blocks on the (gt,gh,gw) token grid, then
adjust to a fixed masked count so the batch's visible tokens are gatherable into
a dense (B, N_vis, D) tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MultiBlockMask:
    def __init__(
        self,
        grid: tuple[int, int, int] = (16, 12, 6),
        mask_ratio: float = 0.8,
        n_long: int = 2,
        n_short: int = 4,
    ) -> None:
        self.gt, self.gh, self.gw = grid
        self.n = self.gt * self.gh * self.gw
        self.n_masked = max(1, min(self.n - 1, int(round(mask_ratio * self.n))))
        self.n_long, self.n_short = n_long, n_short

    def _one(self) -> Tensor:
        g = torch.zeros(self.gt, self.gh, self.gw, dtype=torch.bool)
        # long-range: full temporal span, large spatial block
        for _ in range(self.n_long):
            bh = max(1, int(self.gh * 0.5))
            bw = max(1, int(self.gw * 0.5))
            h0 = int(torch.randint(0, self.gh - bh + 1, (1,)))
            w0 = int(torch.randint(0, self.gw - bw + 1, (1,)))
            g[:, h0 : h0 + bh, w0 : w0 + bw] = True  # noqa: E203
        # short-range: few frames, smaller spatial block
        for _ in range(self.n_short):
            bt = max(1, int(self.gt * 0.2))
            bh = max(1, int(self.gh * 0.25))
            bw = max(1, int(self.gw * 0.25))
            t0 = int(torch.randint(0, self.gt - bt + 1, (1,)))
            h0 = int(torch.randint(0, self.gh - bh + 1, (1,)))
            w0 = int(torch.randint(0, self.gw - bw + 1, (1,)))
            g[t0 : t0 + bt, h0 : h0 + bh, w0 : w0 + bw] = True  # noqa: E203
        flat = g.reshape(-1)
        # adjust to exactly n_masked: add/remove random positions
        idx = torch.randperm(self.n)
        cur = int(flat.sum())
        if cur < self.n_masked:
            add = idx[~flat[idx]][: self.n_masked - cur]
            flat[add] = True
        elif cur > self.n_masked:
            rem = idx[flat[idx]][: cur - self.n_masked]
            flat[rem] = False
        return flat

    def sample(self, batch_size: int) -> Tensor:
        return torch.stack([self._one() for _ in range(batch_size)], dim=0)
