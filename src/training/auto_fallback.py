"""SIGReg -> VICReg auto-fallback state machine.

Reference: CLAUDE.md "Risk-management" and HANDOFF.md D5.

The training loop holds one ``AutoFallbackController`` instance. Every
``diagnostic_every`` iterations it computes ``PR(z)`` and the linear
probe ``R^2`` for ``c`` on a held-out Test B sub-batch, then calls
``controller.step(iter, pr, probe_r2)``. When all three conditions

- ``iter >= 20_000``
- ``PR(z) < 0.3 * d``
- ``probe_R^2(c | z) < 0.7``

are met simultaneously, ``step`` returns ``True`` once. The training loop
responds by calling ``jepa.set_anticollapse(VICReg(...))`` and logging the
event prominently to W&B and stdout. After firing the controller is
idempotent: subsequent calls return ``False`` even if the conditions
remain true.
"""

from __future__ import annotations


class AutoFallbackController:
    """State machine deciding when to switch SIGReg to VICReg.

    Attributes:
        d: Latent dimension; PR threshold is ``pr_threshold * d``.
        threshold_iter: Earliest iteration at which fallback can fire.
        pr_threshold: PR / d below which the latent is considered collapsed.
        r2_threshold: Probe R^2 below which c is considered not recoverable.
        fired: ``True`` once the controller has fired the fallback.
        fired_at_iter: The iteration at which the fallback fired, or ``None``.
        history: List of ``(iteration, pr, probe_r2)`` tuples in step order
            (for W&B logging and post-hoc inspection).
    """

    def __init__(
        self,
        d: int,
        threshold_iter: int = 20_000,
        pr_threshold: float = 0.3,
        r2_threshold: float = 0.7,
    ) -> None:
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        if threshold_iter < 0:
            raise ValueError(f"threshold_iter must be non-negative, got {threshold_iter}")
        if not 0.0 <= pr_threshold <= 1.0:
            raise ValueError(f"pr_threshold must be in [0, 1], got {pr_threshold}")

        self.d = int(d)
        self.threshold_iter = int(threshold_iter)
        self.pr_threshold = float(pr_threshold)
        self.r2_threshold = float(r2_threshold)

        self.fired: bool = False
        self.fired_at_iter: int | None = None
        self.history: list[tuple[int, float, float]] = []

    def step(self, iteration: int, pr: float, probe_r2: float) -> bool:
        """Returns ``True`` exactly once when all three conditions fire.

        Args:
            iteration: Current training iteration (0-indexed).
            pr: Latest participation ratio of the encoder latents on the
                held-out diagnostic batch.
            probe_r2: Latest linear-probe R^2 for ``c`` from the same batch.

        Returns:
            ``True`` if the controller is firing the fallback this step.
            Idempotent: once fired, returns ``False`` on every subsequent call.
        """
        self.history.append((int(iteration), float(pr), float(probe_r2)))
        if self.fired:
            return False
        if iteration < self.threshold_iter:
            return False
        if pr >= self.pr_threshold * self.d:
            return False
        if probe_r2 >= self.r2_threshold:
            return False
        self.fired = True
        self.fired_at_iter = int(iteration)
        return True
