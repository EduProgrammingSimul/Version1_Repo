"""Safety utilities for reinforcement-learning controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


class LagrangianSafety:
    """Simple dual update for constraint handling."""

    def __init__(self, constraint_targets: Dict[str, float], lr: float, lam_clip: float):
        self.constraint_targets = dict(constraint_targets)
        self.lr = float(lr)
        self.lam_clip = float(lam_clip)
        self.lambdas = {name: 0.0 for name in self.constraint_targets}

    def update_duals(self, estimates: Dict[str, float]) -> Dict[str, float]:
        updated = {}
        for name, target in self.constraint_targets.items():
            estimate = float(estimates.get(name, 0.0))
            gap = estimate - float(target)
            lam = self.lambdas.get(name, 0.0) + self.lr * gap
            lam = float(np.clip(lam, 0.0, self.lam_clip))
            self.lambdas[name] = lam
            updated[name] = lam
        return updated

    def penalty(self, g: Dict[str, float]) -> float:
        total = 0.0
        for name, lam in self.lambdas.items():
            total += float(lam) * float(g.get(name, 0.0))
        return float(total)


def residual_action(u_flc: float, du_rl: float, du_max: float, u_min: float, u_max: float) -> float:
    """Combine fuzzy logic controller action with residual RL increment."""

    u_min_f = float(u_min)
    u_max_f = float(u_max)
    base = float(u_flc)
    delta = float(np.clip(du_rl, -abs(du_max), abs(du_max)))
    out = float(np.clip(base + delta, u_min_f, u_max_f))
    return out


__all__ = ["LagrangianSafety", "residual_action"]
