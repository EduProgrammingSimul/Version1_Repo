"""Action shielding utilities ensuring rate and magnitude limits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShieldLimits:
    """Boundary parameters for the shield."""

    u_min: float
    u_max: float
    rate_limit: float


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def apply_shield(u_raw: float, u_prev: float, lim: ShieldLimits) -> tuple[float, bool]:
    """Apply magnitude and slew-rate limits to an action.

    Parameters
    ----------
    u_raw:
        Proposed control action.
    u_prev:
        Previously applied action.
    lim:
        Shield parameterization.

    Returns
    -------
    tuple
        Safe action and whether any clipping occurred.
    """

    u_mag = _clip(float(u_raw), float(lim.u_min), float(lim.u_max))
    delta = u_mag - float(u_prev)
    max_step = abs(float(lim.rate_limit))
    if max_step == 0.0:
        u_rate = float(u_prev)
    else:
        delta = _clip(delta, -max_step, max_step)
        u_rate = float(u_prev) + delta
    u_safe = _clip(u_rate, float(lim.u_min), float(lim.u_max))
    was_clipped = not (
        u_safe == u_raw
        and u_safe == u_mag
        and abs(u_safe - float(u_prev)) <= max_step + 1e-12
    )
    return float(u_safe), bool(was_clipped)


__all__ = ["ShieldLimits", "apply_shield"]
