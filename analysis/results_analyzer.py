import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


def calculate_kpis(
    results_df: pd.DataFrame, target_speed_rpm: float
) -> Dict[str, float]:

    kpis = {
        "iae_speed_rpm_s": np.nan,
        "overshoot_speed_percent": np.nan,
    }

    if results_df.empty:
        logger.warning("KPI Calculation failed: Input DataFrame is empty.")
        return kpis
    if (
        "time_s" not in results_df.columns
        or "turbine_speed_rpm" not in results_df.columns
    ):
        logger.warning(
            f"KPI Calculation failed: DataFrame is missing required columns ('time_s', 'turbine_speed_rpm')."
        )
        return kpis
    if len(results_df) < 2:
        logger.warning("KPI Calculation failed: At least 2 data points are required.")
        return kpis

    try:

        df = results_df.sort_values("time_s").reset_index(drop=True)
        dt = df["time_s"].diff().fillna(0).values
        speed = df["turbine_speed_rpm"].values

        speed_error = speed - target_speed_rpm

        kpis["iae_speed_rpm_s"] = np.nansum(np.abs(speed_error) * dt)

        max_speed = np.nanmax(speed)
        if pd.notna(max_speed) and abs(target_speed_rpm) > 1e-6:
            overshoot_value = max_speed - target_speed_rpm

            kpis["overshoot_speed_percent"] = max(
                0.0, (overshoot_value / target_speed_rpm) * 100.0
            )
        else:
            kpis["overshoot_speed_percent"] = 0.0

        logger.info(f"KPIs calculated successfully: {kpis}")

    except Exception as e:
        logger.error(f"Error during KPI calculation: {e}", exc_info=True)

        return {key: np.nan for key in kpis}

    return kpis
