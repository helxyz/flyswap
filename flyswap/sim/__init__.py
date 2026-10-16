"""Simulators: differentiable rate model (Tier A) and its calibration."""

from flyswap.sim.calibrate import (  # noqa: F401
    CalibrationCriteria,
    CalibrationResult,
    calibrate_gain,
)
from flyswap.sim.rate import RateParams, RateSimulator, to_bcoo  # noqa: F401
