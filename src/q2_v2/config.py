"""Fixed physical constants and Stage-2 test settings for Q2-v2."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DELTA_T_HOURS = 1.0 / 6.0
N_SLOTS = 144
SOC_MIN_KWH = 1200.0
SOC_MAX_KWH = 10800.0
SOC_INITIAL_KWH = 6000.0
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
POWER_MAX_KW = 5000.0
ENERGY_MAX_KWH = POWER_MAX_KW * DELTA_T_HOURS
EMERGENCY_PRICE_MULTIPLIER = 5.0
TOL = 1e-6
# Absolute tolerance for declaring two DP action objective values tied.  This
# is a numerical secondary rule only; it is not a battery degradation charge.
ACTION_OBJECTIVE_TIE_TOL_YUAN = 1e-8

# Stage 2 is a mechanism test, not parameter selection.
STAGE2_START_DATE = "2025-01-15"
STAGE2_END_DATE = "2025-01-17"
STAGE2_SCENARIO_CAP = 14
STAGE2_DP_GRIDS_KWH = (100.0, 50.0)
# Stage 2 only verifies the mechanism.  A time-limited incumbent is admissible,
# but it must be reported as such and must never be labelled optimal.
MIP_REL_GAP = 1e-4
MIP_TIME_LIMIT_SECONDS = 45.0
