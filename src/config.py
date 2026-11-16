"""项目路径、储能参数与统一常量。"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECK_DIR = DATA_DIR / "checks"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
DOC_DIR = PROJECT_DIR / "docs"
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"

DELTA_T = 1 / 6
N_PERIODS = 144
DAY_HOURS = N_PERIODS * DELTA_T
S_INIT = 6000.0
S_FINAL = 6000.0
S_MIN = 1200.0
S_MAX = 10800.0
S_RATED_CAPACITY = 12000.0
ETA_C = 0.9
ETA_D = 0.9
P_B_MAX = 5000.0
E_B_MAX = P_B_MAX * DELTA_T
N_FOUR_HOUR_BLOCKS = 6
PERIODS_PER_FOUR_HOUR_BLOCK = 24
Q1_FOUR_HOUR_BLOCKS = (
    ("0:00-4:00", 2, 0, 24),
    ("4:00-8:00", 3, 24, 48),
    ("8:00-12:00", 4, 48, 72),
    ("12:00-16:00", 5, 72, 96),
    ("16:00-20:00", 6, 96, 120),
    ("20:00-24:00", 7, 120, 144),
)
Q1_TABLE1_PERIODS = (
    ("10:00-10:10", 59, 61),
    ("12:00-12:10", 71, 73),
    ("14:00-14:10", 83, 85),
    ("16:00-16:10", 95, 97),
    ("18:00-18:10", 107, 109),
    ("20:00-20:10", 119, 121),
)

UNITS = {
    "time_step": "h",
    "power": "kW",
    "energy": "kWh",
    "price": "yuan/kWh",
    "cost": "yuan",
}

# Compatibility aliases: use the names above in new code.
DT_HOURS = DELTA_T
N_SLOTS = N_PERIODS
SOC_INITIAL = S_INIT
SOC_MIN = S_MIN
SOC_MAX = S_MAX
ETA_CHARGE = ETA_C
ETA_DISCHARGE = ETA_D
BATTERY_RATED_CAPACITY = S_RATED_CAPACITY
BATTERY_POWER_MAX_KW = P_B_MAX
CHARGE_MAX_KWH = E_B_MAX
DISCHARGE_MAX_KWH = E_B_MAX

Q1_PROCESSED_FILE = PROCESSED_DATA_DIR / "q1_day_data.csv"
Q1_BASELINE_INTERFACE_FILE = PROCESSED_DATA_DIR / "q1_baseline_dispatch.csv"
Q1_RESULT_FILE = TABLE_DIR / "result1.xlsx"
Q1_PAPER_TABLES_FILE = TABLE_DIR / "q1_paper_tables.csv"
Q1_SOLVE_LOG_FILE = LOG_DIR / "q1_solve_log.txt"
Q1_VALIDATION_LOG_FILE = LOG_DIR / "q1_validation.txt"
Q1_SUBMISSION_CHECK_FILE = LOG_DIR / "q1_submission_check.txt"

# Question 2: January causal-backtest configuration.  These values are model
# candidates, not results; the chosen combination is determined by backtest.
Q2_INPUT_PATH = RAW_DATA_DIR / "附件2.xlsx"
Q2_JANUARY_BACKTEST_FILE = TABLE_DIR / "q2_january_parameter_comparison.csv"
Q2_BACKTEST_START = "2025-01-22"
Q2_BACKTEST_END = "2025-01-31"
Q2_HISTORY_WINDOWS_DAYS = (7, 14, 21)
Q2_NET_LOAD_QUANTILES = (0.60, 0.65, 0.70, 0.75, 0.80)
Q2_RESERVE_SOC_CANDIDATES = (S_MIN, 3600.0, 6000.0)
# Multiplied by the maximum normal price in Attachment 1 to obtain rho in yuan/kWh.
Q2_RESERVE_PENALTY_PRICE_MULTIPLIERS = (0.5, 1.0, 5.0)
Q2_BACKTEST_INITIAL_SOC = S_INIT
Q2_VALIDATION_TOLERANCE = 1e-6
Q2_FIXED_WINDOW_DAYS = 7
Q2_FIXED_QUANTILE = 0.75
Q2_FIXED_RHO = 0.6976
Q2_SHADOW_WARMUP_START = "2025-01-01"
Q2_SHADOW_FORMAL_START = "2025-02-01"
Q2_SHADOW_END = "2025-12-31"
Q2_FULL_YEAR_SHADOW_COMPARISON_FILE = TABLE_DIR / "q2_full_year_shadow_comparison.csv"
Q2_FULL_YEAR_FORECAST_METRICS_FILE = TABLE_DIR / "q2_full_year_forecast_metrics.csv"
Q2_DAILY_SHADOW_RESULTS_FILE = TABLE_DIR / "q2_daily_shadow_results.csv"
Q2_FULL_YEAR_SHADOW_VALIDATION_FILE = LOG_DIR / "q2_full_year_shadow_validation.txt"
Q2_BASELINE_APLUS_FIGURE = FIGURE_DIR / "q2_baseline_vs_aplus_summary.png"

# Question 3: deterministic short-run engineering validation only. These
# settings are not full-year selected parameters and must not be described as
# optimal. They are kept separate from the frozen Question-2 parameters.
Q3_FORECAST_INPUT_PATH = RAW_DATA_DIR / "附件3.xlsx"
Q3_ISSUE_HOURS = (0, 6, 12, 18)
Q3_FIRST_MODIFIABLE_SLOT = {0: 0, 6: 35, 12: 71, 18: 107}
Q3_LOAD_HISTORY_DAYS = 14
Q3_LOAD_DECAY_HALF_LIFE_DAYS = 7.0
Q3_LOAD_BIAS_LOOKBACK_SLOTS = 12
Q3_TEST_START = "2025-01-15"
Q3_TEST_END = "2025-01-17"
Q3_TEST_RESERVE_SOC_KWH = S_MIN
Q3_TEST_RESERVE_PENALTY_YUAN_PER_KWH = 0.0
Q3_VALIDATION_TOLERANCE = 1e-6
Q3_TEST_OUTPUT_DIR = OUTPUT_DIR / "q3_small_validation"

# Question 3: January causal parameter backtest.  These are deliberately
# modest candidates.  They are not inherited optimums from Question 2.
Q3_JANUARY_WARMUP_START = "2025-01-01"
Q3_JANUARY_EVALUATION_START = "2025-01-22"
Q3_JANUARY_EVALUATION_END = "2025-01-31"
Q3_LOAD_HISTORY_DAY_CANDIDATES = (7, 14, 21)
Q3_LOAD_BIAS_LOOKBACK_CANDIDATES = (0, 12)
Q3_LOAD_HALF_LIFE_DAYS = 7.0
Q3_RESERVE_SOC_CANDIDATES = (S_MIN, 3600.0, 6000.0)
# Reused only as a candidate protection coefficient, not as a Q3 optimum.
Q3_RESERVE_PENALTY_CANDIDATE = 0.6976
Q3_ADJUSTMENT_POLICIES = ("all_updates", "no_adjustment")
Q3_JANUARY_OUTPUT_DIR = OUTPUT_DIR / "q3_january_backtest"
Q3_JANUARY_PARAMETER_FILE = TABLE_DIR / "q3_january_parameter_comparison.csv"
Q3_INFORMATION_AUDIT_FILE = CHECK_DIR / "q3_pv_information_inputs.csv"
Q3_INFORMATION_AUDIT_LOG = CHECK_DIR / "q3_information_boundary_audit.txt"
Q3_FORMAL_START = "2025-02-01"
Q3_FORMAL_END = "2025-12-31"
Q3_FULL_YEAR_OUTPUT_DIR = OUTPUT_DIR / "q3_full_year"
Q3_FULL_YEAR_COMPARISON_FILE = TABLE_DIR / "q3_full_year_comparison.csv"
Q3_DAILY_RESULTS_FILE = TABLE_DIR / "q3_daily_results.csv"
Q3_FULL_YEAR_VALIDATION_FILE = LOG_DIR / "q3_full_year_validation.txt"
