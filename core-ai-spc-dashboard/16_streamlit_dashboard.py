# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TABLE_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

SYSTEM_FILE = (
    TABLE_DIR
    / "final_ai_spc_system.csv"
)

METRICS_FILE = (
    TABLE_DIR
    / "final_evaluation_metrics.csv"
)

FIRST_ALARM_FILE = (
    TABLE_DIR
    / "final_first_alarm_performance.csv"
)

THRESHOLD_FILE = (
    TABLE_DIR
    / "final_threshold_analysis.csv"
)

PROCESS_IMPORTANCE_FILE = (
    TABLE_DIR
    / "shap_process_interpretation.csv"
)