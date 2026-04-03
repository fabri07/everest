from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_DIR = FIGURES_DIR / "shap"


def ensure_output_dirs() -> None:
    for directory in [
        DATA_PROCESSED_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        SHAP_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

