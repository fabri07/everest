from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "01_data_ingestion" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "02_data_processing" / "data"
METRICS_DIR = ROOT_DIR / "04_model_evaluation" / "metrics"
PREDICTIONS_DIR = ROOT_DIR / "04_model_evaluation" / "predictions"
TABLES_DIR = ROOT_DIR / "05_data_analytics" / "tables"
FIGURES_DIR = ROOT_DIR / "04_model_evaluation" / "figures"
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
