from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .paths import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    SHAP_DIR,
    TABLES_DIR,
    ensure_output_dirs,
)

RANDOM_SEED = 42
TARGET = "exito"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "myear",
    "age_clean",
    "peak_heightm",
    "exp_totmembers",
    "exp_tothired",
    "exp_camps",
    "ref_n_refs",
    "ref_n_unique_types",
    "ref_pubyear_min",
    "ref_pubyear_max",
    "ref_n_languages",
    "peak_prev_summit_rate",
    "peak_prev_death_rate",
]

# Categoricas de baja cardinalidad (seguras para OHE en sklearn)
CATEGORICAL_FEATURES = [
    "mseason",
    "sex",
    "occupation_group",
    "leader",
    "deputy",
    "support",
    "disabled",
    "hired",
    "sherpa",
    "tibetan",
    "bconly",
    "nottobc",
    "peak_himal",
    "peak_region",
    "peak_open",
    "peak_trekking",
    "peak_restrict",
    "peak_phost",
    "peak_pstatus",
    "exp_host",
    "exp_traverse",
    "exp_ski",
    "exp_parapente",
    "exp_nohired",
    "exp_rope",
    "age_missing_flag",
    "age_out_of_range_flag",
    "occupation_missing_flag",
]

# Categoricas de alta cardinalidad: solo CatBoost las maneja bien
HIGH_CARD_FEATURES = [
    "citizen_clean",
    "status",
    "exp_route1",
    "exp_nation",
]


def build_feature_list(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Features compartidas por todos los modelos (sin alta cardinalidad)."""
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num, cat, num + cat


def build_catboost_feature_list(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Features para CatBoost (incluye alta cardinalidad)."""
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES + HIGH_CARD_FEATURES if c in df.columns]
    return num, cat, num + cat


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["myear"] <= 2014].copy()
    valid = df.loc[(df["myear"] >= 2015) & (df["myear"] <= 2019)].copy()
    test = df.loc[df["myear"] >= 2020].copy()
    return train, valid, test


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_catboost_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    features = df[numeric_features + categorical_features].copy()
    for col in numeric_features:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    for col in categorical_features:
        features[col] = (
            features[col]
            .astype("object")
            .where(features[col].notna(), "Unknown")
            .astype(str)
            .replace("", "Unknown")
        )
    return features


def make_sklearn_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    estimator,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True,
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def train_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    x_train = prepare_catboost_features(train, numeric_features, categorical_features)
    y_train = train[TARGET].astype(int)
    x_valid = prepare_catboost_features(valid, numeric_features, categorical_features)
    y_valid = valid[TARGET].astype(int)
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        cat_features=categorical_features,
        early_stopping_rounds=50,
    )
    return model


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logreg(
    train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    all_features: list[str],
) -> Pipeline:
    pipeline = make_sklearn_pipeline(
        numeric_features,
        categorical_features,
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            penalty="l1",
            C=1.0,
            random_state=RANDOM_SEED,
        ),
    )
    pipeline.fit(train[all_features], train[TARGET].astype(int))
    return pipeline


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    all_features: list[str],
) -> Pipeline:
    pipeline = make_sklearn_pipeline(
        numeric_features,
        categorical_features,
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    pipeline.fit(train[all_features], train[TARGET].astype(int))
    return pipeline


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------

def get_sklearn_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(columns).tolist())
    return names


def catboost_importance(model: CatBoostClassifier) -> pd.DataFrame:
    imp = model.get_feature_importance()
    names = model.feature_names_
    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def logreg_importance(pipeline: Pipeline) -> pd.DataFrame:
    feature_names = get_sklearn_feature_names(pipeline)
    coefs = np.abs(pipeline.named_steps["model"].coef_[0])
    df = pd.DataFrame({"feature": feature_names, "importance": coefs})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def rf_importance(pipeline: Pipeline) -> pd.DataFrame:
    feature_names = get_sklearn_feature_names(pipeline)
    imp = pipeline.named_steps["model"].feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def compute_permutation_importance(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
    name: str,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, x, y,
        n_repeats=10,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    feature_names = list(x.columns)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["model"] = name
    return df


def build_consensus_ranking(
    catboost_imp: pd.DataFrame,
    logreg_imp: pd.DataFrame,
    rf_imp: pd.DataFrame,
    original_features: set[str] | None = None,
) -> pd.DataFrame:
    """Consensus basado en features originales (no expandidas por OHE).
    Para LogReg y RF, las features OHE se colapsan sumando importancia por prefijo."""

    # Features originales: si no se pasan, usar las de CatBoost
    cb_features = set(catboost_imp["feature"].tolist())
    if original_features is None:
        original_features = cb_features

    all_known = original_features | cb_features

    def map_to_original(feat: str) -> str:
        if feat in all_known:
            return feat
        for orig in sorted(all_known, key=len, reverse=True):
            if feat.startswith(orig + "_"):
                return orig
        return feat

    def collapse_to_original(imp_df: pd.DataFrame) -> pd.DataFrame:
        mapped = imp_df.copy()
        mapped["original_feature"] = mapped["feature"].apply(map_to_original)
        collapsed = (
            mapped.groupby("original_feature", as_index=False)["importance"]
            .sum()
            .rename(columns={"original_feature": "feature"})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        collapsed["rank"] = range(1, len(collapsed) + 1)
        return collapsed

    lr_collapsed = collapse_to_original(logreg_imp)
    rf_collapsed = collapse_to_original(rf_imp)

    # Re-rank catboost
    cb = catboost_imp[["feature", "rank"]].rename(columns={"rank": "rank_catboost"})
    lr = lr_collapsed[["feature", "rank"]].rename(columns={"rank": "rank_logreg"})
    rf = rf_collapsed[["feature", "rank"]].rename(columns={"rank": "rank_rf"})

    consensus = cb.merge(lr, on="feature", how="outer").merge(rf, on="feature", how="outer")

    max_rank = len(consensus) + 1
    for col in ["rank_catboost", "rank_logreg", "rank_rf"]:
        consensus[col] = consensus[col].fillna(max_rank)

    consensus["mean_rank"] = consensus[["rank_catboost", "rank_logreg", "rank_rf"]].mean(axis=1)
    consensus["n_top10"] = (
        (consensus["rank_catboost"] <= 10).astype(int)
        + (consensus["rank_logreg"] <= 10).astype(int)
        + (consensus["rank_rf"] <= 10).astype(int)
    )
    consensus = consensus.sort_values(["n_top10", "mean_rank"], ascending=[False, True])
    consensus = consensus.reset_index(drop=True)
    consensus["consensus_rank"] = range(1, len(consensus) + 1)
    return consensus


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def export_shap(model: CatBoostClassifier, x_sample: pd.DataFrame) -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    importance = (
        pd.DataFrame({
            "feature": x_sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(SHAP_DIR / "exito_shap_importance.csv", index=False)

    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, x_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "exito_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "exito_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_consensus(consensus: pd.DataFrame, top_n: int = 20) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = consensus.head(top_n).copy()
    top = top.sort_values("mean_rank", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(top))
    bars = ax.barh(y_pos, top["mean_rank"].max() - top["mean_rank"] + 1, color="#2196F3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Relevancia (inverso del rank promedio)")
    ax.set_title(f"Top {top_n} features - Consenso CatBoost + LogReg + RF")

    for i, (_, row) in enumerate(top.iterrows()):
        label = f"top10 en {int(row['n_top10'])}/3"
        ax.text(0.5, i, label, va="center", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "consensus_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    df = pd.read_csv(DATA_PROCESSED_DIR / "himalaya_model_ready.csv", low_memory=False)
    df = df.dropna(subset=["msuccess", "death"]).copy()

    if TARGET not in df.columns:
        df[TARGET] = ((df["msuccess"] == True) & (df["death"] == False)).astype(int)

    # Features separadas: CatBoost soporta alta cardinalidad, sklearn no
    num_feat, cat_feat, sk_features = build_feature_list(df)
    cb_num, cb_cat, cb_features = build_catboost_feature_list(df)
    train, valid, test = temporal_split(df)

    print(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
    print(f"CatBoost features: {len(cb_features)} | sklearn features: {len(sk_features)}")
    print(f"Target distribution (train): {train[TARGET].value_counts().to_dict()}")

    # --- CatBoost (con alta cardinalidad) ---
    print("\nEntrenando CatBoost...")
    cb_model = train_catboost(train, valid, cb_num, cb_cat)
    x_test_cb = prepare_catboost_features(test, cb_num, cb_cat)
    x_valid_cb = prepare_catboost_features(valid, cb_num, cb_cat)
    cb_prob_test = cb_model.predict_proba(x_test_cb)[:, 1]
    cb_prob_valid = cb_model.predict_proba(x_valid_cb)[:, 1]

    # --- Logistic Regression (sin alta cardinalidad) ---
    print("Entrenando Logistic Regression (L1)...")
    lr_pipeline = train_logreg(train, num_feat, cat_feat, sk_features)
    lr_prob_test = lr_pipeline.predict_proba(test[sk_features])[:, 1]
    lr_prob_valid = lr_pipeline.predict_proba(valid[sk_features])[:, 1]

    # --- Random Forest (sin alta cardinalidad) ---
    print("Entrenando Random Forest...")
    rf_pipeline = train_random_forest(train, num_feat, cat_feat, sk_features)
    rf_prob_test = rf_pipeline.predict_proba(test[sk_features])[:, 1]
    rf_prob_valid = rf_pipeline.predict_proba(valid[sk_features])[:, 1]

    # --- Metrics ---
    y_valid = valid[TARGET].astype(int).values
    y_test = test[TARGET].astype(int).values

    metrics = {
        "target": TARGET,
        "description": "Clasificacion binaria: exito (cima + vive) vs todo lo demas",
        "features": {
            "catboost": {"numeric": cb_num, "categorical": cb_cat, "total": len(cb_features)},
            "sklearn": {"numeric": num_feat, "categorical": cat_feat, "total": len(sk_features)},
        },
        "splits": {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
        },
        "target_distribution": {
            "train": train[TARGET].value_counts().to_dict(),
            "valid": valid[TARGET].value_counts().to_dict(),
            "test": test[TARGET].value_counts().to_dict(),
        },
        "validation": {
            "catboost": binary_metrics(y_valid, cb_prob_valid),
            "logreg": binary_metrics(y_valid, lr_prob_valid),
            "random_forest": binary_metrics(y_valid, rf_prob_valid),
        },
        "test": {
            "catboost": binary_metrics(y_test, cb_prob_test),
            "logreg": binary_metrics(y_test, lr_prob_test),
            "random_forest": binary_metrics(y_test, rf_prob_test),
        },
    }

    with (METRICS_DIR / "success_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)
    print("Metricas guardadas.")

    # --- Predictions ---
    predictions = test[["expid", "peakid", "myear", "outcome_4", TARGET]].copy()
    predictions["p_exito_catboost"] = cb_prob_test
    predictions["p_exito_logreg"] = lr_prob_test
    predictions["p_exito_rf"] = rf_prob_test
    predictions.to_csv(PREDICTIONS_DIR / "success_predictions.csv", index=False)

    # --- Feature importance ---
    print("\nCalculando importancias...")
    cb_imp = catboost_importance(cb_model)
    lr_imp = logreg_importance(lr_pipeline)
    rf_imp_df = rf_importance(rf_pipeline)

    cb_imp.to_csv(TABLES_DIR / "importance_catboost.csv", index=False)
    lr_imp.to_csv(TABLES_DIR / "importance_logreg.csv", index=False)
    rf_imp_df.to_csv(TABLES_DIR / "importance_rf.csv", index=False)

    # --- Permutation importance (CatBoost on test) ---
    print("Calculando permutation importance (CatBoost test)...")
    cb_perm = compute_permutation_importance(
        cb_model, x_test_cb, y_test, "catboost"
    )
    cb_perm.to_csv(TABLES_DIR / "permutation_importance_catboost.csv", index=False)

    # --- Consensus ranking (colapsa OHE a features originales de sklearn) ---
    # Usamos las features compartidas (sk) como referencia para el consenso
    all_original = set(num_feat + cat_feat)
    consensus = build_consensus_ranking(cb_imp, lr_imp, rf_imp_df, all_original)
    consensus.to_csv(TABLES_DIR / "consensus_feature_ranking.csv", index=False)
    print("\nTop 15 features por consenso:")
    print(consensus.head(15)[["consensus_rank", "feature", "n_top10", "mean_rank"]].to_string(index=False))

    # --- SHAP (CatBoost) ---
    print("\nGenerando SHAP...")
    shap_sample = x_test_cb.sample(n=min(3000, len(x_test_cb)), random_state=RANDOM_SEED)
    export_shap(cb_model, shap_sample)

    # --- Consensus plot ---
    plot_consensus(consensus)

    print("\nPipeline completo.")
    print(f"Outputs en: {METRICS_DIR}, {PREDICTIONS_DIR}, {TABLES_DIR}, {FIGURES_DIR}")


if __name__ == "__main__":
    main()
