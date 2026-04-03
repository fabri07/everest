from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
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
from .paths import DATA_PROCESSED_DIR, METRICS_DIR, PREDICTIONS_DIR, SHAP_DIR, TABLES_DIR, ensure_output_dirs


RANDOM_SEED = 42


@dataclass
class PlattCalibrator:
    model: LogisticRegression | None = None
    constant_probability: float | None = None

    def fit(self, raw_scores: np.ndarray, y_true: pd.Series) -> "PlattCalibrator":
        y_array = np.asarray(y_true).astype(int)
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        unique_classes = np.unique(y_array)
        if len(unique_classes) < 2:
            self.constant_probability = float(y_array.mean())
            return self

        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        self.model.fit(raw_scores, y_array)
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        if self.model is not None:
            return self.model.predict_proba(raw_scores)[:, 1]
        if self.constant_probability is None:
            raise RuntimeError("Calibrator has not been fitted.")
        return np.full(shape=(len(raw_scores),), fill_value=self.constant_probability)


def build_feature_list(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_features = [
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
    categorical_features = [
        "mseason",
        "sex",
        "status",
        "citizen_clean",
        "residence_clean",
        "occupation_group",
        "occupation_raw_clean",
        "leader",
        "deputy",
        "support",
        "disabled",
        "hired",
        "sherpa",
        "tibetan",
        "bconly",
        "nottobc",
        "peakid",
        "peak_pkname",
        "peak_himal",
        "peak_region",
        "peak_open",
        "peak_trekking",
        "peak_restrict",
        "peak_phost",
        "peak_pstatus",
        "exp_host",
        "exp_route1",
        "exp_nation",
        "exp_traverse",
        "exp_ski",
        "exp_parapente",
        "exp_nohired",
        "exp_rope",
        "exped_ambiguous_expid",
        "age_missing_flag",
        "age_out_of_range_flag",
        "occupation_missing_flag",
    ]
    numeric_features = [column for column in numeric_features if column in df.columns]
    categorical_features = [column for column in categorical_features if column in df.columns]
    all_features = numeric_features + categorical_features
    return numeric_features, categorical_features, all_features


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["myear"] <= 2014].copy()
    valid = df.loc[(df["myear"] >= 2015) & (df["myear"] <= 2019)].copy()
    test = df.loc[df["myear"] >= 2020].copy()
    return train, valid, test


def prepare_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    features = df[numeric_features + categorical_features].copy()
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical_features:
        features[column] = (
            features[column]
            .astype("object")
            .where(features[column].notna(), "Unknown")
            .astype(str)
            .replace("", "Unknown")
        )
    return features


def make_catboost_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def binary_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true_array = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_array, y_prob)),
        "brier": float(brier_score_loss(y_true_array, y_prob)),
        "log_loss": float(log_loss(y_true_array, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "classification_report": classification_report(
            y_true_array,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }
    if len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob))
    return metrics


def multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def fit_binary_model(
    train_df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    model = make_catboost_model()
    x_train = prepare_features(train_df, numeric_features, categorical_features)
    y_train = train_df[target_column].astype(int)
    model.fit(x_train, y_train, cat_features=categorical_features)
    return model, x_train


def predict_raw_scores(model: CatBoostClassifier, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features, prediction_type="RawFormulaVal").astype(float)


def export_shap_outputs(
    model: CatBoostClassifier,
    x_sample: pd.DataFrame,
    model_name: str,
) -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    importance = (
        pd.DataFrame(
            {
                "feature": x_sample.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(SHAP_DIR / f"{model_name}_shap_importance.csv", index=False)

    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, x_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"{model_name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"{model_name}_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def sample_for_shap(df: pd.DataFrame, max_rows: int = 3000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=RANDOM_SEED).copy()


def evaluate_hierarchical_outputs(
    split_df: pd.DataFrame,
    summit_prob: np.ndarray,
    death_given_summit: np.ndarray,
    death_given_no_summit: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    death_prob = (summit_prob * death_given_summit) + (
        (1 - summit_prob) * death_given_no_summit
    )
    outcome_probs = pd.DataFrame(
        {
            "abandona_vive": (1 - summit_prob) * (1 - death_given_no_summit),
            "abandona_muere": (1 - summit_prob) * death_given_no_summit,
            "cima_vive": summit_prob * (1 - death_given_summit),
            "cima_muere": summit_prob * death_given_summit,
        },
        index=split_df.index,
    )
    outcome_pred = outcome_probs.idxmax(axis=1)

    metrics = {
        "summit_metrics": binary_metrics(split_df["msuccess"].astype(int), summit_prob),
        "death_metrics": binary_metrics(split_df["death"].astype(int), death_prob),
        "outcome_4_metrics": multiclass_metrics(split_df["outcome_4"], outcome_pred),
    }

    predictions = split_df[["expid", "peakid", "myear", "outcome_4"]].copy()
    predictions["p_summit"] = summit_prob
    predictions["p_death"] = death_prob
    predictions["p_death_given_summit"] = death_given_summit
    predictions["p_death_given_no_summit"] = death_given_no_summit
    predictions["p_abandona_vive"] = outcome_probs["abandona_vive"].values
    predictions["p_abandona_muere"] = outcome_probs["abandona_muere"].values
    predictions["p_cima_vive"] = outcome_probs["cima_vive"].values
    predictions["p_cima_muere"] = outcome_probs["cima_muere"].values
    predictions["outcome_pred"] = outcome_pred.values
    return metrics, predictions


def main() -> None:
    ensure_output_dirs()

    df = pd.read_csv(DATA_PROCESSED_DIR / "himalaya_model_ready.csv", low_memory=False)
    train, valid, test = temporal_split(df)
    train = train.dropna(subset=["msuccess", "death"]).copy()
    valid = valid.dropna(subset=["msuccess", "death"]).copy()
    test = test.dropna(subset=["msuccess", "death"]).copy()

    numeric_features, categorical_features, all_features = build_feature_list(df)
    x_valid = prepare_features(valid, numeric_features, categorical_features)
    x_test = prepare_features(test, numeric_features, categorical_features)

    summit_model, _ = fit_binary_model(train, "msuccess", numeric_features, categorical_features)
    train_summit = train.loc[train["msuccess"] == True].copy()
    train_no_summit = train.loc[train["msuccess"] == False].copy()
    valid_summit = valid.loc[valid["msuccess"] == True].copy()
    valid_no_summit = valid.loc[valid["msuccess"] == False].copy()

    death_summit_model, _ = fit_binary_model(
        train_summit,
        "death",
        numeric_features,
        categorical_features,
    )
    death_no_summit_model, _ = fit_binary_model(
        train_no_summit,
        "death",
        numeric_features,
        categorical_features,
    )

    summit_calibrator = PlattCalibrator().fit(
        predict_raw_scores(summit_model, x_valid),
        valid["msuccess"].astype(int),
    )
    death_summit_calibrator = PlattCalibrator().fit(
        predict_raw_scores(
            death_summit_model,
            prepare_features(valid_summit, numeric_features, categorical_features),
        ),
        valid_summit["death"].astype(int),
    )
    death_no_summit_calibrator = PlattCalibrator().fit(
        predict_raw_scores(
            death_no_summit_model,
            prepare_features(valid_no_summit, numeric_features, categorical_features),
        ),
        valid_no_summit["death"].astype(int),
    )

    valid_uncalibrated_metrics, valid_uncalibrated_predictions = evaluate_hierarchical_outputs(
        valid,
        summit_model.predict_proba(x_valid)[:, 1],
        death_summit_model.predict_proba(x_valid)[:, 1],
        death_no_summit_model.predict_proba(x_valid)[:, 1],
    )
    test_uncalibrated_metrics, test_uncalibrated_predictions = evaluate_hierarchical_outputs(
        test,
        summit_model.predict_proba(x_test)[:, 1],
        death_summit_model.predict_proba(x_test)[:, 1],
        death_no_summit_model.predict_proba(x_test)[:, 1],
    )

    test_calibrated_metrics, test_calibrated_predictions = evaluate_hierarchical_outputs(
        test,
        summit_calibrator.predict_proba(predict_raw_scores(summit_model, x_test)),
        death_summit_calibrator.predict_proba(
            predict_raw_scores(death_summit_model, x_test)
        ),
        death_no_summit_calibrator.predict_proba(
            predict_raw_scores(death_no_summit_model, x_test)
        ),
    )

    metrics = {
        "model_type": "catboost_hierarchical_binary_with_platt_calibration",
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "all_count": len(all_features),
        },
        "splits": {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
        },
        "validation_uncalibrated": valid_uncalibrated_metrics,
        "calibration_fit_split": "valid",
        "test_uncalibrated": test_uncalibrated_metrics,
        "test_calibrated": test_calibrated_metrics,
    }

    with (METRICS_DIR / "catboost_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)

    valid_uncalibrated_predictions["split"] = "valid"
    valid_uncalibrated_predictions["calibration"] = "uncalibrated"
    test_uncalibrated_predictions["split"] = "test"
    test_uncalibrated_predictions["calibration"] = "uncalibrated"
    test_calibrated_predictions["split"] = "test"
    test_calibrated_predictions["calibration"] = "calibrated"

    pd.concat(
        [
            valid_uncalibrated_predictions,
            test_uncalibrated_predictions,
            test_calibrated_predictions,
        ],
        ignore_index=True,
    ).to_csv(PREDICTIONS_DIR / "catboost_predictions.csv", index=False)

    shap_samples = {
        "summit_model": sample_for_shap(x_test),
        "death_given_summit_model": sample_for_shap(
            prepare_features(test.loc[test["msuccess"] == True], numeric_features, categorical_features),
            max_rows=2000,
        ),
        "death_given_no_summit_model": sample_for_shap(
            prepare_features(test.loc[test["msuccess"] == False], numeric_features, categorical_features),
            max_rows=2000,
        ),
    }

    export_shap_outputs(summit_model, shap_samples["summit_model"], "summit_model")
    export_shap_outputs(
        death_summit_model,
        shap_samples["death_given_summit_model"],
        "death_given_summit_model",
    )
    export_shap_outputs(
        death_no_summit_model,
        shap_samples["death_given_no_summit_model"],
        "death_given_no_summit_model",
    )

    calibration_summary = pd.DataFrame(
        [
            {
                "model": "summit",
                "valid_uncalibrated_mean_prob": float(
                    summit_model.predict_proba(x_valid)[:, 1].mean()
                ),
                "valid_calibrated_mean_prob": float(
                    summit_calibrator.predict_proba(predict_raw_scores(summit_model, x_valid)).mean()
                ),
                "valid_observed_rate": float(valid["msuccess"].mean()),
                "test_uncalibrated_mean_prob": float(
                    summit_model.predict_proba(x_test)[:, 1].mean()
                ),
                "test_calibrated_mean_prob": float(
                    summit_calibrator.predict_proba(predict_raw_scores(summit_model, x_test)).mean()
                ),
                "test_observed_rate": float(test["msuccess"].mean()),
            },
            {
                "model": "death_given_summit",
                "valid_uncalibrated_mean_prob": float(
                    death_summit_model.predict_proba(
                        prepare_features(valid_summit, numeric_features, categorical_features)
                    )[:, 1].mean()
                ),
                "valid_calibrated_mean_prob": float(
                    death_summit_calibrator.predict_proba(
                        predict_raw_scores(
                            death_summit_model,
                            prepare_features(valid_summit, numeric_features, categorical_features),
                        )
                    ).mean()
                ),
                "valid_observed_rate": float(valid_summit["death"].mean()),
                "test_uncalibrated_mean_prob": float(
                    death_summit_model.predict_proba(
                        prepare_features(
                            test.loc[test["msuccess"] == True],
                            numeric_features,
                            categorical_features,
                        )
                    )[:, 1].mean()
                ),
                "test_calibrated_mean_prob": float(
                    death_summit_calibrator.predict_proba(
                        predict_raw_scores(
                            death_summit_model,
                            prepare_features(
                                test.loc[test["msuccess"] == True],
                                numeric_features,
                                categorical_features,
                            ),
                        )
                    ).mean()
                ),
                "test_observed_rate": float(test.loc[test["msuccess"] == True, "death"].mean()),
            },
            {
                "model": "death_given_no_summit",
                "valid_uncalibrated_mean_prob": float(
                    death_no_summit_model.predict_proba(
                        prepare_features(
                            valid_no_summit,
                            numeric_features,
                            categorical_features,
                        )
                    )[:, 1].mean()
                ),
                "valid_calibrated_mean_prob": float(
                    death_no_summit_calibrator.predict_proba(
                        predict_raw_scores(
                            death_no_summit_model,
                            prepare_features(
                                valid_no_summit,
                                numeric_features,
                                categorical_features,
                            ),
                        )
                    ).mean()
                ),
                "valid_observed_rate": float(valid_no_summit["death"].mean()),
                "test_uncalibrated_mean_prob": float(
                    death_no_summit_model.predict_proba(
                        prepare_features(
                            test.loc[test["msuccess"] == False],
                            numeric_features,
                            categorical_features,
                        )
                    )[:, 1].mean()
                ),
                "test_calibrated_mean_prob": float(
                    death_no_summit_calibrator.predict_proba(
                        predict_raw_scores(
                            death_no_summit_model,
                            prepare_features(
                                test.loc[test["msuccess"] == False],
                                numeric_features,
                                categorical_features,
                            ),
                        )
                    ).mean()
                ),
                "test_observed_rate": float(test.loc[test["msuccess"] == False, "death"].mean()),
            },
        ]
    )
    calibration_summary.to_csv(TABLES_DIR / "catboost_calibration_summary.csv", index=False)

    print("CatBoost training complete.")
    print(f"Train rows: {len(train)}")
    print(f"Valid rows: {len(valid)}")
    print(f"Test rows: {len(test)}")
    print(f"Features used: {len(all_features)}")
    print("Outputs written to reports/metrics, reports/predictions, reports/tables and reports/figures/shap.")


if __name__ == "__main__":
    main()
