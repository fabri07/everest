from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .paths import DATA_PROCESSED_DIR, METRICS_DIR, PREDICTIONS_DIR, ensure_output_dirs


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
        "leader",
        "deputy",
        "support",
        "disabled",
        "hired",
        "sherpa",
        "tibetan",
        "bconly",
        "nottobc",
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


def make_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="saga",
                ),
            ),
        ]
    )


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["myear"] <= 2014].copy()
    valid = df.loc[(df["myear"] >= 2015) & (df["myear"] <= 2019)].copy()
    test = df.loc[df["myear"] >= 2020].copy()
    return train, valid, test


def binary_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


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


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv(DATA_PROCESSED_DIR / "himalaya_model_ready.csv", low_memory=False)
    train, valid, test = temporal_split(df)
    numeric_features, categorical_features, all_features = build_feature_list(df)

    train = train.dropna(subset=["msuccess", "death"])
    valid = valid.dropna(subset=["msuccess", "death"])
    test = test.dropna(subset=["msuccess", "death"])

    summit_model = make_model(numeric_features, categorical_features)
    summit_model.fit(train[all_features], train["msuccess"].astype(int))

    death_summit_model = make_model(numeric_features, categorical_features)
    train_summit = train.loc[train["msuccess"] == True].copy()
    death_summit_model.fit(train_summit[all_features], train_summit["death"].astype(int))

    death_no_summit_model = make_model(numeric_features, categorical_features)
    train_no_summit = train.loc[train["msuccess"] == False].copy()
    death_no_summit_model.fit(train_no_summit[all_features], train_no_summit["death"].astype(int))

    metrics = {"splits": {}}
    predictions_output = []

    for split_name, split_df in [("valid", valid), ("test", test)]:
        summit_prob = summit_model.predict_proba(split_df[all_features])[:, 1]
        death_given_summit = death_summit_model.predict_proba(split_df[all_features])[:, 1]
        death_given_no_summit = death_no_summit_model.predict_proba(split_df[all_features])[:, 1]
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

        metrics["splits"][split_name] = {
            "n_rows": int(len(split_df)),
            "summit_metrics": binary_metrics(split_df["msuccess"].astype(int), summit_prob),
            "death_metrics": binary_metrics(split_df["death"].astype(int), death_prob),
            "outcome_4_metrics": multiclass_metrics(split_df["outcome_4"], outcome_pred),
        }

        split_predictions = split_df[["expid", "peakid", "myear", "outcome_4"]].copy()
        split_predictions["split"] = split_name
        split_predictions["p_abandona_vive"] = outcome_probs["abandona_vive"].values
        split_predictions["p_abandona_muere"] = outcome_probs["abandona_muere"].values
        split_predictions["p_cima_vive"] = outcome_probs["cima_vive"].values
        split_predictions["p_cima_muere"] = outcome_probs["cima_muere"].values
        split_predictions["outcome_pred"] = outcome_pred.values
        predictions_output.append(split_predictions)

    with (METRICS_DIR / "baseline_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)

    pd.concat(predictions_output, ignore_index=True).to_csv(
        PREDICTIONS_DIR / "baseline_predictions.csv",
        index=False,
    )

    print("Baseline training complete.")
    print(f"Train rows: {len(train)}")
    print(f"Valid rows: {len(valid)}")
    print(f"Test rows: {len(test)}")
    print(f"Features used: {len(all_features)}")


if __name__ == "__main__":
    main()
