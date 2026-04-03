from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, METRICS_DIR, TABLES_DIR, ensure_output_dirs


MEMBER_DROP_COLUMNS = [
    "membid",
    "fname",
    "lname",
    "mchksum",
    "hcn",
    "msmtbid",
    "msmtterm",
    "msmtdate1",
    "msmtdate2",
    "msmtdate3",
    "msmttime1",
    "msmttime2",
    "msmttime3",
    "deathtype",
    "deathdate",
    "deathtime",
    "deathclass",
    "deathhgtm",
    "mhighpt",
    "mperhighpt",
    "mo2note",
    "mroute1",
    "mroute2",
    "mroute3",
    "mascent1",
    "mascent2",
    "mascent3",
    "mo2used",
    "mo2none",
    "mo2climb",
    "mo2descent",
    "mo2sleep",
    "mo2medical",
    "yob",
    "occupation",
]

EXPED_DROP_COLUMNS = [
    "exp_year",
    "exp_season",
    "exp_route2",
    "exp_route3",
    "exp_route4",
    "exp_ascent1",
    "exp_ascent2",
    "exp_ascent3",
    "exp_ascent4",
    "exp_success1",
    "exp_success2",
    "exp_success3",
    "exp_success4",
    "exp_claimed",
    "exp_disputed",
    "exp_countries",
    "exp_approach",
    "exp_bcdate",
    "exp_smtdate",
    "exp_smttime",
    "exp_smtdays",
    "exp_totdays",
    "exp_termdate",
    "exp_termreason",
    "exp_termnote",
    "exp_highpoint",
    "exp_smtmembers",
    "exp_mdeaths",
    "exp_smthired",
    "exp_hdeaths",
    "exp_o2used",
    "exp_o2none",
    "exp_o2climb",
    "exp_o2descent",
    "exp_o2sleep",
    "exp_o2medical",
    "exp_o2taken",
    "exp_o2unkwn",
    "exp_othersmts",
    "exp_campsites",
    "exp_accidents",
    "exp_achievment",
    "exp_agency",
    "exp_comrte",
    "exp_stdrte",
    "exp_primrte",
    "exp_primmem",
    "exp_primref",
    "exp_primid",
    "exp_chksum",
    "exp_sponsor",
    "exp_leaders",
]

PEAK_DROP_COLUMNS = [
    "peak_heightf",
    "peak_pkname2",
    "peak_trekyear",
    "peak_psummiters",
    "peak_psmtnote",
    "peak_pexpid",
    "peak_pcountry",
    "peak_pseason",
    "peak_pmonth",
    "peak_pday",
]

REF_RTYPE_MAP = {
    "website": "website",
    "journal": "journal",
    "magazine": "magazine",
    "book": "book",
    "report / brochure": "report_brochure",
    "himalayan database report": "hd_report",
}

OCCUPATION_PATTERNS = {
    "guide_instructor": [
        r"\bguide\b",
        r"\balpine guide\b",
        r"\bmountain guide\b",
        r"\binstructor\b",
        r"\bexpedition leader\b",
    ],
    "climber_alpinist": [
        r"\balpin",
        r"\bmountaineer\b",
        r"\bclimber\b",
    ],
    "doctor_health": [
        r"\bdoctor\b",
        r"\bphysician\b",
        r"\bsurgeon\b",
        r"\bdentist\b",
        r"\bnurse\b",
        r"\bmedical\b",
        r"\bparamedic\b",
    ],
    "engineer": [
        r"\bengineer\b",
        r"\btechnician\b",
    ],
    "military_police": [
        r"\barmy\b",
        r"\bmilitary\b",
        r"\bair force\b",
        r"\bnavy\b",
        r"\bpolice\b",
        r"\bsoldier\b",
        r"\bfirefighter\b",
    ],
    "student": [
        r"\bstudent\b",
        r"\bpupil\b",
    ],
    "business_exec": [
        r"\bbusiness",
        r"\bmanager\b",
        r"\bexecutive\b",
        r"\bentrepreneur\b",
        r"\bowner\b",
        r"\bpresident\b",
        r"\bceo\b",
        r"\bdirector\b",
        r"\bconsultant\b",
        r"\bbanker\b",
        r"\baccountant\b",
        r"\battorney\b",
        r"\blawyer\b",
    ],
    "scientist_academic": [
        r"\bprofessor\b",
        r"\bscientist\b",
        r"\bresearcher\b",
        r"\bteacher\b",
        r"\blecturer\b",
        r"\bacademic\b",
        r"\bgeologist\b",
        r"\bbiologist\b",
    ],
    "media_photo": [
        r"\bjournalist\b",
        r"\bwriter\b",
        r"\bphotographer\b",
        r"\bcameraman\b",
        r"\bfilm\b",
        r"\bcinema\b",
        r"\beditor\b",
        r"\bmedia\b",
    ],
    "trades_manual": [
        r"\bcarpenter\b",
        r"\belectrician\b",
        r"\bconstruction\b",
        r"\bbuilder\b",
        r"\bfarmer\b",
        r"\bgardener\b",
        r"\bdriver\b",
        r"\bworker\b",
        r"\bmechanic\b",
    ],
}


def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    members = pd.read_csv(DATA_RAW_DIR / "members.csv", low_memory=False)
    exped = pd.read_csv(DATA_RAW_DIR / "exped.csv", low_memory=False)
    peaks = pd.read_csv(DATA_RAW_DIR / "peaks.csv", low_memory=False)
    refer = pd.read_csv(DATA_RAW_DIR / "refer.csv", encoding="latin1", low_memory=False)
    return members, exped, peaks, refer


def collapse_group(series: pd.Series):
    values = [value for value in series.dropna().unique().tolist() if value != ""]
    if not values:
        return np.nan
    if len(values) == 1:
        return values[0]
    return np.nan


def resolve_exped_duplicates(exped: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    duplicate_expids = (
        exped.loc[exped.duplicated("expid", keep=False), "expid"].drop_duplicates().tolist()
    )
    grouped = exped.groupby("expid", dropna=False).agg(collapse_group).reset_index()
    grouped["exped_ambiguous_expid"] = grouped["expid"].isin(duplicate_expids)
    rename_map = {column: f"exp_{column}" for column in grouped.columns if column != "expid"}
    grouped = grouped.rename(columns=rename_map)
    return grouped, duplicate_expids


def summarize_references(refer: pd.DataFrame) -> pd.DataFrame:
    refer = refer.copy()
    refer["rtype_clean"] = (
        refer["rtype"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    refer["rpubdate_num"] = pd.to_numeric(refer["rpubdate"], errors="coerce")

    summary = refer.groupby("expid", dropna=False).agg(
        ref_n_refs=("refid", "count"),
        ref_n_unique_types=("rtype_clean", "nunique"),
        ref_pubyear_min=("rpubdate_num", "min"),
        ref_pubyear_max=("rpubdate_num", "max"),
        ref_n_languages=("rlanguage", "nunique"),
    )

    for raw_name, clean_name in REF_RTYPE_MAP.items():
        counts = (
            refer.assign(flag=refer["rtype_clean"].eq(raw_name).astype(int))
            .groupby("expid", dropna=False)["flag"]
            .sum()
            .rename(f"ref_n_{clean_name}")
        )
        summary = summary.join(counts, how="left")

    summary = summary.fillna(0).reset_index()
    return summary


def clean_text_column(series: pd.Series, fill_value: str = "Unknown") -> pd.Series:
    return (
        series.fillna(fill_value)
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", fill_value)
    )


def normalize_occupation(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s/\-]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", "unknown")
    )


def map_occupation_group(series: pd.Series) -> pd.Series:
    groups = pd.Series("other", index=series.index, dtype="object")
    groups[series.eq("unknown")] = "unknown"
    for group_name, patterns in OCCUPATION_PATTERNS.items():
        mask = pd.Series(False, index=series.index)
        for pattern in patterns:
            mask = mask | series.str.contains(pattern, regex=True, na=False)
        groups[mask & groups.eq("other")] = group_name
    return groups


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["outcome_4"] = "abandona_vive"
    df.loc[(df["msuccess"] == True) & (df["death"] == False), "outcome_4"] = "cima_vive"
    df.loc[(df["msuccess"] == True) & (df["death"] == True), "outcome_4"] = "cima_muere"
    df.loc[(df["msuccess"] != True) & (df["death"] == True), "outcome_4"] = "abandona_muere"
    return df


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_raw"] = df["myear"] - df["yob"]
    df["age_missing_flag"] = df["yob"].isna()
    df["age_out_of_range_flag"] = df["age_raw"].notna() & ~df["age_raw"].between(13, 85)
    df["age_clean"] = df["age_raw"].where(df["age_raw"].between(13, 85))
    return df


def add_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["citizen_clean"] = clean_text_column(df["citizen"])
    df["residence_clean"] = clean_text_column(df["residence"])
    df["occupation_missing_flag"] = df["occupation"].isna()
    df["occupation_raw_clean"] = normalize_occupation(df["occupation"])
    df["occupation_group"] = map_occupation_group(df["occupation_raw_clean"])
    return df


def add_historical_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["peakid", "myear", "expid"]).copy()
    expedition_history = (
        df.assign(
            _member_count=1,
            _summit_count=df["msuccess"].eq(True).astype(int),
            _death_count=df["death"].eq(True).astype(int),
        )
        .groupby(["peakid", "myear", "expid"], dropna=False, sort=False)
        .agg(
            expedition_members=("_member_count", "sum"),
            expedition_summits=("_summit_count", "sum"),
            expedition_deaths=("_death_count", "sum"),
        )
        .reset_index()
        .sort_values(["peakid", "myear", "expid"])
    )
    peak_group = expedition_history.groupby("peakid", dropna=False)
    expedition_history["peak_prev_members"] = (
        peak_group["expedition_members"].cumsum() - expedition_history["expedition_members"]
    )
    expedition_history["peak_prev_summits"] = (
        peak_group["expedition_summits"].cumsum() - expedition_history["expedition_summits"]
    )
    expedition_history["peak_prev_deaths"] = (
        peak_group["expedition_deaths"].cumsum() - expedition_history["expedition_deaths"]
    )
    expedition_history["peak_prev_summit_rate"] = np.where(
        expedition_history["peak_prev_members"] > 0,
        expedition_history["peak_prev_summits"] / expedition_history["peak_prev_members"],
        np.nan,
    )
    expedition_history["peak_prev_death_rate"] = np.where(
        expedition_history["peak_prev_members"] > 0,
        expedition_history["peak_prev_deaths"] / expedition_history["peak_prev_members"],
        np.nan,
    )
    df = df.merge(
        expedition_history[
            [
                "peakid",
                "myear",
                "expid",
                "peak_prev_members",
                "peak_prev_summits",
                "peak_prev_deaths",
                "peak_prev_summit_rate",
                "peak_prev_death_rate",
            ]
        ],
        on=["peakid", "myear", "expid"],
        how="left",
    )
    return df


def build_model_ready(master: pd.DataFrame) -> pd.DataFrame:
    drop_columns = set(MEMBER_DROP_COLUMNS + EXPED_DROP_COLUMNS + PEAK_DROP_COLUMNS)
    existing_drop_columns = [column for column in drop_columns if column in master.columns]
    model_ready = master.drop(columns=existing_drop_columns).copy()

    for column in ["citizen_clean", "residence_clean", "occupation_group", "status", "sex"]:
        if column in model_ready.columns:
            model_ready[column] = clean_text_column(model_ready[column])

    numeric_fill_columns = [
        "age_clean",
        "peak_prev_summit_rate",
        "peak_prev_death_rate",
        "ref_n_refs",
        "ref_n_unique_types",
        "ref_pubyear_min",
        "ref_pubyear_max",
        "ref_n_languages",
    ]
    for column in numeric_fill_columns:
        if column in model_ready.columns:
            model_ready[column] = pd.to_numeric(model_ready[column], errors="coerce")

    return model_ready


def make_quality_summary(master: pd.DataFrame, duplicate_expids: list[str]) -> dict:
    age_valid_mask = master["age_clean"].notna()
    return {
        "rows_master": int(len(master)),
        "duplicate_expids_in_exped": duplicate_expids,
        "n_duplicate_expids_in_exped": int(len(duplicate_expids)),
        "outcome_4_counts": master["outcome_4"].value_counts().to_dict(),
        "age_valid_count": int(age_valid_mask.sum()),
        "age_invalid_count": int(master["age_out_of_range_flag"].sum()),
        "age_missing_count": int(master["age_missing_flag"].sum()),
        "occupation_missing_count": int(master["occupation_missing_flag"].sum()),
        "occupation_group_counts": master["occupation_group"].value_counts().to_dict(),
        "top_missing_columns_master": (
            (master.isna().mean() * 100)
            .sort_values(ascending=False)
            .head(20)
            .round(2)
            .to_dict()
        ),
    }


def export_summaries(master: pd.DataFrame) -> None:
    age_issues = master.loc[
        master["age_out_of_range_flag"],
        [
            "expid",
            "peakid",
            "myear",
            "mseason",
            "sex",
            "status",
            "citizen_clean",
            "occupation_raw_clean",
            "age_raw",
        ],
    ].copy()
    age_issues.to_csv(TABLES_DIR / "age_inconsistencies.csv", index=False)

    occupation_group_summary = (
        master.groupby("occupation_group", dropna=False)
        .agg(
            n=("occupation_group", "size"),
            summit_rate=("msuccess", "mean"),
            death_rate=("death", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )
    occupation_group_summary.to_csv(TABLES_DIR / "occupation_group_summary.csv", index=False)

    occupation_raw_summary = (
        master.loc[master["occupation_raw_clean"].ne("unknown")]
        .groupby("occupation_raw_clean", dropna=False)
        .agg(
            n=("occupation_raw_clean", "size"),
            summit_rate=("msuccess", "mean"),
            death_rate=("death", "mean"),
        )
        .query("n >= 80")
        .sort_values(["summit_rate", "n"], ascending=[False, False])
        .reset_index()
    )
    occupation_raw_summary.to_csv(TABLES_DIR / "occupation_raw_summary.csv", index=False)

    region_summary = (
        master.groupby("peak_region", dropna=False)
        .agg(
            members=("peak_region", "size"),
            summit_rate=("msuccess", "mean"),
            death_rate=("death", "mean"),
        )
        .sort_values("members", ascending=False)
        .reset_index()
    )
    region_summary.to_csv(TABLES_DIR / "region_summary.csv", index=False)


def main() -> None:
    ensure_output_dirs()
    members, exped, peaks, refer = load_csvs()

    exped_resolved, duplicate_expids = resolve_exped_duplicates(exped)
    peaks_prefixed = peaks.rename(columns={column: f"peak_{column}" for column in peaks.columns if column != "peakid"})
    refer_summary = summarize_references(refer)

    master = members.merge(exped_resolved, on="expid", how="left")
    master = master.merge(peaks_prefixed, on="peakid", how="left")
    master = master.merge(refer_summary, on="expid", how="left")

    master = create_targets(master)
    master = create_age_features(master)
    master = add_clean_features(master)
    master = add_historical_rates(master)

    model_ready = build_model_ready(master)

    master.to_csv(DATA_PROCESSED_DIR / "himalaya_master_clean.csv", index=False)
    model_ready.to_csv(DATA_PROCESSED_DIR / "himalaya_model_ready.csv", index=False)
    export_summaries(master)

    summary = make_quality_summary(master, duplicate_expids)
    with (METRICS_DIR / "data_quality_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2)

    print("Pipeline complete.")
    print(f"Master rows: {len(master)}")
    print(f"Model-ready rows: {len(model_ready)}")
    print(f"Duplicate expids in exped resolved conservatively: {len(duplicate_expids)}")
    print(f"Age invalid rows exported: {int(master['age_out_of_range_flag'].sum())}")


if __name__ == "__main__":
    main()
