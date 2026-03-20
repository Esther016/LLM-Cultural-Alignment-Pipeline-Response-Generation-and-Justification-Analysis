import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from justification_config import DEFAULT_OUTPUT_ROOT, apply_overrides, load_config, resolve_output_dirs
from justification_utils import read_csv_if_exists, require_columns, write_df, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CSS-oriented group summary tables from final justification outputs.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--core-dir", type=str, default=None)
    parser.add_argument("--post-dir", type=str, default=None)
    parser.add_argument("--model-metadata", type=str, default="model_metadata.csv")
    parser.add_argument("--question-metadata", type=str, default="AllQuestions.xlsx")
    parser.add_argument("--g0-used", type=str, default=None)
    return parser.parse_args()


def normalize_text(s: object) -> str:
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    # Normalize common mojibake punctuation before whitespace-stripping match.
    for bad, good in {
        "锛?": ",",
        "銆?": ".",
        "鈥?": '"',
    }.items():
        t = t.replace(bad, good)
    t = re.sub(r"\s+", "", t)
    return t


def safe_read_excel_all(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    blocks = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df["sheet_name"] = sheet
        blocks.append(df)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def build_model_metadata(model_meta_path: Path, post_dir: Path, warnings: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    mm = pd.read_csv(model_meta_path)
    require_columns(mm, ["model"], "model_metadata.csv")

    expected = ["provider", "origin_group", "family_group", "open_closed", "reasoning_profile", "notes"]
    for col in expected:
        if col not in mm.columns:
            mm[col] = pd.NA
            warnings.append(f"model_metadata missing expected column: {col}")

    mm["model"] = mm["model"].astype(str).str.strip()
    dup = mm["model"].duplicated(keep=False)
    dup_rows = mm[dup].copy()
    if not dup_rows.empty:
        warnings.append(f"model_metadata duplicated keys detected; keeping first occurrence for {dup_rows['model'].nunique()} models")
        mm = mm.drop_duplicates(subset=["model"], keep="first").copy()

    clean_cols = ["model"] + expected
    mm_clean = mm[clean_cols].copy()
    write_df(mm_clean, post_dir / "model_metadata_clean.csv")

    notes = {
        "model_metadata_columns": clean_cols,
        "duplicate_model_count": int(dup_rows["model"].nunique()) if not dup_rows.empty else 0,
        "records": int(len(mm_clean)),
    }
    return mm_clean, notes


def build_question_metadata(question_meta_path: Path, g0_path: Path, post_dir: Path, warnings: List[str]) -> Tuple[pd.DataFrame, Dict[str, object], bool]:
    aq_raw = safe_read_excel_all(question_meta_path)
    if aq_raw.empty:
        warnings.append("AllQuestions.xlsx contains no rows")
        out = pd.DataFrame(columns=["question_id", "issue_domain", "ideological_salience", "state_individual_relevance", "domestic_vs_global"])
        write_df(out, post_dir / "question_metadata_clean.csv")
        return out, {"column_mapping": []}, False

    colmap = []
    cols = list(aq_raw.columns)
    source_col = find_column(cols, ["source"])
    type_col = find_column(cols, ["type", "issue_domain", "issue"])
    qid_col = find_column(cols, ["question_id", "qid", "questionid", "id"])
    q_en_col = find_column(cols, ["question", "question_en", "question_text", "text"])
    q_zh_col = find_column(cols, ["question_cn", "question_zh", "question_chinese"])
    sal_col = find_column(cols, ["ideological_salience", "salience", "ideology_salience"])
    si_col = find_column(cols, ["state_individual_relevance", "state_vs_individual", "state_individual"])
    dg_col = find_column(cols, ["domestic_vs_global", "domestic_global", "global_vs_domestic"])

    if source_col is None:
        warnings.append("AllQuestions.xlsx source column not found; source-based matching disabled")
    if type_col is None:
        warnings.append("AllQuestions.xlsx type/issue column not found; issue_domain will fallback to g0 type only")
    if q_en_col is None and q_zh_col is None and qid_col is None:
        warnings.append("AllQuestions.xlsx has no question text and no question_id column; question metadata mapping limited")

    colmap.append({"original_column": source_col, "standardized_column": "source"})
    colmap.append({"original_column": type_col, "standardized_column": "issue_domain"})
    colmap.append({"original_column": qid_col, "standardized_column": "question_id"})
    colmap.append({"original_column": q_en_col, "standardized_column": "question_en"})
    colmap.append({"original_column": q_zh_col, "standardized_column": "question_zh"})
    colmap.append({"original_column": sal_col, "standardized_column": "ideological_salience"})
    colmap.append({"original_column": si_col, "standardized_column": "state_individual_relevance"})
    colmap.append({"original_column": dg_col, "standardized_column": "domestic_vs_global"})

    aq = pd.DataFrame()
    aq["source"] = aq_raw[source_col].astype(str) if source_col else ""
    aq["issue_domain"] = aq_raw[type_col].astype(str) if type_col else ""
    aq["question_id_raw"] = aq_raw[qid_col].astype(str) if qid_col else ""
    aq["question_en"] = aq_raw[q_en_col].astype(str) if q_en_col else ""
    aq["question_zh"] = aq_raw[q_zh_col].astype(str) if q_zh_col else ""
    aq["ideological_salience"] = aq_raw[sal_col] if sal_col else pd.NA
    aq["state_individual_relevance"] = aq_raw[si_col] if si_col else pd.NA
    aq["domestic_vs_global"] = aq_raw[dg_col] if dg_col else pd.NA
    aq["source_n"] = aq["source"].map(normalize_text)
    aq["issue_domain_n"] = aq["issue_domain"].map(normalize_text)
    aq["question_en_n"] = aq["question_en"].map(normalize_text)
    aq["question_zh_n"] = aq["question_zh"].map(normalize_text)
    aq["row_id"] = np.arange(len(aq))

    g0 = pd.read_csv(g0_path, usecols=["question_id", "question", "source", "type", "lang"]).drop_duplicates().copy()
    g0["question_id"] = g0["question_id"].astype(str)
    g0["question_n"] = g0["question"].map(normalize_text)
    g0["source_n"] = g0["source"].map(normalize_text)
    g0["type_n"] = g0["type"].map(normalize_text)

    resolved = []
    for _, r in g0.iterrows():
        qn = r["question_n"]
        sn = r["source_n"]
        tn = r["type_n"]
        lang = r["lang"]
        candidates = pd.DataFrame()

        if lang == "en" and "question_en_n" in aq.columns:
            candidates = aq[(aq["question_en_n"] == qn)]
        elif lang == "zh" and "question_zh_n" in aq.columns:
            candidates = aq[(aq["question_zh_n"] == qn)]

        match_level = "lang_text"
        if candidates.empty:
            candidates = aq[(aq["question_en_n"] == qn) | (aq["question_zh_n"] == qn)]
            match_level = "any_text"

        if not candidates.empty and source_col:
            s2 = candidates[candidates["source_n"] == sn]
            if not s2.empty:
                candidates = s2
                match_level += "+source"

        if not candidates.empty and type_col:
            t2 = candidates[candidates["issue_domain_n"] == tn]
            if not t2.empty:
                candidates = t2
                match_level += "+type"

        if candidates.empty:
            resolved.append(
                {
                    "question_id": r["question_id"],
                    "lang": lang,
                    "issue_domain": r["type"] if str(r["type"]).strip() else "unknown",
                    "ideological_salience": pd.NA,
                    "state_individual_relevance": pd.NA,
                    "domestic_vs_global": pd.NA,
                    "match_status": "unmatched_fallback_to_g0_type",
                    "matched_rows": 0,
                }
            )
            continue

        pick = candidates.iloc[0]
        resolved.append(
            {
                "question_id": r["question_id"],
                "lang": lang,
                "issue_domain": pick["issue_domain"] if str(pick["issue_domain"]).strip() else (r["type"] if str(r["type"]).strip() else "unknown"),
                "ideological_salience": pick["ideological_salience"],
                "state_individual_relevance": pick["state_individual_relevance"],
                "domestic_vs_global": pick["domestic_vs_global"],
                "match_status": "matched_" + match_level if len(candidates) == 1 else "matched_ambiguous_" + match_level,
                "matched_rows": int(len(candidates)),
            }
        )

    qres = pd.DataFrame(resolved)
    if qres.empty:
        qclean = pd.DataFrame(columns=["question_id", "issue_domain", "ideological_salience", "state_individual_relevance", "domestic_vs_global"])
    else:
        # consolidate en/zh rows to question-level; keep modal issue_domain and first non-null optional fields
        def first_nonnull(s: pd.Series) -> object:
            for v in s:
                if pd.notna(v) and str(v).strip() != "":
                    return v
            return pd.NA

        qclean = (
            qres.groupby("question_id", as_index=False)
            .agg(
                issue_domain=("issue_domain", lambda s: s.dropna().astype(str).mode().iat[0] if not s.dropna().empty else "unknown"),
                ideological_salience=("ideological_salience", first_nonnull),
                state_individual_relevance=("state_individual_relevance", first_nonnull),
                domestic_vs_global=("domestic_vs_global", first_nonnull),
            )
        )
        qclean["issue_domain"] = qclean["issue_domain"].fillna("unknown").replace("", "unknown")

    write_df(qclean, post_dir / "question_metadata_clean.csv")

    matched = int((qres["match_status"].str.startswith("matched")).sum()) if not qres.empty else 0
    ambiguous = int((qres["match_status"].str.startswith("matched_ambiguous")).sum()) if not qres.empty else 0
    unmatched = int((qres["match_status"] == "unmatched_fallback_to_g0_type").sum()) if not qres.empty else 0
    notes = {
        "column_mapping": colmap,
        "question_id_detection": {
            "question_id_column_in_xlsx": qid_col,
            "selection_note": "AllQuestions.xlsx had no reliable question_id column; question_id derived from outputs/core/g0_standardized_justification_used.csv and matched by normalized question text/source/type."
            if qid_col is None
            else "question_id column found in AllQuestions.xlsx and used directly where available.",
        },
        "match_stats": {
            "row_level_records": int(len(qres)),
            "matched_rows": matched,
            "ambiguous_rows": ambiguous,
            "unmatched_rows": unmatched,
            "matched_rate": float(matched / max(len(qres), 1)),
        },
    }

    optional_available = any([sal_col is not None, si_col is not None, dg_col is not None])
    if not optional_available:
        warnings.append("Optional question metadata columns not found in AllQuestions.xlsx: ideological_salience/state_individual_relevance/domestic_vs_global")
    return qclean, notes, optional_available


def add_model_meta(df: pd.DataFrame, mm: pd.DataFrame, table_name: str, warnings: List[str]) -> pd.DataFrame:
    out = df.merge(mm, on="model", how="left")
    missing = out["origin_group"].isna().sum() if "origin_group" in out.columns else 0
    if missing > 0:
        warnings.append(f"{table_name}: {missing} rows have missing model metadata after merge")
    return out


def summary_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "count": int(series.size),
        "mean": float(series.mean()) if len(series) else np.nan,
        "median": float(series.median()) if len(series) else np.nan,
        "std": float(series.std(ddof=0)) if len(series) else np.nan,
    }


def build_g1(core_dir: Path, css_dir: Path, mm: pd.DataFrame, warnings: List[str]) -> List[str]:
    created = []
    g1 = read_csv_if_exists(core_dir / "g1_pressure_embedding_drift.csv")
    if g1 is None or g1.empty:
        warnings.append("G1 missing; skipped CSS G1 summaries")
        return created
    require_columns(g1, ["model", "lang", "drift_distance"], "g1_pressure_embedding_drift.csv")
    g1 = add_model_meta(g1, mm, "g1_pressure_embedding_drift.csv", warnings)

    by_lang = g1.groupby("lang")["drift_distance"].agg(["count", "mean", "median", "std"]).reset_index()
    write_df(by_lang, css_dir / "g1_by_lang_summary.csv")
    created.append(str(css_dir / "g1_by_lang_summary.csv"))

    by_origin = g1.groupby("origin_group")["drift_distance"].agg(["count", "mean", "median", "std"]).reset_index()
    write_df(by_origin, css_dir / "g1_by_origin_group_summary.csv")
    created.append(str(css_dir / "g1_by_origin_group_summary.csv"))

    by_lang_origin = (
        g1.groupby(["lang", "origin_group"])["drift_distance"].agg(["count", "mean", "median", "std"]).reset_index()
    )
    write_df(by_lang_origin, css_dir / "g1_by_lang_origin_group_summary.csv")
    created.append(str(css_dir / "g1_by_lang_origin_group_summary.csv"))

    empty_issue = pd.DataFrame(columns=["issue_domain", "count", "mean", "median", "std"])
    write_df(empty_issue, css_dir / "g1_by_issue_domain_summary.csv")
    created.append(str(css_dir / "g1_by_issue_domain_summary.csv"))
    empty_issue_lang = pd.DataFrame(columns=["issue_domain", "lang", "count", "mean", "median", "std"])
    write_df(empty_issue_lang, css_dir / "g1_by_issue_domain_lang_summary.csv")
    created.append(str(css_dir / "g1_by_issue_domain_lang_summary.csv"))
    warnings.append("G1 issue-domain summary emitted as empty placeholder: g1_pressure_embedding_drift.csv has no reliable question_id key for merge")
    return created


def build_g3(core_dir: Path, css_dir: Path, mm: pd.DataFrame, warnings: List[str]) -> List[str]:
    created = []
    g3 = read_csv_if_exists(core_dir / "g3_axis_pressure_drift.csv")
    if g3 is None or g3.empty:
        warnings.append("G3 missing; skipped CSS G3 summaries")
        return created
    require_columns(
        g3,
        ["axis", "model", "lang", "axis_shift_pressure_minus_no_pressure", "effect_size"],
        "g3_axis_pressure_drift.csv",
    )
    g3 = add_model_meta(g3, mm, "g3_axis_pressure_drift.csv", warnings)
    g3["abs_shift"] = g3["axis_shift_pressure_minus_no_pressure"].abs()
    g3["abs_effect_size"] = g3["effect_size"].abs()

    by_axis_lang = (
        g3.groupby(["axis", "lang"], as_index=False)
        .agg(
            mean_shift=("axis_shift_pressure_minus_no_pressure", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            mean_effect_size=("effect_size", "mean"),
            mean_abs_effect_size=("abs_effect_size", "mean"),
            count=("axis_shift_pressure_minus_no_pressure", "size"),
        )
    )
    write_df(by_axis_lang, css_dir / "g3_by_axis_lang_summary.csv")
    created.append(str(css_dir / "g3_by_axis_lang_summary.csv"))

    by_axis_origin = (
        g3.groupby(["axis", "origin_group"], as_index=False)
        .agg(
            mean_shift=("axis_shift_pressure_minus_no_pressure", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            mean_effect_size=("effect_size", "mean"),
            mean_abs_effect_size=("abs_effect_size", "mean"),
            count=("axis_shift_pressure_minus_no_pressure", "size"),
        )
    )
    write_df(by_axis_origin, css_dir / "g3_by_axis_origin_group_summary.csv")
    created.append(str(css_dir / "g3_by_axis_origin_group_summary.csv"))

    by_axis_lang_origin = (
        g3.groupby(["axis", "lang", "origin_group"], as_index=False)
        .agg(
            mean_shift=("axis_shift_pressure_minus_no_pressure", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            mean_effect_size=("effect_size", "mean"),
            mean_abs_effect_size=("abs_effect_size", "mean"),
            count=("axis_shift_pressure_minus_no_pressure", "size"),
        )
    )
    write_df(by_axis_lang_origin, css_dir / "g3_by_axis_lang_origin_group_summary.csv")
    created.append(str(css_dir / "g3_by_axis_lang_origin_group_summary.csv"))

    empty_issue = pd.DataFrame(columns=["axis", "issue_domain", "mean_shift", "mean_abs_shift", "mean_effect_size", "mean_abs_effect_size", "count"])
    write_df(empty_issue, css_dir / "g3_by_axis_issue_domain_summary.csv")
    created.append(str(css_dir / "g3_by_axis_issue_domain_summary.csv"))
    empty_issue_lang = pd.DataFrame(columns=["axis", "issue_domain", "lang", "mean_shift", "mean_abs_shift", "mean_effect_size", "mean_abs_effect_size", "count"])
    write_df(empty_issue_lang, css_dir / "g3_by_axis_issue_domain_lang_summary.csv")
    created.append(str(css_dir / "g3_by_axis_issue_domain_lang_summary.csv"))
    warnings.append("G3 issue-domain summary emitted as empty placeholder: g3_axis_pressure_drift.csv has no reliable question_id key for merge")
    return created


def build_g5(core_dir: Path, css_dir: Path, mm: pd.DataFrame, warnings: List[str]) -> List[str]:
    created = []
    g5 = read_csv_if_exists(core_dir / "g5_style_pressure_drift.csv")
    if g5 is None or g5.empty:
        warnings.append("G5 missing; skipped CSS G5 summaries")
        return created
    require_columns(g5, ["model", "lang", "metric", "shift_pressure_minus_no_pressure"], "g5_style_pressure_drift.csv")
    g5 = add_model_meta(g5, mm, "g5_style_pressure_drift.csv", warnings)
    g5["abs_shift"] = g5["shift_pressure_minus_no_pressure"].abs()

    by_metric_lang = (
        g5.groupby(["metric", "lang"], as_index=False)
        .agg(
            mean_shift=("shift_pressure_minus_no_pressure", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            count=("shift_pressure_minus_no_pressure", "size"),
        )
    )
    write_df(by_metric_lang, css_dir / "g5_by_metric_lang_summary.csv")
    created.append(str(css_dir / "g5_by_metric_lang_summary.csv"))

    by_metric_origin = (
        g5.groupby(["metric", "origin_group"], as_index=False)
        .agg(
            mean_shift=("shift_pressure_minus_no_pressure", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            count=("shift_pressure_minus_no_pressure", "size"),
        )
    )
    write_df(by_metric_origin, css_dir / "g5_by_metric_origin_group_summary.csv")
    created.append(str(css_dir / "g5_by_metric_origin_group_summary.csv"))

    by_metric_lang_origin = (
        g5.groupby(["metric", "lang", "origin_group"], as_index=False)
        .agg(
            mean_shift=("shift_pressure_minus_no_pressure", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            count=("shift_pressure_minus_no_pressure", "size"),
        )
    )
    write_df(by_metric_lang_origin, css_dir / "g5_by_metric_lang_origin_group_summary.csv")
    created.append(str(css_dir / "g5_by_metric_lang_origin_group_summary.csv"))

    empty_issue = pd.DataFrame(columns=["metric", "issue_domain", "mean_shift", "mean_abs_shift", "count"])
    write_df(empty_issue, css_dir / "g5_by_metric_issue_domain_summary.csv")
    created.append(str(css_dir / "g5_by_metric_issue_domain_summary.csv"))
    empty_issue_lang = pd.DataFrame(columns=["metric", "issue_domain", "lang", "mean_shift", "mean_abs_shift", "count"])
    write_df(empty_issue_lang, css_dir / "g5_by_metric_issue_domain_lang_summary.csv")
    created.append(str(css_dir / "g5_by_metric_issue_domain_lang_summary.csv"))
    warnings.append("G5 issue-domain summary emitted as empty placeholder: g5_style_pressure_drift.csv has no reliable question_id key for merge")
    return created


def build_g4(post_dir: Path, css_dir: Path, mm: pd.DataFrame, qmeta: pd.DataFrame, warnings: List[str]) -> List[str]:
    created = []
    rank = read_csv_if_exists(post_dir / "g4_final_model_ranking.csv")
    pairs = read_csv_if_exists(post_dir / "g4_crosslingual_pairs_final.csv")
    if rank is None or rank.empty:
        warnings.append("G4 ranking missing; skipped g4 model-level CSS summaries")
    else:
        rank_m = add_model_meta(rank, mm, "g4_final_model_ranking.csv", warnings)
        write_df(rank_m, css_dir / "g4_model_metadata_joined.csv")
        created.append(str(css_dir / "g4_model_metadata_joined.csv"))

        need_cols = ["mean_axis_shift", "mean_style_shift", "translation_rate", "reframing_rate", "n_pairs"]
        missing = [c for c in need_cols if c not in rank_m.columns]
        if missing:
            warnings.append(f"G4 origin summary skipped; missing columns in ranking: {missing}")
        else:
            tmp = rank_m.copy()
            tmp["n_pairs"] = pd.to_numeric(tmp["n_pairs"], errors="coerce").fillna(0.0)

            def wavg(g: pd.DataFrame, col: str) -> float:
                w = g["n_pairs"].to_numpy(dtype=float)
                x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                if np.nansum(w) <= 0:
                    return float(np.nanmean(x))
                return float(np.nansum(x * w) / np.nansum(w))

            rows = []
            for og, g in tmp.groupby("origin_group", dropna=False):
                rows.append(
                    {
                        "origin_group": og,
                        "mean_axis_shift": wavg(g, "mean_axis_shift"),
                        "mean_style_shift": wavg(g, "mean_style_shift"),
                        "translation_rate": wavg(g, "translation_rate"),
                        "reframing_rate": wavg(g, "reframing_rate"),
                        "n_pairs": int(g["n_pairs"].sum()),
                        "n_models": int(len(g)),
                    }
                )
            out = pd.DataFrame(rows)
            write_df(out, css_dir / "g4_by_origin_group_summary.csv")
            created.append(str(css_dir / "g4_by_origin_group_summary.csv"))

    if pairs is None or pairs.empty:
        warnings.append("G4 pairs missing; skipped g4 issue-domain summaries")
        return created

    if "question_id" not in pairs.columns:
        warnings.append("G4 pairs has no question_id; skipped issue-domain summaries")
        return created

    pairs_q = pairs.merge(qmeta[["question_id", "issue_domain"]], on="question_id", how="left")
    pairs_q["issue_domain"] = pairs_q["issue_domain"].fillna("unknown")

    need_pair_cols = ["axis_shift_l2", "style_shift_l2", "translation_proximal_flag", "reframing_proximal_flag"]
    missing = [c for c in need_pair_cols if c not in pairs_q.columns]
    if missing:
        warnings.append(f"G4 issue-domain summary skipped; missing pair columns: {missing}")
        return created

    by_issue = (
        pairs_q.groupby("issue_domain", as_index=False)
        .agg(
            mean_axis_shift=("axis_shift_l2", "mean"),
            mean_style_shift=("style_shift_l2", "mean"),
            translation_rate=("translation_proximal_flag", "mean"),
            reframing_rate=("reframing_proximal_flag", "mean"),
            n_pairs=("question_id", "size"),
        )
    )
    by_issue["lang"] = "crosslingual"
    by_lang_issue = by_issue[["lang", "issue_domain", "mean_axis_shift", "mean_style_shift", "translation_rate", "reframing_rate", "n_pairs"]]
    write_df(by_lang_issue, css_dir / "g4_by_lang_issue_domain_summary.csv")
    created.append(str(css_dir / "g4_by_lang_issue_domain_summary.csv"))
    return created


def build_notes(css_dir: Path, warnings: List[str], qmeta: pd.DataFrame, q_optional_available: bool) -> Tuple[str, Dict[str, object]]:
    notes_path = css_dir.parent / "css_analysis_notes.md"
    lines = []
    lines.append("# CSS Analysis Notes")
    lines.append("")
    lines.append("## Answered CSS Questions")
    lines.append("- Language effects: summarized via G1/G3/G5 `by_lang` and `lang x origin_group` tables.")
    lines.append("- Model-origin effects: summarized via G1/G3/G4/G5 `by_origin_group` tables.")
    lines.append("- Pressure interaction effects: interpreted from pressure-induced shift metrics stratified by language/model-origin (interaction-style descriptive comparisons).")
    lines.append("- Issue-domain effects: available only where `question_id` is present for reliable merge (currently G4 pair-level).")
    lines.append("")
    lines.append("## Metadata Usage")
    lines.append("- Model metadata used: `model`, `provider`, `origin_group`, `family_group`, `open_closed`, `reasoning_profile`, `notes`.")
    lines.append("- Question metadata target fields: `question_id`, `issue_domain`, optional `ideological_salience`, `state_individual_relevance`, `domestic_vs_global`.")
    lines.append("- `AllQuestions.xlsx` had no reliable `question_id` column; `question_id` was anchored from `g0_standardized_justification_used.csv` and matched using normalized question text/source/type.")
    if not q_optional_available:
        lines.append("- Optional question metadata columns were not present and are left as unknown.")
    lines.append("")
    lines.append("## Boundary Conditions")
    lines.append("- G2 topic layer remains unaudited exploratory and is excluded from main conclusions.")
    lines.append("- G3 is an anchor-based framing measure, not a gold-standard ideology score.")
    lines.append("- G4 translation-proximal / reframing-proximal are operational labels.")
    lines.append("- If `embedding_available = false`, G4 should be interpreted as axis/style-based cross-lingual consistency only.")
    lines.append("- Group-level differences are descriptive/associational and do not establish causality.")
    lines.append("")
    lines.append("## Uncertainties / Manual Checks")
    lines.append("- `AllQuestions.xlsx` question-text matching is partial and can be ambiguous for some rows.")
    lines.append("- G1/G3/G5 issue-domain breakdown is not generated because final core tables do not carry reliable `question_id` keys.")
    if warnings:
        lines.append("- Runtime warnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    notes_text = "\n".join(lines) + "\n"
    notes_path.write_text(notes_text, encoding="utf-8")

    q_missing = {
        col: float(qmeta[col].isna().mean()) if col in qmeta.columns and len(qmeta) > 0 else 1.0
        for col in ["issue_domain", "ideological_salience", "state_individual_relevance", "domestic_vs_global"]
    }
    manifest_extra = {"question_metadata_missing_rate": q_missing}
    return str(notes_path), manifest_extra


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config if args.config else None)
    cfg = apply_overrides(cfg, {"output_root": args.output_root})
    if not cfg.output_root:
        cfg.output_root = str(DEFAULT_OUTPUT_ROOT)

    dirs = resolve_output_dirs(cfg)
    core_dir = Path(args.core_dir) if args.core_dir else dirs["core"]
    post_dir = Path(args.post_dir) if args.post_dir else dirs["postprocessed"]
    css_dir = post_dir / "css"
    css_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    created_files: List[str] = []

    model_meta_path = Path(args.model_metadata)
    question_meta_path = Path(args.question_metadata)
    g0_path = Path(args.g0_used) if args.g0_used else (core_dir / "g0_standardized_justification_used.csv")

    if not model_meta_path.exists():
        raise FileNotFoundError(f"model metadata file not found: {model_meta_path}")
    if not question_meta_path.exists():
        raise FileNotFoundError(f"question metadata file not found: {question_meta_path}")
    if not g0_path.exists():
        raise FileNotFoundError(f"g0 file not found for question_id matching: {g0_path}")

    mm, mm_notes = build_model_metadata(model_meta_path, post_dir, warnings)
    qmeta, qm_notes, q_optional_available = build_question_metadata(question_meta_path, g0_path, post_dir, warnings)

    created_files.extend(build_g1(core_dir, css_dir, mm, warnings))
    created_files.extend(build_g3(core_dir, css_dir, mm, warnings))
    created_files.extend(build_g5(core_dir, css_dir, mm, warnings))
    created_files.extend(build_g4(post_dir, css_dir, mm, qmeta, warnings))

    merge_notes = {
        "model_metadata": mm_notes,
        "question_metadata": qm_notes,
        "warnings": warnings,
    }
    write_json(merge_notes, post_dir / "metadata_merge_notes.json")
    created_files.append(str(post_dir / "model_metadata_clean.csv"))
    created_files.append(str(post_dir / "question_metadata_clean.csv"))
    created_files.append(str(post_dir / "metadata_merge_notes.json"))

    notes_path, manifest_extra = build_notes(css_dir, warnings, qmeta, q_optional_available)
    created_files.append(notes_path)

    manifest = {
        "inputs": {
            "model_metadata": str(model_meta_path),
            "question_metadata": str(question_meta_path),
            "core_dir": str(core_dir),
            "post_dir": str(post_dir),
            "g0_for_question_mapping": str(g0_path),
        },
        "outputs": sorted(created_files),
        "used_metadata_columns": {
            "model_metadata": ["model", "provider", "origin_group", "family_group", "open_closed", "reasoning_profile", "notes"],
            "question_metadata": ["question_id", "issue_domain", "ideological_salience", "state_individual_relevance", "domestic_vs_global"],
        },
        "warnings": warnings,
    }
    manifest.update(manifest_extra)
    write_json(manifest, post_dir / "css_analysis_manifest.json")

    print(f"done {css_dir}")


if __name__ == "__main__":
    main()


