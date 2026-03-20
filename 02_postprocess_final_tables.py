import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from justification_config import DEFAULT_OUTPUT_ROOT, apply_overrides, load_config, resolve_output_dirs
from justification_utils import read_csv_if_exists, write_df, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess justification outputs: audit merge + final tables + notes.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--core-dir", type=str, default=None)
    parser.add_argument("--post-dir", type=str, default=None)
    parser.add_argument("--legacy-dir", type=str, default=None)
    parser.add_argument("--g2-audit-csv", type=str, default=None)
    return parser.parse_args()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def prepare_g2_outputs(core_dir: Path, post_dir: Path, audit_csv: Path) -> Dict[str, object]:
    info: Dict[str, object] = {
        "g2_audit_status": "missing",
        "g2_refined_plots_allowed": False,
    }

    terms = read_csv_if_exists(core_dir / "g2_topic_terms.csv")
    prev = read_csv_if_exists(core_dir / "g2_topic_prevalence.csv")
    if terms is None or terms.empty or prev is None or prev.empty:
        info["g2_audit_status"] = "not_available"
        return info

    base = terms[["lang", "topic_id"]].drop_duplicates().copy()
    if audit_csv.exists():
        audit = pd.read_csv(audit_csv)
        info["g2_audit_status"] = "manual_audited"
        info["g2_refined_plots_allowed"] = True
    else:
        audit = base.copy()
        audit["keep_topic"] = 1
        audit["noise_topic"] = 0
        audit["question_specific"] = 0
        audit["manual_label"] = ""
        audit["notes"] = "AUTO_TEMPLATE_NO_MANUAL_AUDIT_YET"
        write_df(audit, post_dir / "g2_topic_audit_template.csv")
        info["g2_audit_status"] = "auto_template_only"
        info["g2_refined_plots_allowed"] = False

    for col in ["keep_topic", "noise_topic", "question_specific"]:
        if col not in audit.columns:
            audit[col] = 0
        audit[col] = audit[col].fillna(0).astype(int)
    for col in ["manual_label", "notes"]:
        if col not in audit.columns:
            audit[col] = ""

    audit = audit[["lang", "topic_id", "keep_topic", "noise_topic", "question_specific", "manual_label", "notes"]].copy()

    terms_audited = terms.merge(audit, on=["lang", "topic_id"], how="left")
    terms_audited[["keep_topic", "noise_topic", "question_specific"]] = terms_audited[["keep_topic", "noise_topic", "question_specific"]].fillna(0).astype(int)
    terms_audited["manual_label"] = terms_audited["manual_label"].fillna("")
    terms_audited["notes"] = terms_audited["notes"].fillna("")
    write_df(terms_audited, post_dir / "g2_topic_terms_audited.csv")

    prev2 = prev.rename(columns={"topic": "topic_id"}).copy()
    prev_audited = prev2.merge(audit[["lang", "topic_id", "keep_topic", "noise_topic", "question_specific"]], on=["lang", "topic_id"], how="left")
    prev_audited[["keep_topic", "noise_topic", "question_specific"]] = prev_audited[["keep_topic", "noise_topic", "question_specific"]].fillna(0).astype(int)

    prev_valid = prev_audited[prev_audited["keep_topic"] == 1].copy()
    write_df(prev_valid, post_dir / "g2_topic_prevalence_valid.csv")

    valid_topics = terms_audited[terms_audited["keep_topic"] == 1][["lang", "topic_id", "top_terms", "manual_label", "notes"]].copy()
    write_df(valid_topics, post_dir / "g2_valid_topics.csv")

    lang_total = prev_audited.groupby("lang", as_index=False)["n"].sum().rename(columns={"n": "total_assignment_n"})
    lang_valid = prev_audited[prev_audited["keep_topic"] == 1].groupby("lang", as_index=False)["n"].sum().rename(columns={"n": "valid_assignment_n"})
    lang_noise = prev_audited[prev_audited["noise_topic"] == 1].groupby("lang", as_index=False)["n"].sum().rename(columns={"n": "noise_assignment_n"})
    summary = lang_total.merge(lang_valid, on="lang", how="left").merge(lang_noise, on="lang", how="left").fillna(0)
    summary["valid_coverage_rate"] = summary["valid_assignment_n"] / summary["total_assignment_n"].clip(lower=1)
    summary["noise_mass_rate"] = summary["noise_assignment_n"] / summary["total_assignment_n"].clip(lower=1)
    write_df(summary[["lang", "total_assignment_n", "valid_assignment_n", "valid_coverage_rate"]], post_dir / "g2_valid_topic_summary.csv")

    js_rows = []
    for (choice, model, lang), g in prev_valid.groupby(["choice", "model", "lang"]):
        gp = g[g["pressure"] == "pressure"]
        gn = g[g["pressure"] == "no_pressure"]
        if gp.empty or gn.empty:
            continue
        topics = sorted(set(gp["topic_id"]).union(set(gn["topic_id"])))
        p = np.array([gp.loc[gp["topic_id"] == t, "n"].sum() for t in topics], dtype=float)
        q = np.array([gn.loc[gn["topic_id"] == t, "n"].sum() for t in topics], dtype=float)
        if p.sum() <= 0 or q.sum() <= 0:
            continue
        js_rows.append(
            {
                "choice": choice,
                "model": model,
                "lang": lang,
                "topic_jsd_pressure_vs_no_pressure": js_divergence(p, q),
            }
        )
    write_df(pd.DataFrame(js_rows), post_dir / "g2_topic_pressure_jsd_valid.csv")
    return info


def resolve_g4_final(core_dir: Path, legacy_dir: Path, post_dir: Path) -> Dict[str, str]:
    strict_rank = read_csv_if_exists(core_dir / "g4_crosslingual_model_ranking_strict_1to1.csv")
    strict_pairs = read_csv_if_exists(core_dir / "g4_crosslingual_pairs_strict_1to1_enriched.csv")

    if strict_rank is not None and not strict_rank.empty:
        write_df(strict_rank, post_dir / "g4_final_model_ranking.csv")
        status = "strict_1to1"
    else:
        fallback = read_csv_if_exists(legacy_dir / "g4_crosslingual_model_ranking_many_to_many.csv")
        if fallback is not None and not fallback.empty:
            write_df(fallback, post_dir / "g4_final_model_ranking.csv")
            status = "legacy_many_to_many_fallback"
        else:
            write_df(pd.DataFrame(), post_dir / "g4_final_model_ranking.csv")
            status = "missing"

    if strict_pairs is not None and not strict_pairs.empty:
        write_df(strict_pairs, post_dir / "g4_crosslingual_pairs_final.csv")
    else:
        write_df(pd.DataFrame(), post_dir / "g4_crosslingual_pairs_final.csv")

    return {"g4_final_source": status}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config if args.config else None)
    cfg = apply_overrides(cfg, {"output_root": args.output_root, "g2_audit_csv": args.g2_audit_csv})
    if not cfg.output_root:
        cfg.output_root = str(DEFAULT_OUTPUT_ROOT)

    dirs = resolve_output_dirs(cfg)
    core_dir = Path(args.core_dir) if args.core_dir else dirs["core"]
    post_dir = Path(args.post_dir) if args.post_dir else dirs["postprocessed"]
    legacy_dir = Path(args.legacy_dir) if args.legacy_dir else dirs["legacy"]

    post_dir.mkdir(parents=True, exist_ok=True)

    audit_csv = Path(cfg.g2_audit_csv) if cfg.g2_audit_csv else (core_dir / "g2_topic_audit.csv")

    g2_info = prepare_g2_outputs(core_dir, post_dir, audit_csv)
    g4_info = resolve_g4_final(core_dir, legacy_dir, post_dir)

    metric_definitions = {
        "selection_rules": {
            "min_cell_n": cfg.min_cell_n,
            "min_model_cells": cfg.min_model_cells,
            "min_model_pairs": cfg.min_model_pairs,
            "g4_default_pairing": "strict_1to1",
        },
        "g2_audit_status": g2_info.get("g2_audit_status", "missing"),
        "g2_refined_plots_allowed": bool(g2_info.get("g2_refined_plots_allowed", False)),
        "g4_final_source": g4_info["g4_final_source"],
        "notes": [
            "G4 translation/reframing proximal flags are recomputed on strict 1-to-1 sample.",
            "Strict flags are not inherited from legacy many-to-many pairs.",
        ],
    }
    write_json(metric_definitions, post_dir / "metric_definitions.json")

    manifest = {
        "g2_topic_terms_audited": str(post_dir / "g2_topic_terms_audited.csv"),
        "g2_topic_prevalence_valid": str(post_dir / "g2_topic_prevalence_valid.csv"),
        "g2_valid_topic_summary": str(post_dir / "g2_valid_topic_summary.csv"),
        "g2_topic_pressure_jsd_valid": str(post_dir / "g2_topic_pressure_jsd_valid.csv"),
        "g4_final_model_ranking": str(post_dir / "g4_final_model_ranking.csv"),
        "g4_crosslingual_pairs_final": str(post_dir / "g4_crosslingual_pairs_final.csv"),
        "metric_definitions": str(post_dir / "metric_definitions.json"),
    }
    write_json(manifest, post_dir / "postprocess_manifest.json")

    print(f"done {post_dir}")


if __name__ == "__main__":
    main()


