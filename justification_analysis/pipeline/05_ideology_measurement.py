from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# EDIT THESE: draft ideology anchors, expected to be revised by the professor.
ANCHOR_SENTENCES = {
    "gov_role": {
        "pos": [
            "Government should actively regulate markets and provide public services to ensure social welfare.",
            "The state bears responsibility for reducing inequality through redistribution and social programs.",
            "Strong government oversight is necessary to protect citizens from market failures.",
        ],
        "neg": [
            "Free markets and individual enterprise should operate with minimal state interference.",
            "Personal responsibility and limited government lead to better outcomes than central planning.",
            "Excessive government regulation stifles innovation and individual freedom.",
        ],
    },
    "liberty_security": {
        "pos": [
            "National security and public safety justify certain restrictions on individual freedoms.",
            "Governments must prioritize social stability and order, even at the cost of some civil liberties.",
            "Surveillance and security measures are necessary to protect society from threats.",
        ],
        "neg": [
            "Individual rights and civil liberties must be protected above all else.",
            "Freedom of speech and privacy are fundamental rights that should not be compromised for security.",
            "Government surveillance and restrictions on personal freedom are inherently dangerous.",
        ],
    },
    "equality_merit": {
        "pos": [
            "Society should ensure equal outcomes through redistribution and affirmative policies.",
            "Reducing inequality is more important than rewarding individual achievement.",
            "Social justice requires systemic interventions to level the playing field.",
        ],
        "neg": [
            "Outcomes should be determined by individual merit, effort, and competition.",
            "Equality of opportunity, not equality of outcome, is what matters.",
            "Redistribution undermines incentives and harms economic efficiency.",
        ],
    },
    "censorship_acceptance": {
        "pos": [
            "Governments may legitimately restrict speech to maintain social harmony and prevent harm.",
            "Content moderation and platform regulation are necessary to combat misinformation.",
            "Some limits on expression are acceptable to protect public order and social cohesion.",
        ],
        "neg": [
            "Freedom of expression must be protected even for controversial or offensive speech.",
            "Censorship by governments or platforms is a threat to open democratic debate.",
            "People should be free to express any opinion without fear of punishment or suppression.",
        ],
    },
}

# EDIT THESE: keyword fallback only for models missing in metadata.
CHINESE_ORIGIN_KEYWORDS = [
    "qwen",
    "chatglm",
    "glm",
    "baichuan",
    "yi-",
    "yi_",
    "deepseek",
    "internlm",
    "moss",
    "tigerbot",
    "aquila",
    "xverse",
    "skywork",
    "minimax",
    "moonshot",
    "zhipu",
    "ernie",
    "spark",
    "doubao",
    "hunyuan",
    "sensenova",
    "abab",
    "chinese",
    "cn",
]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "core" / "g0_standardized_justification_all.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ideology"
DEFAULT_MODEL_METADATA = PROJECT_ROOT / "model_metadata.csv"

REQUIRED_COLUMNS = [
    "response_instance_id",
    "question_id",
    "model",
    "model_raw",
    "choice",
    "lang",
    "pressure",
    "justification",
    "source",
    "type",
    "question",
]

AXIS_COLUMN_MAP = {
    "gov_role": "ideology_gov_role",
    "liberty_security": "ideology_liberty_security",
    "equality_merit": "ideology_equality_merit",
    "censorship_acceptance": "ideology_censorship_acceptance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ideology dimensions from cleaned justification CSV using multilingual sentence embeddings."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", type=str, default="sentence-transformers/LaBSE")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_input_table(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    require_columns(df, REQUIRED_COLUMNS, input_path.name)
    return df


def load_model_metadata_if_available(model_metadata_path: Path) -> Optional[pd.DataFrame]:
    if not model_metadata_path.exists():
        return None
    df = pd.read_csv(model_metadata_path)
    require_columns(df, ["model", "origin_group"], model_metadata_path.name)
    out = df[["model", "origin_group"]].copy()
    out["model"] = out["model"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["model"], keep="first")
    return out


def standardize_origin_group(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"chinese", "chinese-origin"}:
        return "Chinese"
    if text in {"western", "western-origin"}:
        return "Western"
    return None


def classify_origin(model_name: str) -> str:
    lower = model_name.lower()
    for keyword in CHINESE_ORIGIN_KEYWORDS:
        if keyword in lower:
            return "Chinese"
    return "Western"


def resolve_origin_group(df: pd.DataFrame, model_metadata_path: Path) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = df.copy()
    info: Dict[str, object] = {
        "input_origin_group_used": False,
        "metadata_join_used": False,
        "keyword_fallback_rows": 0,
        "unresolved_after_standardization": 0,
        "model_metadata_path": str(model_metadata_path),
    }

    if "origin_group" in out.columns:
        out["origin_group"] = out["origin_group"].map(standardize_origin_group)
        info["input_origin_group_used"] = True
    else:
        out["origin_group"] = None

    metadata = load_model_metadata_if_available(model_metadata_path)
    if metadata is not None:
        metadata = metadata.copy()
        metadata["origin_group"] = metadata["origin_group"].map(standardize_origin_group)
        out = out.merge(metadata, on="model", how="left", suffixes=("", "_meta"))
        fill_mask = out["origin_group"].isna() & out["origin_group_meta"].notna()
        out.loc[fill_mask, "origin_group"] = out.loc[fill_mask, "origin_group_meta"]
        out = out.drop(columns=["origin_group_meta"])
        info["metadata_join_used"] = True

    unresolved_mask = out["origin_group"].isna()
    info["unresolved_after_standardization"] = int(unresolved_mask.sum())
    if unresolved_mask.any():
        out.loc[unresolved_mask, "origin_group"] = (
            out.loc[unresolved_mask, "model"].astype(str).map(classify_origin)
        )
        info["keyword_fallback_rows"] = int(unresolved_mask.sum())

    return out, info


def normalize_centroid(vectors: np.ndarray) -> np.ndarray:
    centroid = vectors.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return centroid / norm


def load_embedding_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def encode_justifications(
    texts: Sequence[str],
    model,
    batch_size: int,
) -> tuple[np.ndarray, int, float]:
    start = time.perf_counter()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - start
    return embeddings, int(model.get_sentence_embedding_dimension()), elapsed


def encode_anchor_centroids(
    model,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    centroids: Dict[str, Dict[str, np.ndarray]] = {}
    for axis_name, anchors in ANCHOR_SENTENCES.items():
        print(f"[anchor] Computing scores for axis: {axis_name}")
        pos_embeddings = model.encode(
            anchors["pos"],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        neg_embeddings = model.encode(
            anchors["neg"],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        centroids[axis_name] = {
            "pos": normalize_centroid(pos_embeddings),
            "neg": normalize_centroid(neg_embeddings),
        }
    return centroids


def compute_axis_scores(df: pd.DataFrame, embeddings: np.ndarray, centroids: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    out = df.copy()
    for axis_name, column_name in AXIS_COLUMN_MAP.items():
        pos_centroid = centroids[axis_name]["pos"]
        neg_centroid = centroids[axis_name]["neg"]
        sim_pos = (embeddings @ pos_centroid.T).ravel()
        sim_neg = (embeddings @ neg_centroid.T).ravel()
        out[column_name] = sim_pos - sim_neg
    return out


def safe_z(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(values.mean())) / std


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_components = []
    for column_name in AXIS_COLUMN_MAP.values():
        z_components.append(safe_z(out[column_name]))
    out["ideology_overall"] = pd.concat(z_components, axis=1).mean(axis=1)
    return out


def get_measure_columns() -> Dict[str, str]:
    measures = dict(AXIS_COLUMN_MAP)
    measures["overall"] = "ideology_overall"
    return measures


def build_group_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    blocks = []
    for axis_name, column_name in get_measure_columns().items():
        grouped = (
            df.groupby(group_cols, dropna=False)[column_name]
            .agg(mean="mean", std="std", median="median", n="count")
            .reset_index()
        )
        grouped.insert(len(group_cols), "axis", axis_name)
        blocks.append(grouped)
    if not blocks:
        return pd.DataFrame(columns=group_cols + ["axis", "mean", "std", "median", "n"])
    return pd.concat(blocks, ignore_index=True)


def build_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_name, column_name in get_measure_columns().items():
        series = df[column_name]
        min_value = float(series.min())
        max_value = float(series.max())
        rows.append(
            {
                "axis": axis_name,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": min_value,
                "max": max_value,
                "range": max_value - min_value,
                "n": int(series.count()),
            }
        )
    return pd.DataFrame(rows)


def build_pressure_drift(df: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for axis_name, column_name in get_measure_columns().items():
        grouped = (
            df.groupby(["model", "lang", "origin_group", "pressure"], dropna=False)[column_name]
            .agg(mean_score="mean", n="count")
            .reset_index()
        )
        if grouped.empty:
            continue
        pivot = grouped.pivot_table(
            index=["model", "lang", "origin_group"],
            columns="pressure",
            values=["mean_score", "n"],
            aggfunc="first",
        )
        if pivot.empty:
            continue
        pivot = pivot.reset_index()
        pivot.columns = [
            "_".join(str(part) for part in col if str(part))
            if isinstance(col, tuple)
            else str(col)
            for col in pivot.columns
        ]
        required = {"mean_score_pressure", "mean_score_no_pressure", "n_pressure", "n_no_pressure"}
        if not required.issubset(set(pivot.columns)):
            continue
        pivot = pivot.dropna(subset=["mean_score_pressure", "mean_score_no_pressure"]).copy()
        if pivot.empty:
            continue
        pivot["axis"] = axis_name
        pivot["drift_pressure_minus_no_pressure"] = (
            pivot["mean_score_pressure"] - pivot["mean_score_no_pressure"]
        )
        pivot = pivot.rename(
            columns={
                "mean_score_pressure": "mean_pressure",
                "mean_score_no_pressure": "mean_no_pressure",
                "n_pressure": "n_pressure",
                "n_no_pressure": "n_no_pressure",
            }
        )
        blocks.append(
            pivot[
                [
                    "model",
                    "lang",
                    "origin_group",
                    "axis",
                    "mean_pressure",
                    "mean_no_pressure",
                    "drift_pressure_minus_no_pressure",
                    "n_pressure",
                    "n_no_pressure",
                ]
            ]
        )
    if not blocks:
        return pd.DataFrame(
            columns=[
                "model",
                "lang",
                "origin_group",
                "axis",
                "mean_pressure",
                "mean_no_pressure",
                "drift_pressure_minus_no_pressure",
                "n_pressure",
                "n_no_pressure",
            ]
        )
    return pd.concat(blocks, ignore_index=True)


def write_outputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    origin_info: Dict[str, object],
    embedding_dimension: int,
    model_metadata_path: Path,
) -> None:
    ensure_dir(output_dir)

    scores_path = output_dir / "ideology_scores.csv"
    print(f"[save] Writing ideology_scores.csv ({len(df)} rows, {len(df.columns)} columns)")
    write_df(df, scores_path)

    if args.save_embeddings:
        np.save(output_dir / "ideology_embeddings.npy", embeddings)

    print("[save] Writing summary tables...")
    write_df(build_group_summary(df, ["origin_group"]), output_dir / "ideology_summary_by_origin.csv")
    write_df(build_group_summary(df, ["lang"]), output_dir / "ideology_summary_by_lang.csv")
    write_df(
        build_group_summary(df, ["lang", "origin_group"]),
        output_dir / "ideology_summary_by_lang_origin.csv",
    )
    write_df(
        build_group_summary(df, ["pressure", "origin_group"]),
        output_dir / "ideology_summary_by_pressure_origin.csv",
    )
    write_df(build_group_summary(df, ["model"]), output_dir / "ideology_summary_by_model.csv")
    write_df(build_axis_summary(df), output_dir / "ideology_summary_by_axis.csv")
    write_df(build_pressure_drift(df), output_dir / "ideology_drift_pressure.csv")

    anchor_config = {
        "model_name": args.model_name,
        "device": args.device,
        "input_path": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "normalize_embeddings": True,
        "batch_size": args.batch_size,
        "embedding_dimension": embedding_dimension,
        "save_embeddings": bool(args.save_embeddings),
        "anchor_sentences": ANCHOR_SENTENCES,
        "origin_resolution": {
            **origin_info,
            "model_metadata_exists": model_metadata_path.exists(),
        },
    }
    write_json(anchor_config, output_dir / "ideology_anchor_config.json")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    model_metadata_path = DEFAULT_MODEL_METADATA

    resolved_device = resolve_device(args.device)
    args.device = resolved_device

    df = load_input_table(input_path)
    print(f"[load] Reading {len(df)} rows from {input_path}")

    missing_justifications = int(df["justification"].isna().sum())
    empty_justifications = int(df["justification"].fillna("").astype(str).str.strip().eq("").sum())
    if missing_justifications or empty_justifications:
        print(
            f"[load] Missing/blank justifications detected: missing={missing_justifications}, blank={empty_justifications}"
        )
    df["justification"] = df["justification"].fillna("").astype(str)

    df, origin_info = resolve_origin_group(df, model_metadata_path)
    if origin_info["metadata_join_used"]:
        print(f"[load] model_metadata.csv used for origin mapping: {model_metadata_path}")
    if origin_info["keyword_fallback_rows"]:
        print(f"[load] Keyword fallback used for {origin_info['keyword_fallback_rows']} rows")

    texts = df["justification"].tolist()
    print(f"[embed] Encoding {len(texts)} justifications with {args.model_name} on {resolved_device}...")
    model = load_embedding_model(args.model_name, resolved_device)
    embeddings, embedding_dimension, elapsed = encode_justifications(
        texts=texts,
        model=model,
        batch_size=args.batch_size,
    )
    print(f"[embed] Done in {elapsed:.1f}s")

    centroids = encode_anchor_centroids(
        model=model,
        batch_size=args.batch_size,
    )
    df = compute_axis_scores(df, embeddings, centroids)
    df = compute_composite_score(df)

    write_outputs(
        df=df,
        embeddings=embeddings,
        output_dir=output_dir,
        args=args,
        origin_info=origin_info,
        embedding_dimension=embedding_dimension,
        model_metadata_path=model_metadata_path,
    )
    print(f"[done] All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
