from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _find_metric_bases(columns: List[str], prefix: str) -> List[str]:
    return sorted(
        c[:-3]
        for c in columns
        if c.startswith(prefix) and c.endswith("_zh") and f"{c[:-3]}_en" in columns
    )


def build_g4_strict_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build strict 1-to-1 cross-lingual pairs.

    Why strict is default:
    - Legacy key-only merge can create many-to-many pair explosion when one side has duplicates.
    - Deterministic 1-to-1 matching keeps each zh/en instance used at most once per key.
    """
    key = ["question_id", "model", "pressure", "choice"]
    axis_cols = [c for c in df.columns if c.startswith("axis_")]
    style_cols = [c for c in df.columns if c.startswith("style_")]

    use_cols = key + ["lang", "justification"] + axis_cols + style_cols
    use = df[use_cols].copy()

    dedup = use.drop_duplicates(subset=key + ["lang", "justification"]).copy()
    zh = dedup[dedup["lang"] == "zh"].copy()
    en = dedup[dedup["lang"] == "en"].copy()

    if zh.empty or en.empty:
        return pd.DataFrame()

    for frame in (zh, en):
        frame["just_len"] = frame["justification"].astype(str).str.len().fillna(0).astype(int)

    # deterministic ranking within key: longer justifications first, then lexical tie-break.
    sort_cols = key + ["just_len", "justification"]
    asc = [True, True, True, True, False, True]
    zh = zh.sort_values(sort_cols, ascending=asc)
    en = en.sort_values(sort_cols, ascending=asc)
    zh["instance_id"] = zh.groupby(key).cumcount() + 1
    en["instance_id"] = en.groupby(key).cumcount() + 1

    zh = zh.drop(columns=["lang", "just_len"])
    en = en.drop(columns=["lang", "just_len"])

    pair = zh.merge(en, on=key + ["instance_id"], how="inner", suffixes=("_zh", "_en"))
    return pair


def maybe_compute_embedding_distance(pair: pd.DataFrame) -> Tuple[np.ndarray, bool, str]:
    embedding_dist = np.full(len(pair), np.nan, dtype=float)
    if pair.empty:
        return embedding_dist, False, "no pairs"

    try:
        from sentence_transformers import SentenceTransformer

        model = None
        for name in ["paraphrase-multilingual-MiniLM-L12-v2", "distiluse-base-multilingual-cased-v2"]:
            try:
                model = SentenceTransformer(name, local_files_only=True)
                break
            except Exception:
                model = None

        if model is None:
            return embedding_dist, False, "No local multilingual sentence-transformers model found."

        zh_vec = model.encode(
            pair["justification_zh"].fillna("").tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        en_vec = model.encode(
            pair["justification_en"].fillna("").tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding_dist = 1.0 - np.sum(zh_vec * en_vec, axis=1)
        return embedding_dist, True, "ok"
    except Exception as exc:
        return embedding_dist, False, f"Sentence-transformers unavailable: {type(exc).__name__}"


def recompute_strict_proximal_flags(pair: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], bool, str]:
    out = pair.copy()
    if out.empty:
        return out, {}, False, "no pairs"

    axis_bases = _find_metric_bases(list(out.columns), "axis_")
    style_bases = _find_metric_bases(list(out.columns), "style_")

    if not axis_bases or not style_bases:
        out["axis_shift_l2"] = np.nan
        out["style_shift_l2"] = np.nan
        out["embedding_distance"] = np.nan
        out["translation_like"] = False
        out["reframing_like"] = False
        out["translation_proximal_flag"] = 0
        out["reframing_proximal_flag"] = 0
        return out, {}, False, "axis/style columns unavailable"

    axis_zh = out[[f"{b}_zh" for b in axis_bases]].to_numpy(dtype=float)
    axis_en = out[[f"{b}_en" for b in axis_bases]].to_numpy(dtype=float)
    style_zh = out[[f"{b}_zh" for b in style_bases]].to_numpy(dtype=float)
    style_en = out[[f"{b}_en" for b in style_bases]].to_numpy(dtype=float)

    out["axis_shift_l2"] = np.linalg.norm(axis_zh - axis_en, axis=1)
    out["style_shift_l2"] = np.linalg.norm(style_zh - style_en, axis=1)

    embedding_dist, embed_ok, embed_note = maybe_compute_embedding_distance(out)
    out["embedding_distance"] = embedding_dist

    q_axis_low = float(np.nanpercentile(out["axis_shift_l2"], 33))
    q_axis_high = float(np.nanpercentile(out["axis_shift_l2"], 67))
    q_style_low = float(np.nanpercentile(out["style_shift_l2"], 33))
    q_style_high = float(np.nanpercentile(out["style_shift_l2"], 67))

    thresholds = {
        "axis_low_q33": q_axis_low,
        "axis_high_q67": q_axis_high,
        "style_low_q33": q_style_low,
        "style_high_q67": q_style_high,
    }

    if embed_ok and np.isfinite(out["embedding_distance"]).any():
        q_emb_low = float(np.nanpercentile(out["embedding_distance"], 33))
        q_emb_high = float(np.nanpercentile(out["embedding_distance"], 67))
        thresholds["embedding_low_q33"] = q_emb_low
        thresholds["embedding_high_q67"] = q_emb_high

        out["translation_like"] = (
            (out["axis_shift_l2"] <= q_axis_low)
            & (out["style_shift_l2"] <= q_style_low)
            & (out["embedding_distance"] <= q_emb_low)
        )
        out["reframing_like"] = (
            (out["axis_shift_l2"] >= q_axis_high)
            | (out["style_shift_l2"] >= q_style_high)
            | (out["embedding_distance"] >= q_emb_high)
        )
    else:
        out["translation_like"] = (out["axis_shift_l2"] <= q_axis_low) & (out["style_shift_l2"] <= q_style_low)
        out["reframing_like"] = (out["axis_shift_l2"] >= q_axis_high) | (out["style_shift_l2"] >= q_style_high)

    out["translation_proximal_flag"] = out["translation_like"].astype(bool).astype(int)
    out["reframing_proximal_flag"] = out["reframing_like"].astype(bool).astype(int)

    return out, thresholds, embed_ok, embed_note


def compute_g4_strict_metrics(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pd.DataFrame()
    return (
        pair.groupby("model")
        .agg(
            n_pairs=("model", "size"),
            mean_axis_shift=("axis_shift_l2", "mean"),
            mean_style_shift=("style_shift_l2", "mean"),
            mean_embedding_distance=("embedding_distance", "mean"),
            translation_proximal_rate=("translation_proximal_flag", "mean"),
            reframing_proximal_rate=("reframing_proximal_flag", "mean"),
            # backward-compatible aliases used by old downstream code
            translation_rate=("translation_proximal_flag", "mean"),
            reframing_rate=("reframing_proximal_flag", "mean"),
        )
        .reset_index()
        .sort_values(["mean_axis_shift", "mean_style_shift"], ascending=[True, True])
    )


def compute_g4_legacy_many_to_many(df: pd.DataFrame) -> pd.DataFrame:
    key = ["question_id", "model", "pressure", "choice"]
    axis_cols = [c for c in df.columns if c.startswith("axis_")]
    style_cols = [c for c in df.columns if c.startswith("style_")]

    zh_cols = key + ["justification"] + axis_cols + style_cols
    en_cols = key + ["justification"] + axis_cols + style_cols

    zh = df[df["lang"] == "zh"][zh_cols].copy()
    en = df[df["lang"] == "en"][en_cols].copy()
    zh = zh.rename(columns={c: f"{c}_zh" for c in zh.columns if c not in key})
    en = en.rename(columns={c: f"{c}_en" for c in en.columns if c not in key})
    return zh.merge(en, on=key, how="inner")
