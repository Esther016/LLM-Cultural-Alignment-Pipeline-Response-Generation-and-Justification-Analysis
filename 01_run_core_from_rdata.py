import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyreadr
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from g4_strict import (
    build_g4_strict_pairs,
    compute_g4_legacy_many_to_many,
    compute_g4_strict_metrics,
    recompute_strict_proximal_flags,
)
from justification_config import DEFAULT_OUTPUT_ROOT, DEFAULT_RDATA, apply_overrides, load_config, resolve_output_dirs
from justification_utils import write_df, write_json

RANDOM_SEED = 42


@dataclass
class GateConfig:
    min_total_docs: int = 3000
    min_models: int = 3
    min_choices: int = 3
    min_cell_n: int = 8
    min_within_cells: int = 100
    min_pressure_pairs: int = 30
    min_crosslingual_pairs: int = 200


AXIS_ANCHORS = {
    "en": {
        "gov_role": {
            "pos": [
                "government responsibility",
                "public welfare",
                "state support",
                "regulation",
                "social safety net",
            ],
            "neg": [
                "small government",
                "personal responsibility",
                "market freedom",
                "limited state",
                "individual choice",
            ],
        },
        "liberty_security": {
            "pos": [
                "national security",
                "public order",
                "social stability",
                "safety first",
            ],
            "neg": [
                "civil liberties",
                "free speech",
                "privacy rights",
                "individual freedom",
            ],
        },
        "equality_merit": {
            "pos": ["equality", "redistribution", "fairness", "social justice"],
            "neg": ["merit", "efficiency", "incentives", "competition"],
        },
        "censorship_acceptance": {
            "pos": [
                "content moderation",
                "restrict harmful speech",
                "platform regulation",
                "misinformation control",
            ],
            "neg": [
                "anti censorship",
                "freedom of expression",
                "open debate",
                "speech rights",
            ],
        },
    },
    "zh": {
        "gov_role": {
            "pos": ["政府责任", "公共福利", "国家支持", "监管", "社会保障"],
            "neg": ["小政府", "个人责任", "市场自由", "有限政府", "个体选择"],
        },
        "liberty_security": {
            "pos": ["国家安全", "社会秩序", "社会稳定", "安全优先"],
            "neg": ["公民自由", "言论自由", "隐私权", "个人自由"],
        },
        "equality_merit": {
            "pos": ["平等", "再分配", "公平", "社会正义"],
            "neg": ["绩效", "效率", "激励", "竞争"],
        },
        "censorship_acceptance": {
            "pos": ["内容审查", "限制有害言论", "平台监管", "治理虚假信息"],
            "neg": ["反对审查", "表达自由", "公开辩论", "言论权利"],
        },
    },
}


STYLE_LEXICON = {
    "en": {
        "hedge": ["maybe", "perhaps", "might", "could", "likely", "possibly"],
        "assertive": ["must", "clearly", "definitely", "certainly", "undoubtedly"],
        "negation": ["not", "no", "never", "none", "cannot", "can't"],
    },
    "zh": {
        "hedge": ["可能", "也许", "或许", "大概", "未必", "倾向于"],
        "assertive": ["必须", "显然", "当然", "毫无疑问", "必然"],
        "negation": ["不", "没", "无", "并非", "不是", "不能"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run justification NLP analysis directly from .Rdata (G0-G5)."
    )
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--rdata", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None, help="Deprecated alias of --output-root.")
    parser.add_argument("--max-docs", type=int, default=120000)
    parser.add_argument("--topic-k", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-weak-sample", action="store_true")
    parser.add_argument("--min-total-docs", type=int, default=None)
    parser.add_argument("--min-models", type=int, default=None)
    parser.add_argument("--min-choices", type=int, default=None)
    parser.add_argument("--min-cell-n", type=int, default=None)
    parser.add_argument("--min-within-cells", type=int, default=None)
    parser.add_argument("--min-pressure-pairs", type=int, default=None)
    parser.add_argument("--min-crosslingual-pairs", type=int, default=None)
    parser.add_argument("--disable-g2", action="store_true")
    parser.add_argument("--disable-g4", action="store_true")
    parser.add_argument("--disable-legacy-outputs", action="store_true")
    return parser.parse_args()


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def normalize_model_name(model: str) -> str:
    if not isinstance(model, str):
        return str(model)
    model = model.strip()
    model = re.sub(r"_(CN|EN)$", "", model, flags=re.IGNORECASE)
    return model


def cleanup_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_prefix_shells(text: str) -> Tuple[str, bool, bool]:
    """Lightweight post-clean for justification shells while preserving body."""
    t = cleanup_text(text)
    if not t:
        return "", False, False

    changed_prefix = False
    changed_md = False

    prefix_patterns = [
        r"^\s*理由如下\s*[:：]\s*",
        r"^\s*理由说明\s*[:：]\s*",
        r"^\s*理由\s*[:：]\s*",
        r"^\s*justification\s*[:：]\s*",
        r"^\s*followed by justification\s*[:：]\s*",
    ]
    for ptn in prefix_patterns:
        t2 = re.sub(ptn, "", t, flags=re.IGNORECASE)
        if t2 != t:
            t = t2
            changed_prefix = True

    md_patterns = [
        r"^\s*\*\*[^*]{1,30}\*\*\s*[:：\-]\s*",
        r"^\s*#{1,6}\s*[^#:\n]{1,40}\s*[:：\-]\s*",
    ]
    for ptn in md_patterns:
        t2 = re.sub(ptn, "", t)
        if t2 != t:
            t = t2
            changed_md = True

    t2 = re.sub(r"^\s*[*_`#>\-\s]+", "", t)
    if t2 != t:
        t = t2
        changed_md = True

    return cleanup_text(t), changed_prefix, changed_md


def is_truncated_suspect(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 20:
        return False
    if bool(re.search(r"[.!?。！？]$", t)):
        return False
    return len(t) < 120


def is_mixed_lang_suspect(text: str, lang: str) -> bool:
    t = text or ""
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", t))
    has_en = bool(re.search(r"[A-Za-z]", t))
    if lang == "en":
        return has_zh
    if lang == "zh":
        return has_en
    return has_zh and has_en

def extract_justification(response_clean: str, response_choice: str) -> str:
    text = cleanup_text(response_clean)
    choice = cleanup_text(response_choice)
    if not text:
        return ""
    if not choice:
        return text

    choice_escaped = re.escape(choice)
    patterns = [
        rf"^\[?\(?\s*{choice_escaped}\s*[\]\)]?\s*[:：,\.\-，。；;!\?]*\s*",
        rf"^{choice_escaped}\s+",
    ]
    body = text
    for p in patterns:
        body2 = re.sub(p, "", body, flags=re.IGNORECASE)
        if body2 != body:
            body = body2
            break

    if body == text and "\n" in response_clean:
        lines = [x.strip() for x in response_clean.splitlines() if x.strip()]
        if len(lines) >= 2 and choice.lower() in lines[0].lower():
            body = " ".join(lines[1:])

    body = re.sub(r"^[\s\.,:;，。：；\-!\?]+", "", body).strip()
    if len(body) < 2:
        return ""
    return body


def read_and_standardize(rdata_path: Path, verbose: bool = True) -> pd.DataFrame:
    log(f"[read] {rdata_path}", verbose)
    objs = pyreadr.read_r(str(rdata_path))

    rows = []
    for name, df in objs.items():
        if not isinstance(df, pd.DataFrame):
            continue
        lower = name.lower()
        if "data_long" not in lower:
            continue

        pressure = "pressure" if "_p" in lower and "_nop" not in lower else "no_pressure"
        lang = "zh" if lower.endswith("_cn") else "en"

        qid_col = "Qid" if "Qid" in df.columns else "qid"
        choice_col = "Response_choice" if "Response_choice" in df.columns else None
        text_col = "Response_clean" if "Response_clean" in df.columns else None
        model_col = "Model" if "Model" in df.columns else None
        denial_col = "is_denial" if "is_denial" in df.columns else None
        question_col = "Question_CN" if lang == "zh" and "Question_CN" in df.columns else "Question"

        if not all([qid_col, choice_col, text_col, model_col]):
            continue

        tmp = pd.DataFrame(
            {
                "question_id": df[qid_col].astype(str),
                "model_raw": df[model_col].astype(str),
                "choice": df[choice_col].astype(str),
                "response_clean": df[text_col].astype(str),
                "question": df[question_col].astype(str) if question_col in df.columns else "",
                "source": df["Source"].astype(str) if "Source" in df.columns else "",
                "type": df["Type"].astype(str) if "Type" in df.columns else "",
                "lang": lang,
                "pressure": pressure,
            }
        )
        if denial_col:
            tmp["is_denial"] = df[denial_col].astype(str).str.lower().isin({"true", "1"})
        else:
            tmp["is_denial"] = False

        tmp["model"] = tmp["model_raw"].map(normalize_model_name)
        tmp["choice"] = tmp["choice"].map(cleanup_text)

        just_raw = [
            extract_justification(rc, ch)
            for rc, ch in zip(tmp["response_clean"].tolist(), tmp["choice"].tolist())
        ]
        cleaned = [remove_prefix_shells(x) for x in just_raw]
        tmp["justification"] = [x[0] for x in cleaned]
        tmp["has_prefix_marker_removed"] = [bool(x[1]) for x in cleaned]
        tmp["has_markdown_removed"] = [bool(x[2]) for x in cleaned]
        tmp["is_truncated_suspect"] = [is_truncated_suspect(x) for x in tmp["justification"].tolist()]
        tmp["is_mixed_lang_suspect"] = [
            is_mixed_lang_suspect(t, l)
            for t, l in zip(tmp["justification"].tolist(), tmp["lang"].tolist())
        ]
        rows.append(tmp)

    if not rows:
        raise ValueError("No usable long-format objects found in Rdata.")

    out = pd.concat(rows, ignore_index=True)
    out = out[~out["is_denial"]].copy()
    out = out[out["choice"].str.len() > 0]
    out = out[out["justification"].str.len() >= 8]
    out = out.reset_index(drop=True)
    out["response_instance_id"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def stratified_downsample(df: pd.DataFrame, max_docs: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if max_docs <= 0 or len(df) <= max_docs:
        return df.copy()
    grp = df.groupby(["lang", "pressure"], dropna=False)
    shares = grp.size().div(len(df))
    chunks = []
    rng = np.random.default_rng(seed)
    for key, g in grp:
        target = max(1, int(round(shares.loc[key] * max_docs)))
        if len(g) <= target:
            chunks.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=target, replace=False)
            chunks.append(g.loc[idx])
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > max_docs:
        out = out.sample(n=max_docs, random_state=seed)
    return out.reset_index(drop=True)


def gate_dataset(df: pd.DataFrame, gate: GateConfig) -> Tuple[bool, Dict]:
    meta = {
        "total_docs": int(len(df)),
        "models": int(df["model"].nunique()),
        "choices": int(df["choice"].nunique()),
    }
    reasons = []
    if meta["total_docs"] < gate.min_total_docs:
        reasons.append(f"total_docs={meta['total_docs']} < {gate.min_total_docs}")
    if meta["models"] < gate.min_models:
        reasons.append(f"models={meta['models']} < {gate.min_models}")
    if meta["choices"] < gate.min_choices:
        reasons.append(f"choices={meta['choices']} < {gate.min_choices}")

    cell = (
        df.groupby(["choice", "model", "lang", "pressure"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    low_cells = int((cell["n"] < gate.min_cell_n).sum())
    analyzable_cells = int((cell["n"] >= gate.min_cell_n).sum())
    meta["cells_total"] = int(len(cell))
    meta["cells_low_n"] = low_cells
    meta["cells_analyzable_n_ge_min"] = analyzable_cells
    if analyzable_cells < gate.min_within_cells:
        reasons.append(f"cells_analyzable_n_ge_min={analyzable_cells} < {gate.min_within_cells}")

    pressure_pairs = (
        df.groupby(["choice", "model", "lang", "pressure"])
        .size()
        .unstack("pressure", fill_value=0)
        .reset_index()
    )
    if {"pressure", "no_pressure"}.issubset(pressure_pairs.columns):
        n_pair = int(
            (
                (pressure_pairs["pressure"] >= gate.min_cell_n)
                & (pressure_pairs["no_pressure"] >= gate.min_cell_n)
            ).sum()
        )
    else:
        n_pair = 0
    meta["pressure_pair_cells"] = n_pair
    if n_pair < gate.min_pressure_pairs:
        reasons.append(f"pressure_pair_cells={n_pair} < {gate.min_pressure_pairs}")

    zh = df[df["lang"] == "zh"][["question_id", "model", "pressure", "choice"]].copy()
    zh["has_zh"] = 1
    en = df[df["lang"] == "en"][["question_id", "model", "pressure", "choice"]].copy()
    en["has_en"] = 1
    pair = zh.merge(en, on=["question_id", "model", "pressure", "choice"], how="inner")
    meta["crosslingual_pairs"] = int(len(pair))
    if meta["crosslingual_pairs"] < gate.min_crosslingual_pairs:
        reasons.append(f"crosslingual_pairs={meta['crosslingual_pairs']} < {gate.min_crosslingual_pairs}")

    return (len(reasons) == 0), {"meta": meta, "gate_reasons": reasons}


def vectorizer_by_lang(lang: str) -> TfidfVectorizer:
    if lang == "zh":
        return TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=5, max_df=0.95, max_features=80000)
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=80000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b",
        lowercase=True,
    )


def build_embeddings(df: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    n = len(df)
    emb = np.zeros((n, 128), dtype=np.float32)

    for lang in sorted(df["lang"].unique()):
        idx = df.index[df["lang"] == lang].to_numpy()
        texts = df.loc[idx, "justification"].fillna("").tolist()
        vec = vectorizer_by_lang(lang)
        X = vec.fit_transform(texts)
        if X.shape[1] < 3:
            continue
        n_comp = min(128, X.shape[0] - 1, X.shape[1] - 1)
        n_comp = max(2, int(n_comp))
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
        Z = svd.fit_transform(X)
        Z = normalize(Z)
        emb[idx, :n_comp] = Z.astype(np.float32)
        log(f"[embed] lang={lang} n={len(idx)} feat={X.shape[1]} comp={n_comp}", verbose)

    emb = normalize(emb)
    return emb


def mean_similarity_to_centroid(X: np.ndarray) -> Tuple[float, float]:
    if len(X) < 2:
        return math.nan, math.nan
    c = X.mean(axis=0, keepdims=True)
    c = normalize(c)
    sims = cosine_similarity(X, c).ravel()
    return float(np.mean(sims)), float(np.std(sims))


def calc_g1(df: pd.DataFrame, emb: np.ndarray, min_n: int = 8) -> Dict[str, pd.DataFrame]:
    within_rows = []
    group_cols = ["choice", "model", "lang", "pressure"]
    for key, g in df.groupby(group_cols, dropna=False):
        n = len(g)
        if n < min_n:
            continue
        idx = g.index.to_numpy()
        mean_sim, std_sim = mean_similarity_to_centroid(emb[idx])
        within_rows.append(
            {
                "choice": key[0],
                "model": key[1],
                "lang": key[2],
                "pressure": key[3],
                "n": n,
                "cohesion_mean": mean_sim,
                "cohesion_std": std_sim,
            }
        )
    within = pd.DataFrame(within_rows).sort_values(["cohesion_mean", "n"], ascending=[False, False])

    between_rows = []
    for (choice, lang, pressure), g in df.groupby(["choice", "lang", "pressure"]):
        model_groups = {m: x.index.to_numpy() for m, x in g.groupby("model") if len(x) >= min_n}
        models = sorted(model_groups.keys())
        if len(models) < 2:
            continue
        cents = {}
        for m in models:
            c = emb[model_groups[m]].mean(axis=0, keepdims=True)
            cents[m] = normalize(c)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                d = float(cosine_distances(cents[a], cents[b])[0, 0])
                between_rows.append(
                    {
                        "choice": choice,
                        "lang": lang,
                        "pressure": pressure,
                        "model_a": a,
                        "model_b": b,
                        "distance": d,
                    }
                )
    between = pd.DataFrame(between_rows).sort_values("distance", ascending=False)

    drift_rows = []
    for (choice, model, lang), g in df.groupby(["choice", "model", "lang"]):
        p = g[g["pressure"] == "pressure"]
        n = g[g["pressure"] == "no_pressure"]
        if len(p) < min_n or len(n) < min_n:
            continue
        cp = normalize(emb[p.index.to_numpy()].mean(axis=0, keepdims=True))
        cn = normalize(emb[n.index.to_numpy()].mean(axis=0, keepdims=True))
        d = float(cosine_distances(cp, cn)[0, 0])
        drift_rows.append(
            {
                "choice": choice,
                "model": model,
                "lang": lang,
                "n_pressure": len(p),
                "n_no_pressure": len(n),
                "drift_distance": d,
            }
        )
    drift = pd.DataFrame(drift_rows).sort_values("drift_distance", ascending=False)

    return {"within": within, "between": between, "drift": drift}


def fit_topics_by_lang(df: pd.DataFrame, topic_k: int, verbose: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
    assign = pd.Series(index=df.index, dtype="object")
    term_rows = []

    for lang in sorted(df["lang"].unique()):
        idx = df.index[df["lang"] == lang].to_numpy()
        texts = df.loc[idx, "justification"].fillna("").tolist()
        vec = vectorizer_by_lang(lang)
        X = vec.fit_transform(texts)
        if X.shape[0] < topic_k * 3 or X.shape[1] < topic_k * 5:
            continue
        k = min(topic_k, max(2, X.shape[0] // 1000), max(2, X.shape[1] // 50))
        nmf = NMF(n_components=k, init="nndsvda", random_state=RANDOM_SEED, max_iter=400)
        W = nmf.fit_transform(X)
        topics = np.argmax(W, axis=1)
        assign.loc[idx] = [f"{lang}_t{t:02d}" for t in topics]

        terms = np.array(vec.get_feature_names_out())
        for t in range(k):
            top_idx = np.argsort(nmf.components_[t])[::-1][:12]
            top_terms = ", ".join(terms[top_idx].tolist())
            term_rows.append({"lang": lang, "topic_id": f"{lang}_t{t:02d}", "top_terms": top_terms})
        log(f"[topic] lang={lang} k={k} n={len(idx)}", verbose)

    return assign, pd.DataFrame(term_rows)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def calc_g2(df: pd.DataFrame, topic_col: str = "topic") -> Dict[str, pd.DataFrame]:
    valid = df[df[topic_col].notna()].copy()
    if valid.empty:
        empty = pd.DataFrame()
        return {"prevalence": empty, "pressure_jsd": empty}

    prevalence = (
        valid.groupby(["choice", "model", "lang", "pressure", topic_col])
        .size()
        .rename("n")
        .reset_index()
    )
    denom = prevalence.groupby(["choice", "model", "lang", "pressure"])["n"].transform("sum")
    prevalence["prevalence"] = prevalence["n"] / denom

    js_rows = []
    for (choice, model, lang), g in valid.groupby(["choice", "model", "lang"]):
        gp = g[g["pressure"] == "pressure"]
        gn = g[g["pressure"] == "no_pressure"]
        if len(gp) < 8 or len(gn) < 8:
            continue
        topics = sorted(set(gp[topic_col].dropna().unique()).union(set(gn[topic_col].dropna().unique())))
        p_vec = np.array([(gp[topic_col] == t).mean() for t in topics], dtype=float)
        n_vec = np.array([(gn[topic_col] == t).mean() for t in topics], dtype=float)
        js = js_divergence(p_vec, n_vec)
        js_rows.append({"choice": choice, "model": model, "lang": lang, "topic_jsd_pressure_vs_no_pressure": js})
    js_df = pd.DataFrame(js_rows).sort_values("topic_jsd_pressure_vs_no_pressure", ascending=False)

    return {"prevalence": prevalence, "pressure_jsd": js_df}

def calc_axis_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lang in sorted(out["lang"].unique()):
        idx = out.index[out["lang"] == lang].to_numpy()
        if len(idx) == 0:
            continue
        vec = vectorizer_by_lang(lang)
        X = vec.fit_transform(out.loc[idx, "justification"].fillna("").tolist())
        anchors = AXIS_ANCHORS[lang]
        for axis_name, sides in anchors.items():
            pos_text = " ".join(sides["pos"])
            neg_text = " ".join(sides["neg"])
            a_pos = vec.transform([pos_text])
            a_neg = vec.transform([neg_text])
            s_pos = cosine_similarity(X, a_pos).ravel()
            s_neg = cosine_similarity(X, a_neg).ravel()
            out.loc[idx, f"axis_{axis_name}"] = s_pos - s_neg
    return out


def calc_g3(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    axis_cols = [c for c in df.columns if c.startswith("axis_")]
    summary_rows = []
    for axis in axis_cols:
        g = (
            df.groupby(["choice", "model", "lang", "pressure"])[axis]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "axis_aligned_score_mean",
                    "std": "axis_aligned_score_std",
                    "count": "n",
                }
            )
        )
        g["axis"] = axis
        g["axis_method"] = "anchor_based_framing_axis"
        summary_rows.append(g)
    axis_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    drift_rows = []
    for axis in axis_cols:
        for (choice, model, lang), g in df.groupby(["choice", "model", "lang"]):
            gp = g[g["pressure"] == "pressure"][axis].dropna()
            gn = g[g["pressure"] == "no_pressure"][axis].dropna()
            if len(gp) < 8 or len(gn) < 8:
                continue
            mean_diff = float(gp.mean() - gn.mean())
            s1 = float(gp.std(ddof=1)) if len(gp) > 1 else np.nan
            s2 = float(gn.std(ddof=1)) if len(gn) > 1 else np.nan
            den = (len(gp) + len(gn) - 2)
            if den > 0 and np.isfinite(s1) and np.isfinite(s2):
                pooled_std = float(np.sqrt(max(0.0, (((len(gp) - 1) * s1 * s1) + ((len(gn) - 1) * s2 * s2)) / den)))
            else:
                pooled_std = np.nan
            effect_size = float(mean_diff / pooled_std) if np.isfinite(pooled_std) and pooled_std > 1e-12 else np.nan
            drift_rows.append(
                {
                    "axis": axis,
                    "axis_method": "anchor_based_framing_axis",
                    "choice": choice,
                    "model": model,
                    "lang": lang,
                    "n_pressure": int(len(gp)),
                    "n_no_pressure": int(len(gn)),
                    "axis_shift_pressure_minus_no_pressure": mean_diff,
                    "pooled_std": pooled_std,
                    "effect_size": effect_size,
                }
            )
    drift = pd.DataFrame(drift_rows).sort_values("axis_shift_pressure_minus_no_pressure", ascending=False)

    dispersion_rows = []
    for axis in axis_cols:
        g = (
            df.groupby(["choice", "model", "lang", "pressure"])[axis]
            .std()
            .reset_index()
            .rename(columns={axis: "within_cell_axis_aligned_score_std"})
        )
        g["axis"] = axis
        g["axis_method"] = "anchor_based_framing_axis"
        dispersion_rows.append(g)
    dispersion = pd.concat(dispersion_rows, ignore_index=True) if dispersion_rows else pd.DataFrame()

    qc_rows = []
    if "question_id" in df.columns:
        for axis in axis_cols:
            tmp = df[["question_id", "choice", "model", "lang", "pressure", axis]].copy()
            tmp["_axis_qc"] = tmp[axis] - tmp.groupby(["question_id", "lang"])[axis].transform("mean")
            for (choice, model, lang), g in tmp.groupby(["choice", "model", "lang"]):
                gp = g[g["pressure"] == "pressure"]["_axis_qc"].dropna()
                gn = g[g["pressure"] == "no_pressure"]["_axis_qc"].dropna()
                if len(gp) < 8 or len(gn) < 8:
                    continue
                qc_rows.append(
                    {
                        "axis": axis,
                        "axis_method": "anchor_based_framing_axis",
                        "choice": choice,
                        "model": model,
                        "lang": lang,
                        "n_pressure": int(len(gp)),
                        "n_no_pressure": int(len(gn)),
                        "axis_shift_pressure_minus_no_pressure_question_centered": float(gp.mean() - gn.mean()),
                    }
                )
    question_centered = pd.DataFrame(qc_rows)

    return {
        "summary": axis_summary,
        "drift": drift,
        "dispersion": dispersion,
        "question_centered": question_centered,
    }

def tokenize_en(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())


def tokenize_zh_chars(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]", text)


def count_keywords(text: str, kws: List[str]) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return int(sum(t.count(k.lower()) for k in kws))


def add_style_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows = []
    for _, r in out.iterrows():
        lang = r["lang"]
        txt = r["justification"] if isinstance(r["justification"], str) else ""
        if lang == "en":
            toks = tokenize_en(txt)
            unit = max(1, len(toks))
        else:
            toks = tokenize_zh_chars(txt)
            unit = max(1, len(toks))
        lex = STYLE_LEXICON[lang]
        row = {
            "style_len_chars": len(txt),
            "style_len_units": len(toks),
            "style_hedge_per100": 100.0 * count_keywords(txt, lex["hedge"]) / unit,
            "style_assertive_per100": 100.0 * count_keywords(txt, lex["assertive"]) / unit,
            "style_negation_per100": 100.0 * count_keywords(txt, lex["negation"]) / unit,
            "style_digits_per100": 100.0 * len(re.findall(r"\d", txt)) / unit,
            "style_exclaim_per100": 100.0 * (txt.count("!") + txt.count("！")) / unit,
        }
        rows.append(row)
    feat = pd.DataFrame(rows, index=out.index)
    for c in feat.columns:
        out[c] = feat[c]
    return out


def calc_g5(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    style_cols = [c for c in df.columns if c.startswith("style_")]
    summary = (
        df.groupby(["choice", "model", "lang", "pressure"])[style_cols]
        .mean()
        .reset_index()
    )
    drift_rows = []
    for col in style_cols:
        g = (
            df.groupby(["choice", "model", "lang", "pressure"])[col]
            .mean()
            .unstack("pressure")
            .reset_index()
        )
        if {"pressure", "no_pressure"}.issubset(g.columns):
            g["metric"] = col
            g["shift_pressure_minus_no_pressure"] = g["pressure"] - g["no_pressure"]
            drift_rows.append(g[["choice", "model", "lang", "metric", "shift_pressure_minus_no_pressure"]])
    drift = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()
    return {"summary": summary, "drift": drift}

def calc_g4(df: pd.DataFrame, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    # Legacy many-to-many implementation retained for reference only.
    # Main pipeline now uses strict 1-to-1 in g4_strict.py.
    axis_cols = [c for c in df.columns if c.startswith("axis_")]
    style_cols = [c for c in df.columns if c.startswith("style_")]

    key_cols = ["question_id", "model", "pressure", "choice"]
    zh_cols = key_cols + ["justification"] + axis_cols + style_cols
    en_cols = key_cols + ["justification"] + axis_cols + style_cols
    zh = df[df["lang"] == "zh"][zh_cols].copy()
    en = df[df["lang"] == "en"][en_cols].copy()
    zh = zh.rename(columns={c: f"{c}_zh" for c in zh.columns if c not in key_cols})
    en = en.rename(columns={c: f"{c}_en" for c in en.columns if c not in key_cols})

    pair = zh.merge(en, on=key_cols, how="inner")
    if pair.empty:
        empty = pd.DataFrame()
        return {"pairs": empty, "ranking": empty, "notes": pd.DataFrame([{"note": "No zh/en pairs found."}])}

    embed_available = False
    embed_err = ""
    embedding_dist = np.full(len(pair), np.nan, dtype=float)
    try:
        from sentence_transformers import SentenceTransformer

        model = None
        for name in ["paraphrase-multilingual-MiniLM-L12-v2", "distiluse-base-multilingual-cased-v2"]:
            try:
                model = SentenceTransformer(name, local_files_only=True)
                break
            except Exception:
                model = None
        if model is not None:
            zh_vec = model.encode(pair["justification_zh"].fillna("").tolist(), normalize_embeddings=True, show_progress_bar=False)
            en_vec = model.encode(pair["justification_en"].fillna("").tolist(), normalize_embeddings=True, show_progress_bar=False)
            embedding_dist = 1.0 - np.sum(zh_vec * en_vec, axis=1)
            embed_available = True
        else:
            embed_err = "No local multilingual sentence-transformers model found."
    except Exception as exc:
        embed_err = f"Sentence-transformers unavailable: {type(exc).__name__}"

    axis_zh = pair[[f"{a}_zh" for a in axis_cols]].to_numpy(dtype=float)
    axis_en = pair[[f"{a}_en" for a in axis_cols]].to_numpy(dtype=float)
    axis_shift = np.linalg.norm(axis_zh - axis_en, axis=1)

    style_zh = pair[[f"{s}_zh" for s in style_cols]].to_numpy(dtype=float)
    style_en = pair[[f"{s}_en" for s in style_cols]].to_numpy(dtype=float)
    style_shift = np.linalg.norm(style_zh - style_en, axis=1)

    pair["axis_shift_l2"] = axis_shift
    pair["style_shift_l2"] = style_shift
    pair["embedding_distance"] = embedding_dist

    q_axis_low = np.nanpercentile(axis_shift, 33)
    q_axis_high = np.nanpercentile(axis_shift, 67)
    q_style_low = np.nanpercentile(style_shift, 33)
    q_style_high = np.nanpercentile(style_shift, 67)

    if embed_available and np.isfinite(embedding_dist).any():
        q_emb_low = np.nanpercentile(embedding_dist, 33)
        q_emb_high = np.nanpercentile(embedding_dist, 67)
        pair["translation_like"] = (
            (pair["axis_shift_l2"] <= q_axis_low)
            & (pair["style_shift_l2"] <= q_style_low)
            & (pair["embedding_distance"] <= q_emb_low)
        )
        pair["reframing_like"] = (
            (pair["axis_shift_l2"] >= q_axis_high)
            | (pair["style_shift_l2"] >= q_style_high)
            | (pair["embedding_distance"] >= q_emb_high)
        )
    else:
        pair["translation_like"] = (pair["axis_shift_l2"] <= q_axis_low) & (pair["style_shift_l2"] <= q_style_low)
        pair["reframing_like"] = (pair["axis_shift_l2"] >= q_axis_high) | (pair["style_shift_l2"] >= q_style_high)

    ranking = (
        pair.groupby("model")
        .agg(
            n_pairs=("model", "size"),
            mean_axis_shift=("axis_shift_l2", "mean"),
            mean_style_shift=("style_shift_l2", "mean"),
            mean_embedding_distance=("embedding_distance", "mean"),
            translation_rate=("translation_like", "mean"),
            reframing_rate=("reframing_like", "mean"),
        )
        .reset_index()
        .sort_values(["mean_axis_shift", "mean_style_shift"], ascending=[True, True])
    )

    notes = pd.DataFrame(
        [
            {
                "embedding_available": embed_available,
                "embedding_note": "ok" if embed_available else embed_err,
            }
        ]
    )
    log(f"[g4] pairs={len(pair)} embed_available={embed_available}", verbose)
    return {"pairs": pair, "ranking": ranking, "notes": notes}


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    cfg = load_config(args.config if args.config else None)
    cfg = apply_overrides(
        cfg,
        {
            "rdata_path": args.rdata,
            "output_root": args.output_root or args.out_dir,
            "min_total_docs": args.min_total_docs,
            "min_models": args.min_models,
            "min_choices": args.min_choices,
            "min_cell_n": args.min_cell_n,
            "min_within_cells": args.min_within_cells,
            "min_pressure_pairs": args.min_pressure_pairs,
            "min_crosslingual_pairs": args.min_crosslingual_pairs,
        },
    )

    if args.disable_g2:
        cfg.run_g2 = False
    if args.disable_g4:
        cfg.run_g4 = False
    if args.disable_legacy_outputs:
        cfg.keep_legacy_outputs = False
    if not cfg.rdata_path:
        cfg.rdata_path = str(DEFAULT_RDATA)
    if not cfg.output_root:
        cfg.output_root = str(DEFAULT_OUTPUT_ROOT)

    dirs = resolve_output_dirs(cfg)
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    core_dir = dirs["core"]
    legacy_dir = dirs["legacy"]

    summary: Dict[str, object] = {
        "seed": RANDOM_SEED,
        "resolved_paths": {k: str(v) for k, v in dirs.items()},
        "resolved_config": {
            "min_cell_n": cfg.min_cell_n,
            "min_model_cells": cfg.min_model_cells,
            "min_model_pairs": cfg.min_model_pairs,
            "min_pressure_pairs": cfg.min_pressure_pairs,
            "min_crosslingual_pairs": cfg.min_crosslingual_pairs,
            "run_g2": cfg.run_g2,
            "run_g4": cfg.run_g4,
            "keep_legacy_outputs": cfg.keep_legacy_outputs,
        },
    }

    df_all = read_and_standardize(Path(cfg.rdata_path), verbose=True)
    write_df(df_all, core_dir / "g0_standardized_justification_all.csv")

    df = stratified_downsample(df_all, args.max_docs, seed=RANDOM_SEED)
    write_df(df, core_dir / "g0_standardized_justification_used.csv")
    summary["g0"] = {
        "rows_all": int(len(df_all)),
        "rows_used": int(len(df)),
        "models_used": int(df["model"].nunique()),
        "choices_used": int(df["choice"].nunique()),
    }

    gate_ok, gate_info = gate_dataset(
        df,
        GateConfig(
            min_total_docs=cfg.min_total_docs,
            min_models=cfg.min_models,
            min_choices=cfg.min_choices,
            min_cell_n=cfg.min_cell_n,
            min_within_cells=cfg.min_within_cells,
            min_pressure_pairs=cfg.min_pressure_pairs,
            min_crosslingual_pairs=cfg.min_crosslingual_pairs,
        ),
    )
    summary["gate"] = gate_info
    if not gate_ok:
        stop_msg = {
            "status": "stopped_small_sample",
            "reason": "Sample is too small/imbalanced for defensible inference under current design.",
            "details": gate_info,
            "recommendation": [
                "Increase generations per (choice, model, lang, pressure).",
                "Reduce stratification dimensions or collapse rare choices.",
                "Only report descriptive stats instead of drift claims.",
            ],
        }
        write_json(stop_msg, core_dir / "STOP_small_sample.json")
        if not args.allow_weak_sample:
            raise SystemExit("Stopped: sample check failed. See STOP_small_sample.json for rationale.")

    emb = build_embeddings(df, verbose=True)
    g1 = calc_g1(df, emb, min_n=cfg.min_cell_n)
    write_df(g1["within"], core_dir / "g1_within_group_cohesion.csv")
    write_df(g1["between"], core_dir / "g1_between_model_distance.csv")
    write_df(g1["drift"], core_dir / "g1_pressure_embedding_drift.csv")
    summary["g1"] = {k: int(len(v)) for k, v in g1.items()}

    if cfg.run_g2:
        topic_assign, topic_terms = fit_topics_by_lang(df, topic_k=args.topic_k, verbose=True)
        df["topic"] = topic_assign
        write_df(topic_terms, core_dir / "g2_topic_terms.csv")
        g2 = calc_g2(df, topic_col="topic")
        write_df(g2["prevalence"], core_dir / "g2_topic_prevalence.csv")
        write_df(g2["pressure_jsd"], core_dir / "g2_topic_pressure_jsd.csv")
        summary["g2"] = {
            "topic_terms": int(len(topic_terms)),
            "prevalence_rows": int(len(g2["prevalence"])),
            "pressure_jsd_rows": int(len(g2["pressure_jsd"])),
        }

    df = calc_axis_scores(df)
    g3 = calc_g3(df)
    write_df(g3["summary"], core_dir / "g3_axis_summary.csv")
    write_df(g3["drift"], core_dir / "g3_axis_pressure_drift.csv")
    write_df(g3["question_centered"], core_dir / "g3_axis_pressure_drift_question_centered.csv")
    write_df(g3["dispersion"], core_dir / "g3_within_cell_axis_dispersion.csv")
    summary["g3"] = {k: int(len(v)) for k, v in g3.items()}
    summary["g3"]["axis_method"] = "anchor_based_framing_axis"

    df = add_style_features(df)
    g5 = calc_g5(df)
    write_df(g5["summary"], core_dir / "g5_style_summary.csv")
    write_df(g5["drift"], core_dir / "g5_style_pressure_drift.csv")
    summary["g5"] = {k: int(len(v)) for k, v in g5.items()}

    if cfg.run_g4:
        strict_pairs = build_g4_strict_pairs(df)
        strict_enriched, strict_thresholds, embed_ok, embed_note = recompute_strict_proximal_flags(strict_pairs)
        strict_ranking = compute_g4_strict_metrics(strict_enriched)

        write_df(strict_pairs, core_dir / "g4_crosslingual_pairs_strict_1to1.csv")
        write_df(strict_enriched, core_dir / "g4_crosslingual_pairs_strict_1to1_enriched.csv")
        write_df(strict_ranking, core_dir / "g4_crosslingual_model_ranking_strict_1to1.csv")
        # strict is now the default main result chain (compatibility aliases)
        write_df(strict_enriched, core_dir / "g4_crosslingual_pairs.csv")
        write_df(strict_ranking, core_dir / "g4_crosslingual_model_ranking.csv")

        g4_notes = pd.DataFrame(
            [
                {
                    "g4_default_pairing": "strict_1to1",
                    "flags_recomputed_on": "strict_1to1_sample",
                    "legacy_flag_inheritance": False,
                    "embedding_available": embed_ok,
                    "embedding_note": embed_note,
                    "strict_thresholds": json.dumps(strict_thresholds, ensure_ascii=False),
                }
            ]
        )
        write_df(g4_notes, core_dir / "g4_notes.csv")

        if cfg.keep_legacy_outputs:
            legacy_pairs = compute_g4_legacy_many_to_many(df)
            write_df(legacy_pairs, legacy_dir / "g4_crosslingual_pairs_many_to_many.csv")
            if not legacy_pairs.empty:
                legacy_enriched, _, _, _ = recompute_strict_proximal_flags(legacy_pairs)
                legacy_ranking = compute_g4_strict_metrics(legacy_enriched)
            else:
                legacy_ranking = pd.DataFrame()
            write_df(legacy_ranking, legacy_dir / "g4_crosslingual_model_ranking_many_to_many.csv")

        summary["g4"] = {
            "pairing_default": "strict_1to1",
            "pairs": int(len(strict_enriched)),
            "ranking_rows": int(len(strict_ranking)),
            "embedding_available": embed_ok,
            "flags_recomputed_on_strict_sample": True,
            "legacy_outputs_written": bool(cfg.keep_legacy_outputs),
        }

    write_json(summary, core_dir / "analysis_summary.json")

    log(f"[done] outputs -> {dirs['root']}", True)


if __name__ == "__main__":
    main()






