from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def safe_z(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def resolve_g4_rate_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    if {"translation_proximal_rate", "reframing_proximal_rate"}.issubset(df.columns):
        return "translation_proximal_rate", "reframing_proximal_rate"
    if {"translation_rate", "reframing_rate"}.issubset(df.columns):
        return "translation_rate", "reframing_rate"
    return None
