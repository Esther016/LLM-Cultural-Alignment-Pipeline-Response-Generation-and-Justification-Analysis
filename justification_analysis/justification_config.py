from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RDATA = Path("data_use") / "data_use" / "data_use_long_v1.Rdata"
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass
class JustificationConfig:
    rdata_path: str = str(DEFAULT_RDATA)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    core_subdir: str = "core"
    postprocessed_subdir: str = "postprocessed"
    plots_subdir: str = "plots"
    diagnostics_subdir: str = "diagnostics"
    legacy_subdir: str = "legacy"

    min_total_docs: int = 3000
    min_models: int = 3
    min_choices: int = 3
    min_cell_n: int = 8
    min_within_cells: int = 100
    min_pressure_pairs: int = 30
    min_crosslingual_pairs: int = 200

    min_model_cells: int = 5
    min_model_pairs: int = 20

    run_g2: bool = True
    run_g4: bool = True
    keep_legacy_outputs: bool = True

    g2_audit_csv: str = ""


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    # Accept both UTF-8 and UTF-8-with-BOM config files.
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object.")
    return data


def load_config(config_path: Optional[str] = None) -> JustificationConfig:
    data = _load_json(Path(config_path)) if config_path else {}
    base = asdict(JustificationConfig())
    base.update(data)
    return JustificationConfig(**base)


def resolve_output_dirs(cfg: JustificationConfig) -> Dict[str, Path]:
    root = Path(cfg.output_root)
    return {
        "root": root,
        "core": root / cfg.core_subdir,
        "postprocessed": root / cfg.postprocessed_subdir,
        "plots": root / cfg.plots_subdir,
        "diagnostics": root / cfg.diagnostics_subdir,
        "legacy": root / cfg.legacy_subdir,
    }


def apply_overrides(cfg: JustificationConfig, overrides: Dict[str, Any]) -> JustificationConfig:
    data = asdict(cfg)
    for k, v in overrides.items():
        if v is None:
            continue
        if k in data:
            data[k] = v
    return JustificationConfig(**data)
