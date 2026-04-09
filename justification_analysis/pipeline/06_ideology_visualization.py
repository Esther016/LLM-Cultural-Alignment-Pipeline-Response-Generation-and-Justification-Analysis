"""
06_ideology_visualization.py
=============================
Visualize ideology measurement results from 05_ideology_measurement.py.

Usage:
    python 06_ideology_visualization.py --input-dir outputs/ideology/ --output-dir outputs/ideology/figures/

Produces ~8 publication-ready figures covering:
  1. Origin group comparison (grouped bar)
  2. Language × origin group interaction (faceted bar)
  3. Pressure × origin group interaction (faceted bar)
  4. Model-level ideology distribution (strip/box plot)
  5. Pressure drift distribution by origin group
  6. Axis-level heatmap (lang × origin × axis)
  7. Model ranking dot plot (overall ideology)
  8. Pressure effect arrow plot per axis
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# STYLE CONFIGURATION — edit colours/fonts here
# ──────────────────────────────────────────────

CHINESE_COLOR = "#E04040"   # warm red
WESTERN_COLOR = "#3075B0"   # cool blue
PALETTE = {"Chinese": CHINESE_COLOR, "Western": WESTERN_COLOR}

AXIS_ORDER = ["gov_role", "liberty_security", "equality_merit", "censorship_acceptance", "overall"]
AXIS_LABELS = {
    "gov_role": "Government\nRole",
    "liberty_security": "Liberty vs\nSecurity",
    "equality_merit": "Equality vs\nMerit",
    "censorship_acceptance": "Censorship\nAcceptance",
    "overall": "Overall\n(composite)",
}

def setup_style():
    """Set global matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  [warn] File not found: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def load_all(input_dir: Path) -> dict:
    data = {}
    names = [
        "ideology_summary_by_origin",
        "ideology_summary_by_lang_origin",
        "ideology_summary_by_pressure_origin",
        "ideology_summary_by_model",
        "ideology_drift_pressure",
    ]
    for name in names:
        df = load_csv(input_dir / f"{name}.csv")
        if df is not None:
            data[name] = df
            print(f"  [load] {name}: {len(df)} rows")
    return data


# ──────────────────────────────────────────────
# FIGURE 1: Origin Group Comparison
# ──────────────────────────────────────────────

def fig1_origin_comparison(df: pd.DataFrame, out: Path):
    """Grouped bar chart: Chinese vs Western for each axis."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    axes_to_plot = [a for a in AXIS_ORDER if a in df["axis"].unique()]
    x = np.arange(len(axes_to_plot))
    width = 0.35

    for i, origin in enumerate(["Chinese", "Western"]):
        sub = df[df["origin_group"] == origin].set_index("axis")
        means = [sub.loc[a, "mean"] if a in sub.index else 0 for a in axes_to_plot]
        stds = [sub.loc[a, "std"] if a in sub.index else 0 for a in axes_to_plot]
        ns = [sub.loc[a, "n"] if a in sub.index else 0 for a in axes_to_plot]
        # Standard error of mean
        ses = [s / np.sqrt(n) if n > 0 else 0 for s, n in zip(stds, ns)]

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=ses,
                       color=PALETTE[origin], alpha=0.85,
                       edgecolor="white", linewidth=0.8,
                       capsize=3, error_kw={"linewidth": 1, "alpha": 0.6},
                       label=f"{origin}-origin")

        # Value labels
        for bar, m in zip(bars, means):
            y = bar.get_height()
            sign = 1 if y >= 0 else -1
            ax.text(bar.get_x() + bar.get_width() / 2, y + sign * 0.003,
                    f"{m:.4f}", ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in axes_to_plot])
    ax.set_ylabel("Mean Ideology Score")
    ax.set_title("Ideology Score by Origin Group", fontweight="bold", pad=12)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-")

    fig.tight_layout()
    fig.savefig(out / "fig1_origin_comparison.png")
    plt.close(fig)
    print(f"  [save] fig1_origin_comparison.png")


# ──────────────────────────────────────────────
# FIGURE 2: Language × Origin Group Interaction
# ──────────────────────────────────────────────

def fig2_lang_origin(df: pd.DataFrame, out: Path):
    """Faceted bar chart: lang × origin for each axis."""
    axes_to_plot = [a for a in AXIS_ORDER if a in df["axis"].unique()]
    n_axes = len(axes_to_plot)
    fig, axs = plt.subplots(1, n_axes, figsize=(3.2 * n_axes, 5), sharey=False)
    if n_axes == 1:
        axs = [axs]

    langs = sorted(df["lang"].unique())
    width = 0.35
    x = np.arange(len(langs))

    for idx, axis_name in enumerate(axes_to_plot):
        ax = axs[idx]
        sub = df[df["axis"] == axis_name]

        for i, origin in enumerate(["Chinese", "Western"]):
            osub = sub[sub["origin_group"] == origin].set_index("lang")
            means = [osub.loc[l, "mean"] if l in osub.index else 0 for l in langs]
            ns = [osub.loc[l, "n"] if l in osub.index else 1 for l in langs]
            stds = [osub.loc[l, "std"] if l in osub.index else 0 for l in langs]
            ses = [s / np.sqrt(n) for s, n in zip(stds, ns)]

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=ses,
                   color=PALETTE[origin], alpha=0.85,
                   edgecolor="white", linewidth=0.8,
                   capsize=3, error_kw={"linewidth": 1, "alpha": 0.5})

        ax.set_xticks(x)
        ax.set_xticklabels(["English", "Chinese"] if langs == ["en", "zh"] else langs)
        ax.set_title(AXIS_LABELS.get(axis_name, axis_name), fontsize=11)
        ax.axhline(0, color="#888888", linewidth=0.6)

        if idx == 0:
            ax.set_ylabel("Mean Ideology Score")

    # Shared legend
    handles = [mpatches.Patch(color=PALETTE["Chinese"], label="Chinese-origin"),
               mpatches.Patch(color=PALETTE["Western"], label="Western-origin")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    fig.suptitle("Ideology Score: Language × Origin Group", fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(out / "fig2_lang_origin_interaction.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] fig2_lang_origin_interaction.png")


# ──────────────────────────────────────────────
# FIGURE 3: Pressure × Origin Group
# ──────────────────────────────────────────────

def fig3_pressure_origin(df: pd.DataFrame, out: Path):
    """Faceted bar chart: pressure × origin for each axis."""
    axes_to_plot = [a for a in AXIS_ORDER if a in df["axis"].unique()]
    n_axes = len(axes_to_plot)
    fig, axs = plt.subplots(1, n_axes, figsize=(3.2 * n_axes, 5), sharey=False)
    if n_axes == 1:
        axs = [axs]

    pressures = ["no_pressure", "pressure"]
    pressure_labels = ["No Pressure", "Pressure"]
    width = 0.35
    x = np.arange(len(pressures))

    for idx, axis_name in enumerate(axes_to_plot):
        ax = axs[idx]
        sub = df[df["axis"] == axis_name]

        for i, origin in enumerate(["Chinese", "Western"]):
            osub = sub[sub["origin_group"] == origin].set_index("pressure")
            means = [osub.loc[p, "mean"] if p in osub.index else 0 for p in pressures]
            ns = [osub.loc[p, "n"] if p in osub.index else 1 for p in pressures]
            stds = [osub.loc[p, "std"] if p in osub.index else 0 for p in pressures]
            ses = [s / np.sqrt(n) for s, n in zip(stds, ns)]

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=ses,
                   color=PALETTE[origin], alpha=0.85,
                   edgecolor="white", linewidth=0.8,
                   capsize=3, error_kw={"linewidth": 1, "alpha": 0.5})

        ax.set_xticks(x)
        ax.set_xticklabels(pressure_labels)
        ax.set_title(AXIS_LABELS.get(axis_name, axis_name), fontsize=11)
        ax.axhline(0, color="#888888", linewidth=0.6)

        if idx == 0:
            ax.set_ylabel("Mean Ideology Score")

    handles = [mpatches.Patch(color=PALETTE["Chinese"], label="Chinese-origin"),
               mpatches.Patch(color=PALETTE["Western"], label="Western-origin")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    fig.suptitle("Ideology Score: Pressure × Origin Group", fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(out / "fig3_pressure_origin_interaction.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] fig3_pressure_origin_interaction.png")


# ──────────────────────────────────────────────
# FIGURE 4: Model-Level Distribution (Box + Strip)
# ──────────────────────────────────────────────

def fig4_model_distribution(df_model: pd.DataFrame, out: Path):
    """Box plot of model-level mean scores by origin group, for each axis."""
    # df_model has one row per (model, axis) — pivot to get model-level means
    axes_to_plot = [a for a in AXIS_ORDER if a in df_model["axis"].unique()]

    # We need origin_group per model — infer from the drift file or classify
    # For now, use keyword classification
    from typing import List
    CHINESE_KW: List[str] = [
        "qwen", "chatglm", "glm", "baichuan", "yi-", "yi_", "deepseek",
        "internlm", "moss", "tigerbot", "aquila", "xverse", "skywork",
        "minimax", "moonshot", "zhipu", "ernie", "spark", "doubao",
        "hunyuan", "sensenova", "abab", "chinese", "cn", "seed-oss",
        "longcat", "ling-", "ring-", "inclusionai", "bytedance",
        "thudm", "kimi",
    ]
    def classify(m: str) -> str:
        low = m.lower()
        for kw in CHINESE_KW:
            if kw in low:
                return "Chinese"
        return "Western"

    df_model = df_model.copy()
    if "origin_group" not in df_model.columns:
        df_model["origin_group"] = df_model["model"].apply(classify)

    n_axes = len(axes_to_plot)
    fig, axs = plt.subplots(1, n_axes, figsize=(3.5 * n_axes, 6), sharey=False)
    if n_axes == 1:
        axs = [axs]

    for idx, axis_name in enumerate(axes_to_plot):
        ax = axs[idx]
        sub = df_model[df_model["axis"] == axis_name].copy()

        for j, origin in enumerate(["Chinese", "Western"]):
            osub = sub[sub["origin_group"] == origin]["mean"].dropna()
            positions = [j]

            bp = ax.boxplot([osub.values], positions=positions, widths=0.55,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor=PALETTE[origin], alpha=0.3, edgecolor=PALETTE[origin]),
                            medianprops=dict(color=PALETTE[origin], linewidth=2),
                            whiskerprops=dict(color=PALETTE[origin]),
                            capprops=dict(color=PALETTE[origin]))

            # Jittered strip
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(osub))
            ax.scatter(np.full(len(osub), j) + jitter, osub.values,
                       color=PALETTE[origin], alpha=0.5, s=18, edgecolors="white", linewidths=0.3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Chinese", "Western"])
        ax.set_title(AXIS_LABELS.get(axis_name, axis_name), fontsize=11)
        ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--")

        if idx == 0:
            ax.set_ylabel("Model-Level Mean Score")

    fig.suptitle("Model-Level Ideology Score Distribution by Origin Group",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig4_model_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] fig4_model_distribution.png")


# ──────────────────────────────────────────────
# FIGURE 5: Pressure Drift Distribution
# ──────────────────────────────────────────────

def fig5_drift_distribution(df_drift: pd.DataFrame, out: Path):
    """Histogram/KDE of pressure drift by origin group, faceted by axis."""
    four_axes = [a for a in ["gov_role", "liberty_security", "equality_merit", "censorship_acceptance"]
                 if a in df_drift["axis"].unique()]
    n_axes = len(four_axes)
    if n_axes == 0:
        print("  [skip] fig5: no drift data")
        return

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.ravel()

    for idx, axis_name in enumerate(four_axes):
        ax = axs[idx]
        sub = df_drift[df_drift["axis"] == axis_name]

        for origin in ["Chinese", "Western"]:
            vals = sub[sub["origin_group"] == origin]["drift_pressure_minus_no_pressure"].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.45, color=PALETTE[origin],
                    edgecolor="white", linewidth=0.5, density=True, label=f"{origin}")
            # Add mean line
            m = vals.mean()
            ax.axvline(m, color=PALETTE[origin], linewidth=1.8, linestyle="--", alpha=0.8)
            ax.text(m, ax.get_ylim()[1] * 0.92, f"μ={m:.5f}",
                    color=PALETTE[origin], fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        ax.set_title(AXIS_LABELS.get(axis_name, axis_name).replace("\n", " "), fontsize=11)
        ax.set_xlabel("Drift (pressure − no_pressure)")
        ax.axvline(0, color="#888888", linewidth=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Pressure-Induced Ideology Drift Distribution by Origin Group",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "fig5_drift_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] fig5_drift_distribution.png")


# ──────────────────────────────────────────────
# FIGURE 6: Heatmap (Lang × Origin × Axis)
# ──────────────────────────────────────────────

def fig6_heatmap(df_lang_origin: pd.DataFrame, out: Path):
    """Heatmap of mean ideology score: rows = axis, columns = lang × origin."""
    four_axes = [a for a in AXIS_ORDER if a in df_lang_origin["axis"].unique()]

    # Build matrix
    df = df_lang_origin.copy()
    df["group_label"] = df["lang"].map({"en": "EN", "zh": "ZH"}) + " × " + df["origin_group"]
    col_order = ["EN × Chinese", "EN × Western", "ZH × Chinese", "ZH × Western"]
    col_order = [c for c in col_order if c in df["group_label"].unique()]

    matrix = []
    row_labels = []
    for axis_name in four_axes:
        row = []
        for grp in col_order:
            val = df[(df["axis"] == axis_name) & (df["group_label"] == grp)]["mean"]
            row.append(val.values[0] if len(val) > 0 else np.nan)
        matrix.append(row)
        row_labels.append(AXIS_LABELS.get(axis_name, axis_name).replace("\n", " "))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.nanmax(np.abs(matrix)) * 1.1
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_order)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Ideology Score", shrink=0.8)
    ax.set_title("Ideology Score Heatmap: Language × Origin Group × Axis",
                 fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out / "fig6_heatmap_lang_origin_axis.png")
    plt.close(fig)
    print(f"  [save] fig6_heatmap_lang_origin_axis.png")


# ──────────────────────────────────────────────
# FIGURE 7: Model Ranking Dot Plot (Overall)
# ──────────────────────────────────────────────

def fig7_model_ranking(df_model: pd.DataFrame, out: Path):
    """Horizontal dot plot of model-level overall ideology score."""
    sub = df_model[df_model["axis"] == "overall"].copy()
    if sub.empty:
        print("  [skip] fig7: no overall axis in model data")
        return

    CHINESE_KW = [
        "qwen", "chatglm", "glm", "baichuan", "yi-", "yi_", "deepseek",
        "internlm", "moss", "tigerbot", "aquila", "xverse", "skywork",
        "minimax", "moonshot", "zhipu", "ernie", "spark", "doubao",
        "hunyuan", "sensenova", "abab", "chinese", "cn", "seed-oss",
        "longcat", "ling-", "ring-", "inclusionai", "bytedance",
        "thudm", "kimi",
    ]
    def classify(m: str) -> str:
        low = m.lower()
        for kw in CHINESE_KW:
            if kw in low:
                return "Chinese"
        return "Western"

    if "origin_group" not in sub.columns:
        sub["origin_group"] = sub["model"].apply(classify)

    sub = sub.sort_values("mean", ascending=True).reset_index(drop=True)

    n_models = len(sub)
    fig_height = max(8, n_models * 0.22)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y = np.arange(n_models)
    colors = [PALETTE.get(o, "#888888") for o in sub["origin_group"]]

    ax.barh(y, sub["mean"], height=0.7, color=colors, alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#888888", linewidth=0.8)

    # Shorten model names for display
    short_names = sub["model"].str.replace("inclusionAI/", "").str.replace("MiniMaxAI/", "")
    short_names = short_names.str.replace("Qwen/", "").str.replace("THUDM/", "")
    short_names = short_names.str.replace("ByteDance-Seed/", "").str.replace("deepseek-ai/", "")

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Mean Overall Ideology Score (z-scored composite)")
    ax.set_title("Model Ranking by Overall Ideology Score", fontweight="bold", pad=12)

    handles = [mpatches.Patch(color=PALETTE["Chinese"], label="Chinese-origin"),
               mpatches.Patch(color=PALETTE["Western"], label="Western-origin")]
    ax.legend(handles=handles, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out / "fig7_model_ranking_overall.png")
    plt.close(fig)
    print(f"  [save] fig7_model_ranking_overall.png")


# ──────────────────────────────────────────────
# FIGURE 8: Pressure Effect Arrow Plot
# ──────────────────────────────────────────────

def fig8_pressure_arrow(df_pressure: pd.DataFrame, out: Path):
    """Arrow plot: no_pressure → pressure for each origin group × axis."""
    four_axes = [a for a in ["gov_role", "liberty_security", "equality_merit", "censorship_acceptance"]
                 if a in df_pressure["axis"].unique()]
    if not four_axes:
        print("  [skip] fig8: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = 0
    y_ticks = []
    y_labels = []

    for axis_name in four_axes:
        for origin in ["Chinese", "Western"]:
            sub = df_pressure[(df_pressure["axis"] == axis_name) &
                              (df_pressure["origin_group"] == origin)]
            np_val = sub[sub["pressure"] == "no_pressure"]["mean"].values
            p_val = sub[sub["pressure"] == "pressure"]["mean"].values

            if len(np_val) == 0 or len(p_val) == 0:
                continue

            np_val = np_val[0]
            p_val = p_val[0]

            color = PALETTE[origin]

            # Draw arrow from no_pressure to pressure
            ax.annotate("", xy=(p_val, y_pos), xytext=(np_val, y_pos),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
            # Start point
            ax.scatter([np_val], [y_pos], color=color, s=50, zorder=5, edgecolors="white")
            # End point label
            diff = p_val - np_val
            ax.text(p_val + 0.001 * (1 if diff >= 0 else -1), y_pos + 0.15,
                    f"Δ={diff:+.5f}", fontsize=8, color=color,
                    ha="left" if diff >= 0 else "right")

            y_ticks.append(y_pos)
            label = f"{AXIS_LABELS.get(axis_name, axis_name).replace(chr(10), ' ')} ({origin})"
            y_labels.append(label)
            y_pos += 1

        y_pos += 0.5  # Gap between axes

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Mean Ideology Score")
    ax.set_title("Pressure Effect: No Pressure → Pressure (by Origin Group × Axis)",
                 fontweight="bold", pad=12)
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out / "fig8_pressure_arrow.png")
    plt.close(fig)
    print(f"  [save] fig8_pressure_arrow.png")


# ──────────────────────────────────────────────
# FIGURE 9: Distribution Overlap (Density Plot)
# ──────────────────────────────────────────────

def fig9_distribution_overlap(scores_path, out: Path):
    """
    Overlapping density plots showing how much the two origin groups overlap.
    Uses the full ideology_scores.csv (response-level).
    """
    import warnings
    warnings.filterwarnings("ignore")

    if scores_path and Path(scores_path).exists():
        df = pd.read_csv(scores_path, encoding="utf-8-sig")
        print(f"  [load] Response-level scores: {len(df)} rows")
    else:
        print("  [skip] fig9: need ideology_scores.csv for density plot")
        return

    axes_to_plot = ["gov_role", "liberty_security", "equality_merit", "censorship_acceptance", "overall"]
    axis_cols = {a: f"ideology_{a}" for a in axes_to_plot}
    axes_to_plot = [a for a in axes_to_plot if axis_cols[a] in df.columns]

    if not axes_to_plot or "origin_group" not in df.columns:
        print("  [skip] fig9: missing required columns")
        return

    n_axes = len(axes_to_plot)
    n_rows = (n_axes + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axs = axs.ravel()

    for idx, axis_name in enumerate(axes_to_plot):
        ax = axs[idx]
        col = axis_cols[axis_name]

        for origin in ["Chinese", "Western"]:
            vals = df[df["origin_group"] == origin][col].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=80, density=True, alpha=0.4,
                    color=PALETTE[origin], edgecolor="none",
                    label=f"{origin} (n={len(vals):,})")
            m = vals.mean()
            ax.axvline(m, color=PALETTE[origin], linewidth=2, linestyle="--", alpha=0.8)

        ax.set_title(AXIS_LABELS.get(axis_name, axis_name).replace("\n", " "),
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Ideology Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, frameon=True)

        # Overlap percentage + Cohen's d
        ch_vals = df[df["origin_group"] == "Chinese"][col].dropna()
        we_vals = df[df["origin_group"] == "Western"][col].dropna()
        if len(ch_vals) > 0 and len(we_vals) > 0:
            lo = min(ch_vals.min(), we_vals.min())
            hi = max(ch_vals.max(), we_vals.max())
            bins_edges = np.linspace(lo, hi, 100)
            h1, _ = np.histogram(ch_vals, bins=bins_edges, density=True)
            h2, _ = np.histogram(we_vals, bins=bins_edges, density=True)
            bin_w = bins_edges[1] - bins_edges[0]
            overlap_pct = np.sum(np.minimum(h1, h2)) * bin_w * 100

            pooled_std = np.sqrt((ch_vals.std()**2 + we_vals.std()**2) / 2)
            d = abs(ch_vals.mean() - we_vals.mean()) / pooled_std if pooled_std > 0 else 0

            ax.text(0.97, 0.95,
                    f"Overlap: {overlap_pct:.1f}%\nCohen's d: {d:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.9))

    if n_axes < len(axs):
        for i in range(n_axes, len(axs)):
            axs[i].set_visible(False)

    fig.suptitle("Distribution Overlap: Chinese-origin vs Western-origin",
                 fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "fig9_distribution_overlap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] fig9_distribution_overlap.png")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize ideology measurement results")
    p.add_argument("--input-dir", type=str, default="outputs/ideology",
                   help="Directory containing ideology summary CSVs")
    p.add_argument("--output-dir", type=str, default="outputs/ideology/figures",
                   help="Directory to save figures")
    p.add_argument("--scores-path", type=str, default=None,
                   help="Path to ideology_scores.csv for response-level density plots (fig9)")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    print("[start] Loading data...")
    data = load_all(input_dir)

    if not data:
        print("[error] No data files found. Check --input-dir.")
        return

    print("\n[plot] Generating figures...\n")

    if "ideology_summary_by_origin" in data:
        fig1_origin_comparison(data["ideology_summary_by_origin"], output_dir)

    if "ideology_summary_by_lang_origin" in data:
        fig2_lang_origin(data["ideology_summary_by_lang_origin"], output_dir)

    if "ideology_summary_by_pressure_origin" in data:
        fig3_pressure_origin(data["ideology_summary_by_pressure_origin"], output_dir)

    if "ideology_summary_by_model" in data:
        fig4_model_distribution(data["ideology_summary_by_model"], output_dir)

    if "ideology_drift_pressure" in data:
        fig5_drift_distribution(data["ideology_drift_pressure"], output_dir)

    if "ideology_summary_by_lang_origin" in data:
        fig6_heatmap(data["ideology_summary_by_lang_origin"], output_dir)

    if "ideology_summary_by_model" in data:
        fig7_model_ranking(data["ideology_summary_by_model"], output_dir)

    if "ideology_summary_by_pressure_origin" in data:
        fig8_pressure_arrow(data["ideology_summary_by_pressure_origin"], output_dir)

    # Fig 9: distribution overlap (needs response-level scores)
    scores_path = args.scores_path
    if scores_path is None:
        # Auto-detect
        candidate = input_dir / "ideology_scores.csv"
        if candidate.exists():
            scores_path = str(candidate)
    if scores_path:
        fig9_distribution_overlap(scores_path, output_dir)

    print(f"\n[done] All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
