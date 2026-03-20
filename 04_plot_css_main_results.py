import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from justification_config import DEFAULT_OUTPUT_ROOT, apply_overrides, load_config, resolve_output_dirs
from justification_utils import read_csv_if_exists, write_json

# ---------- Global visual spec ----------
FONT_SERIF = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
BASE_FONT = 12
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 10
ANNOT_SIZE = 10
GRID_ALPHA = 0.16
GRID_WIDTH = 0.7
BAR_EDGE = "0.35"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "font.size": BASE_FONT,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "semibold",
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE + 1,
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CSS-oriented group analysis results from postprocessed CSS tables.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--post-dir", type=str, default=None)
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


# ---------- IO helpers ----------
def save_fig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_fig_dual(primary: Path, secondary: Path, dpi: int) -> None:
    primary.parent.mkdir(parents=True, exist_ok=True)
    secondary.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(primary, dpi=dpi, bbox_inches="tight")
    plt.savefig(secondary, dpi=dpi, bbox_inches="tight")
    plt.close()


def warn(msg: str, warnings: List[str]) -> None:
    text = f"[warn] {msg}"
    warnings.append(text)
    print(text)


def resolve_css_table(css_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    primary = css_dir / filename
    fallback = PROJECT_ROOT / "outputs" / "css" / "tables" / filename
    for path in [primary, fallback]:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            return df
    return None


# ---------- Text / style helpers ----------
def prettify_text(value: object) -> str:
    return str(value).replace("_", " ").strip()


def prettify_series(values: Sequence[object]) -> List[str]:
    out = []
    for x in values:
        text = prettify_text(x)
        out.append(text.upper() if text.lower() in {"en", "zh"} else text)
    return out


def apply_times_text(ax) -> None:
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        text.set_fontfamily("Times New Roman")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Times New Roman")
    for txt in ax.texts:
        txt.set_fontfamily("Times New Roman")


def format_axis_text(ax, *, x_rotation: int = 0, x_ha: str = "center", y_rotation: int = 0) -> None:
    xticks = ax.get_xticks()
    xlabels = [prettify_text(t.get_text()) for t in ax.get_xticklabels()]
    if len(xticks) == len(xlabels) and len(xlabels) > 0:
        ax.set_xticks(xticks, labels=xlabels)
    yticks = ax.get_yticks()
    ylabels = [prettify_text(t.get_text()) for t in ax.get_yticklabels()]
    if len(yticks) == len(ylabels) and len(ylabels) > 0:
        ax.set_yticks(yticks, labels=ylabels)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(x_rotation)
        lbl.set_ha(x_ha)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(y_rotation)
    apply_times_text(ax)


def style_axis(ax, *, grid_axis: str = "y", y_zero: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=GRID_ALPHA, linewidth=GRID_WIDTH)
    else:
        ax.grid(False)
    if y_zero:
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=0.8, length=4)


def soften_legend(legend, fontsize: int = LEGEND_SIZE, title_size: int = LEGEND_SIZE) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_alpha(0.72)
    frame.set_facecolor("white")
    frame.set_edgecolor("0.72")
    frame.set_linewidth(0.8)
    title = legend.get_title()
    title.set_fontfamily("Times New Roman")
    title.set_fontsize(title_size)
    for t in legend.get_texts():
        t.set_fontfamily("Times New Roman")
        t.set_fontsize(fontsize)


def bar_label_all(ax, fmt: str, *, padding: int = 3, fontsize: int = ANNOT_SIZE) -> None:
    for cont in ax.containers:
        labels = ax.bar_label(cont, fmt=fmt, padding=padding, fontsize=fontsize)
        for txt in labels:
            txt.set_fontfamily("Times New Roman")


def style_heatmap(ax, *, title: str, xlabel: str, ylabel: str, x_rotation: int = 0, x_ha: str = "center") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis_text(ax, x_rotation=x_rotation, x_ha=x_ha)


def draw_barplot(
    ax,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Sequence[object]] = None,
    hue_order: Optional[Sequence[object]] = None,
    palette="deep",
    width: float = 0.72,
    legend: bool = True,
    dodge: bool = True,
    color=None,
):
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        width=width,
        legend=legend,
        dodge=dodge,
        color=color,
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        ax=ax,
    )


# ---------- Plotters ----------
def plot_g1(css_dir: Path, out_dir: Path, dpi: int, manifest: Dict[str, str], warnings: List[str]) -> None:
    raw = read_csv_if_exists(css_dir.parent / "model_metadata_clean.csv")
    g1_raw = read_csv_if_exists(css_dir.parent.parent / "core" / "g1_pressure_embedding_drift.csv")
    if raw is not None and not raw.empty and g1_raw is not None and not g1_raw.empty:
        g1 = g1_raw.merge(raw[["model", "origin_group"]], on="model", how="left")
        if {"lang", "drift_distance"}.issubset(g1.columns):
            fig, ax = plt.subplots(figsize=(8.4, 6.0))
            sns.boxplot(data=g1, x="lang", y="drift_distance", hue="lang", legend=False, linewidth=0.9, ax=ax)
            style_axis(ax, grid_axis="y")
            ax.set_title("G1 Drift by Language")
            ax.set_xlabel("Language")
            ax.set_ylabel("Drift Distance")
            format_axis_text(ax)
            p = out_dir / "g1_drift_by_lang_boxplot.png"
            save_fig(p, dpi)
            manifest["g1_drift_by_lang_boxplot"] = str(p)
    else:
        warn("g1_drift_by_lang_boxplot skipped: missing source tables", warnings)

    s1 = read_csv_if_exists(css_dir / "g1_by_origin_group_summary.csv")
    if s1 is None or s1.empty:
        warn("g1_drift_by_origin_group_bar skipped: missing g1_by_origin_group_summary.csv", warnings)
    else:
        fig, ax = plt.subplots(figsize=(8.8, 6.0))
        order = [x for x in ["Chinese", "Western"] if x in set(s1["origin_group"].astype(str))] or sorted(s1["origin_group"].astype(str).unique())
        draw_barplot(ax, s1, x="origin_group", y="mean", hue=None, order=order, width=0.58, palette="Blues")
        style_axis(ax, grid_axis="y")
        ax.set_title("G1 Mean Drift by Origin Group")
        ax.set_xlabel("Origin Group")
        ax.set_ylabel("Mean Drift Distance")
        bar_label_all(ax, "%.3f")
        format_axis_text(ax)
        p = out_dir / "g1_drift_by_origin_group_bar.png"
        save_fig(p, dpi)
        manifest["g1_drift_by_origin_group_bar"] = str(p)

    s2 = read_csv_if_exists(css_dir / "g1_by_lang_origin_group_summary.csv")
    if s2 is None or s2.empty:
        warn("g1_drift_lang_by_origin_group_heatmap skipped: missing g1_by_lang_origin_group_summary.csv", warnings)
    else:
        pv = s2.pivot(index="origin_group", columns="lang", values="mean").fillna(0.0)
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"shrink": 0.9}, annot_kws={"fontsize": 10, "fontfamily": "Times New Roman"}, ax=ax)
        style_heatmap(ax, title="G1 Mean Drift: Language × Origin Group", xlabel="Language", ylabel="Origin Group")
        p = out_dir / "g1_drift_lang_by_origin_group_heatmap.png"
        save_fig(p, dpi)
        manifest["g1_drift_lang_by_origin_group_heatmap"] = str(p)


def plot_g1_drift_lang_origin_group_groupedbar(
    css_dir: Path,
    out_dir: Path,
    out_extra_dir: Path,
    dpi: int,
    manifest: Dict[str, str],
    warnings: List[str],
) -> None:
    df = resolve_css_table(css_dir, "g1_by_lang_origin_group_summary.csv")
    if df is None or df.empty:
        warn("g1 groupedbar skipped: missing g1_by_lang_origin_group_summary.csv", warnings)
        return
    req = {"lang", "origin_group", "mean"}
    if not req.issubset(df.columns):
        warn("g1 groupedbar skipped: required columns lang/origin_group/mean missing", warnings)
        return

    lang_order = [x for x in ["en", "zh"] if x in set(df["lang"].astype(str))] or sorted(df["lang"].astype(str).unique())
    origin_order = [x for x in ["Chinese", "Western"] if x in set(df["origin_group"].astype(str))] or sorted(df["origin_group"].astype(str).unique())

    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    draw_barplot(ax, df, x="lang", y="mean", hue="origin_group", order=lang_order, hue_order=origin_order, width=0.64, palette="Set2")
    style_axis(ax, grid_axis="y")
    ax.set_ylim(bottom=0)
    ax.set_title("Pressure-Induced Semantic Drift by Language and Origin Group")
    ax.set_xlabel("Language")
    ax.set_ylabel("Mean Semantic Drift")
    leg = ax.legend(title="Origin Group", frameon=True)
    soften_legend(leg)
    bar_label_all(ax, "%.3f")
    format_axis_text(ax)

    filename = "g1_drift_by_lang_origin_group_groupedbar.png"
    p1 = out_dir / filename
    p2 = out_extra_dir / filename
    save_fig_dual(p1, p2, dpi)
    manifest["g1_drift_by_lang_origin_group_groupedbar"] = str(p2)


def plot_g3(css_dir: Path, out_dir: Path, dpi: int, manifest: Dict[str, str], warnings: List[str]) -> None:
    a = read_csv_if_exists(css_dir / "g3_by_axis_lang_summary.csv")
    if a is None or a.empty:
        warn("g3_abs_shift_by_axis_lang_heatmap skipped: missing g3_by_axis_lang_summary.csv", warnings)
    else:
        pv = a.pivot(index="axis", columns="lang", values="mean_abs_shift").fillna(0.0)
        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="magma", linewidths=0.5, cbar_kws={"shrink": 0.9}, annot_kws={"fontsize": 10, "fontfamily": "Times New Roman"}, ax=ax)
        style_heatmap(ax, title="G3 Mean Absolute Shift: Axis × Language", xlabel="Language", ylabel="Axis")
        p = out_dir / "g3_abs_shift_by_axis_lang_heatmap.png"
        save_fig(p, dpi)
        manifest["g3_abs_shift_by_axis_lang_heatmap"] = str(p)

    b = read_csv_if_exists(css_dir / "g3_by_axis_origin_group_summary.csv")
    if b is None or b.empty:
        warn("g3_abs_shift_by_axis_origin_group_heatmap skipped: missing g3_by_axis_origin_group_summary.csv", warnings)
    else:
        pv = b.pivot(index="axis", columns="origin_group", values="mean_abs_shift").fillna(0.0)
        fig, ax = plt.subplots(figsize=(9.2, 6.0))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="crest", linewidths=0.5, cbar_kws={"shrink": 0.9}, annot_kws={"fontsize": 10, "fontfamily": "Times New Roman"}, ax=ax)
        style_heatmap(ax, title="G3 Mean Absolute Shift: Axis × Origin Group", xlabel="Origin Group", ylabel="Axis")
        p = out_dir / "g3_abs_shift_by_axis_origin_group_heatmap.png"
        save_fig(p, dpi)
        manifest["g3_abs_shift_by_axis_origin_group_heatmap"] = str(p)

    c = read_csv_if_exists(css_dir / "g3_by_axis_lang_origin_group_summary.csv")
    if c is None or c.empty:
        warn("g3_abs_shift_axis_lang_origin_group_faceted skipped: missing g3_by_axis_lang_origin_group_summary.csv", warnings)
    else:
        origin_order = [x for x in ["Chinese", "Western"] if x in set(c["origin_group"].astype(str))] or sorted(c["origin_group"].astype(str).unique())
        lang_order = [x for x in ["en", "zh"] if x in set(c["lang"].astype(str))] or sorted(c["lang"].astype(str).unique())
        g = sns.FacetGrid(c, col="origin_group", col_order=origin_order, sharex=False, sharey=True, height=5.2, aspect=1.18, despine=True)
        g.map_dataframe(
            sns.barplot,
            x="mean_abs_shift",
            y="axis",
            hue="lang",
            hue_order=lang_order,
            palette="Set2",
            width=0.7,
            edgecolor=BAR_EDGE,
            linewidth=0.7,
        )
        g.set_titles(col_template="Origin Group: {col_name}", size=13, weight="semibold")
        g.set_axis_labels("Mean Absolute Shift", "Axis")
        handles = labels = None
        for ax in g.axes.flat:
            style_axis(ax, grid_axis="x")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
            bar_label_all(ax, "%.3f", padding=2, fontsize=9)
            format_axis_text(ax)
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        if handles and labels:
            legend = g.fig.legend(handles, prettify_series(labels), title="Language", loc="upper right", bbox_to_anchor=(0.5, -0.02), ncol=len(labels), frameon=True)
            soften_legend(legend)
        g.fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.26)
        g.fig.suptitle("G3 Mean Absolute Shift by Axis, Faceted by Origin Group", y=0.98, fontfamily='Times New Roman')
        p = out_dir / "g3_abs_shift_axis_lang_origin_group_faceted.png"
        g.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(g.fig)
        manifest["g3_abs_shift_axis_lang_origin_group_faceted"] = str(p)


def plot_g4(css_dir: Path, out_dir: Path, dpi: int, manifest: Dict[str, str], warnings: List[str]) -> None:
    s = read_csv_if_exists(css_dir / "g4_by_origin_group_summary.csv")
    if s is None or s.empty:
        warn("g4 origin-group plots skipped: missing g4_by_origin_group_summary.csv", warnings)
    else:
        origin_order = [x for x in ["Chinese", "Western"] if x in set(s["origin_group"].astype(str))] or sorted(s["origin_group"].astype(str).unique())
        long = s.melt(id_vars=["origin_group"], value_vars=["translation_rate", "reframing_rate"], var_name="metric", value_name="rate")
        fig, ax = plt.subplots(figsize=(9.4, 6.2))
        draw_barplot(ax, long, x="origin_group", y="rate", hue="metric", order=origin_order, hue_order=["translation_rate", "reframing_rate"], width=0.62, palette=["#2a9d8f", "#e76f51"])
        style_axis(ax, grid_axis="y")
        ax.set_ylabel("Rate")
        ax.set_xlabel("Origin Group")
        ax.set_title("G4 Translation vs Reframing by Origin Group", fontfamily='Times New Roman')
        leg = ax.legend(title="Metric", frameon=True)
        soften_legend(leg)
        bar_label_all(ax, "%.3f")
        format_axis_text(ax)
        p = out_dir / "g4_translation_vs_reframing_by_origin_group.png"
        save_fig(p, dpi)
        manifest["g4_translation_vs_reframing_by_origin_group"] = str(p)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
        specs = [
            ("mean_axis_shift", "Axis Shift Inconsistency", "Mean Axis Shift", "Blues"),
            ("mean_style_shift", "Style Shift Inconsistency", "Mean Style Shift", "Oranges"),
        ]
        for ax, (metric, title, ylabel, palette) in zip(axes, specs):
            draw_barplot(ax, s, x="origin_group", y=metric, order=origin_order, width=0.56, palette=palette)
            style_axis(ax, grid_axis="y")
            ax.set_title(title)
            ax.set_xlabel("Origin Group")
            ax.set_ylabel(ylabel)
            fmt = "%.4f" if metric == "mean_axis_shift" else "%.1f"
            bar_label_all(ax, fmt, padding=2, fontsize=9)
            format_axis_text(ax)
        fig.suptitle("G4 Cross-lingual Inconsistency by Origin Group", y=1.02, fontfamily='Times New Roman')
        p2 = out_dir / "g4_crosslingual_inconsistency_by_origin_group.png"
        save_fig(p2, dpi)
        manifest["g4_crosslingual_inconsistency_by_origin_group"] = str(p2)

    jm = read_csv_if_exists(css_dir / "g4_model_metadata_joined.csv")
    if jm is None or jm.empty or "mean_axis_shift" not in jm.columns:
        warn("g4_model_ranking_colored_by_origin_group skipped: missing g4_model_metadata_joined.csv or metric", warnings)
    else:
        top = jm.sort_values("mean_axis_shift", ascending=False).head(20).sort_values("mean_axis_shift", ascending=True)
        fig, ax = plt.subplots(figsize=(11.8, 8.4))
        draw_barplot(ax, top, x="mean_axis_shift", y="model", hue="origin_group", dodge=False, palette="Set2", width=0.72)
        style_axis(ax, grid_axis="x")
        ax.set_title("G4 Model Ranking, Colored by Origin Group (Top 20 by Mean Axis Shift)")
        ax.set_xlabel("Mean Axis Shift")
        ax.set_ylabel("Model")
        leg = ax.legend(title="Origin Group", frameon=True)
        soften_legend(leg)
        format_axis_text(ax)
        p = out_dir / "g4_model_ranking_colored_by_origin_group.png"
        save_fig(p, dpi)
        manifest["g4_model_ranking_colored_by_origin_group"] = str(p)


def plot_g4_translation_reframing_slope_by_origin_group(
    css_dir: Path,
    out_dir: Path,
    out_extra_dir: Path,
    dpi: int,
    manifest: Dict[str, str],
    warnings: List[str],
) -> None:
    df = resolve_css_table(css_dir, "g4_by_origin_group_summary.csv")
    if df is None or df.empty:
        warn("g4 slope skipped: missing g4_by_origin_group_summary.csv", warnings)
        return
    req = {"origin_group", "translation_rate", "reframing_rate"}
    if not req.issubset(df.columns):
        warn("g4 slope skipped: required columns origin_group/translation_rate/reframing_rate missing", warnings)
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    palette = sns.color_palette("Set2", n_colors=max(2, len(df)))
    for idx, row in df.reset_index(drop=True).iterrows():
        y0 = float(row["translation_rate"])
        y1 = float(row["reframing_rate"])
        color = palette[idx % len(palette)]
        label = prettify_text(row["origin_group"])
        ax.plot([0, 1], [y0, y1], marker="o", linewidth=2.0, color=color, label=label)
        t1 = ax.text(-0.03, y0, f"{y0:.3f}", ha="right", va="center", fontsize=9)
        t2 = ax.text(1.03, y1, f"{y1:.3f}", ha="left", va="center", fontsize=9)
        t1.set_fontfamily("Times New Roman")
        t2.set_fontfamily("Times New Roman")
    style_axis(ax, grid_axis="y")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0, max(1.0, float(df[["translation_rate", "reframing_rate"]].max().max()) * 1.12))
    ax.set_xticks([0, 1], labels=["Translation rate", "Reframing rate"])
    ax.set_ylabel("Rate")
    ax.set_title("Translation vs Reframing by Origin Group")
    leg = ax.legend(title="Origin Group", frameon=True)
    soften_legend(leg)
    apply_times_text(ax)

    filename = "g4_translation_reframing_slope_by_origin_group.png"
    p1 = out_dir / filename
    p2 = out_extra_dir / filename
    save_fig_dual(p1, p2, dpi)
    manifest["g4_translation_reframing_slope_by_origin_group"] = str(p2)


def plot_g5(css_dir: Path, out_dir: Path, dpi: int, manifest: Dict[str, str], warnings: List[str]) -> None:
    a = read_csv_if_exists(css_dir / "g5_by_metric_lang_summary.csv")
    if a is None or a.empty:
        warn("g5_shift_by_metric_lang_heatmap skipped: missing g5_by_metric_lang_summary.csv", warnings)
    else:
        pv = a.pivot(index="metric", columns="lang", values="mean_abs_shift").fillna(0.0)
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="rocket_r", linewidths=0.5, cbar_kws={"shrink": 0.9}, annot_kws={"fontsize": 10, "fontfamily": "Times New Roman"}, ax=ax)
        style_heatmap(ax, title="G5 Mean Absolute Shift: Metric × Language", xlabel="Language", ylabel="Metric")
        p = out_dir / "g5_shift_by_metric_lang_heatmap.png"
        save_fig(p, dpi)
        manifest["g5_shift_by_metric_lang_heatmap"] = str(p)

    b = read_csv_if_exists(css_dir / "g5_by_metric_origin_group_summary.csv")
    if b is None or b.empty:
        warn("g5_shift_by_metric_origin_group_heatmap skipped: missing g5_by_metric_origin_group_summary.csv", warnings)
    else:
        pv = b.pivot(index="metric", columns="origin_group", values="mean_abs_shift").fillna(0.0)
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="flare", linewidths=0.5, cbar_kws={"shrink": 0.9}, annot_kws={"fontsize": 10, "fontfamily": "Times New Roman"}, ax=ax)
        style_heatmap(ax, title="G5 Mean Absolute Shift: Metric × Origin Group", xlabel="Origin Group", ylabel="Metric")
        p = out_dir / "g5_shift_by_metric_origin_group_heatmap.png"
        save_fig(p, dpi)
        manifest["g5_shift_by_metric_origin_group_heatmap"] = str(p)


def plot_g5_length_compression_lang_origin_group_faceted(
    css_dir: Path,
    out_dir: Path,
    out_extra_dir: Path,
    dpi: int,
    manifest: Dict[str, str],
    warnings: List[str],
) -> None:
    df = resolve_css_table(css_dir, "g5_by_metric_lang_origin_group_summary.csv")
    if df is None or df.empty:
        warn("g5 length faceted skipped: missing g5_by_metric_lang_origin_group_summary.csv", warnings)
        return
    req = {"metric", "lang", "origin_group", "mean_shift"}
    if not req.issubset(df.columns):
        warn("g5 length faceted skipped: required columns metric/lang/origin_group/mean_shift missing", warnings)
        return

    use = df[df["metric"].isin(["style_len_chars", "style_len_units"])].copy()
    if use.empty:
        warn("g5 length faceted skipped: style_len_chars/style_len_units not found", warnings)
        return

    origin_order = [x for x in ["Chinese", "Western"] if x in set(use["origin_group"].astype(str))] or sorted(use["origin_group"].astype(str).unique())
    lang_order = [x for x in ["en", "zh"] if x in set(use["lang"].astype(str))] or sorted(use["lang"].astype(str).unique())
    metric_order = [m for m in ["style_len_chars", "style_len_units"] if m in set(use["metric"].astype(str))]

    fig, axes = plt.subplots(1, len(metric_order), figsize=(14.8, 7.2), sharey=True)
    if len(metric_order) == 1:
        axes = [axes]
    y_min = float(min(0.0, use["mean_shift"].min() * 1.25))
    y_max = float(max(0.0, use["mean_shift"].max() * 1.25))
    if y_min == y_max:
        y_max = y_min + 1.0

    palette = sns.color_palette("Set2", n_colors=len(lang_order))
    handles = labels = None
    for ax, metric in zip(axes, metric_order):
        sub = use[use["metric"] == metric].copy()
        draw_barplot(ax, sub, x="origin_group", y="mean_shift", hue="lang", order=origin_order, hue_order=lang_order, width=0.58, palette=palette)
        style_axis(ax, grid_axis="y", y_zero=True)
        ax.set_ylim(y_min, y_max)
        ax.set_title(prettify_text(metric.replace("style_len_", "")))
        ax.set_xlabel("Origin Group")
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        bar_label_all(ax, "%.2f")
        format_axis_text(ax)
    axes[0].set_ylabel("Mean Shift")
    if handles and labels:
        legend = fig.legend(handles, prettify_series(labels), title="Language", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(labels), frameon=True)
        soften_legend(legend)
    fig.suptitle("Pressure-Induced Length Compression by Language and Origin Group", y=1.08,  fontfamily='Times New Roman')
    fig.subplots_adjust(top=0.80, wspace=0.28)

    filename = "g5_length_compression_lang_origin_group_faceted.png"
    p1 = out_dir / filename
    p2 = out_extra_dir / filename
    save_fig_dual(p1, p2, dpi)
    manifest["g5_length_compression_lang_origin_group_faceted"] = str(p2)


def plot_issue_domain(css_dir: Path, out_dir: Path, dpi: int, manifest: Dict[str, str], warnings: List[str]) -> None:
    warn("g1_drift_by_issue_domain.png not generated: no reliable question_id in G1 final table", warnings)
    warn("g3_abs_shift_by_issue_domain.png not generated: no reliable question_id in G3 final table", warnings)

    g4i = read_csv_if_exists(css_dir / "g4_by_lang_issue_domain_summary.csv")
    if g4i is None or g4i.empty:
        warn("g4_reframing_rate_by_issue_domain skipped: missing g4_by_lang_issue_domain_summary.csv", warnings)
        return

    order = g4i.sort_values("reframing_rate", ascending=False)["issue_domain"]
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    draw_barplot(ax, g4i, x="issue_domain", y="reframing_rate", order=order, width=0.64, color="#e76f51")
    style_axis(ax, grid_axis="y")
    ax.set_title("G4 Reframing Rate by Issue Domain")
    ax.set_xlabel("Issue Domain")
    ax.set_ylabel("Reframing Rate")
    bar_label_all(ax, "%.3f")
    format_axis_text(ax, x_rotation=0, x_ha="right")
    p = out_dir / "g4_reframing_rate_by_issue_domain.png"
    save_fig(p, dpi)
    manifest["g4_reframing_rate_by_issue_domain"] = str(p)


# ---------- Main ----------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config if args.config else None)
    cfg = apply_overrides(cfg, {"output_root": args.output_root})
    if not cfg.output_root:
        cfg.output_root = str(DEFAULT_OUTPUT_ROOT)
    dirs = resolve_output_dirs(cfg)

    post_dir = Path(args.post_dir) if args.post_dir else dirs["postprocessed"]
    plots_root = Path(args.plots_dir) if args.plots_dir else dirs["plots"]
    css_dir = post_dir / "css"
    out_dir = plots_root / "css"
    out_extra_dir = PROJECT_ROOT / "outputs" / "css" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_extra_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    warnings: List[str] = []
    manifest: Dict[str, str] = {}
    plot_g1(css_dir, out_dir, args.dpi, manifest, warnings)
    plot_g3(css_dir, out_dir, args.dpi, manifest, warnings)
    plot_g5(css_dir, out_dir, args.dpi, manifest, warnings)
    plot_g4(css_dir, out_dir, args.dpi, manifest, warnings)
    plot_g1_drift_lang_origin_group_groupedbar(css_dir, out_dir, out_extra_dir, args.dpi, manifest, warnings)
    plot_g4_translation_reframing_slope_by_origin_group(css_dir, out_dir, out_extra_dir, args.dpi, manifest, warnings)
    plot_g5_length_compression_lang_origin_group_faceted(css_dir, out_dir, out_extra_dir, args.dpi, manifest, warnings)
    plot_issue_domain(css_dir, out_dir, args.dpi, manifest, warnings)

    write_json({"figures": manifest, "warnings": warnings}, out_dir / "css_fig_manifest.json")
    print(f"done {out_dir}")


if __name__ == "__main__":
    main()
