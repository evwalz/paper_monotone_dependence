#!/usr/bin/env python3
"""
Five metric panels (RMSE, SEEPS, ACC, CMA, CID) for precipitation.

3 rows × 2 columns; SEEPS in row 2 is shifted to the horizontal center after
``tight_layout`` (same panel size as the others).
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_FONT_BASE = 18
_FONT_TITLE = 22
_FONT_TICK = 16
_FONT_LEGEND = 16


def load_metric_data(input_dir, metric_name, variable, lead_time):
    """
    Load metric data for a specific variable and lead time.
    ``metric_name`` ``cid`` reads ``cindx_results/cindx_*`` files.
    """
    metric_data = {}
    model_names = ["graphcast", "ifs_hres", "persistence", "climatology", "pangu"]

    filename_prefix = {"cid": "cindx"}.get(metric_name, metric_name)
    results_dir = os.path.join(input_dir, f"{filename_prefix}_results")

    for model in model_names:
        filename = f"{filename_prefix}_{model}_{variable}_lead{lead_time}h.txt"
        filepath = os.path.join(results_dir, filename)

        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            display_name = {
                "climatology": "Climatology",
                "graphcast": "GraphCast",
                "pangu": "Pangu",
                "ifs_hres": "IFS HRES",
                "persistence": "Persistence",
            }.get(model, model)
            metric_data[display_name] = data

    return metric_data


def calculate_skill_score_negative(forecast_metric, climatology_metric):
    with np.errstate(divide="ignore", invalid="ignore"):
        skill_score = (climatology_metric - forecast_metric) / climatology_metric * 100
        skill_score = np.where(np.isfinite(skill_score), skill_score, np.nan)
    return skill_score


def calculate_skill_score_positive(forecast_metric, climatology_metric):
    with np.errstate(divide="ignore", invalid="ignore"):
        skill_score = (forecast_metric - climatology_metric) / climatology_metric * 100
        skill_score = np.where(np.isfinite(skill_score), skill_score, np.nan)
    return skill_score


def _metric_configs(use_skill_scores: bool):
    if use_skill_scores:
        return [
            ("cma", "CMA Skill Score", "CMA SS (%)", True, (-2, 88)),
            ("cid", "CID Skill Score", "CID SS (%)", True, (-2, 88)),
            ("seeps", "SEEPS Skill Score", "SEEPS SS (%)", False, (-2, 88)),
            ("rmse", "RMSE Skill Score", "RMSE SS (%)", False, (-40, 90)),
            ("acc", "ACC", "ACC", None, (-0.02, 1.02)),
        ], "skill_scores"
    return [
        ("cma", "CMA", "CMA", True, (0.43, 1.01)),
        ("cid", "CID", "CID", True, (0.43, 1.01)),
        ("seeps", "SEEPS", "SEEPS Score", False, None),
        ("rmse", "RMSE", "RMSE (m)", False, None),
        ("acc", "ACC", "ACC", None, (-0.2, 1.01)),
    ], "metrics"


def _style_context():
    plt.rcParams.update(
        {
            "font.size": _FONT_BASE,
            "axes.titlesize": _FONT_TITLE,
            "axes.labelsize": _FONT_BASE,
            "xtick.labelsize": _FONT_TICK,
            "ytick.labelsize": _FONT_TICK,
            "legend.fontsize": _FONT_LEGEND,
        }
    )
    color_codes = sns.color_palette("colorblind", 6)
    model_color_map = {
        "Persistence": "#9467bd",
        "Climatology": color_codes[2],
        "GraphCast": color_codes[0],
        "Pangu": color_codes[4],
        "IFS HRES": color_codes[1],
    }
    return model_color_map


def _draw_metric_panel(
    ax,
    *,
    metric_name: str,
    metric_title: str,
    ylabel: str,
    is_positive_oriented,
    ylim,
    metric_data: dict[str, np.ndarray],
    latitudes: np.ndarray,
    model_order: list[str],
    model_color_map: dict[str, str],
    use_skill_scores: bool,
    panel_label: str | None = None,
) -> None:
    plot_skill_score = use_skill_scores and is_positive_oriented is not None

    if plot_skill_score:
        if "Climatology" not in metric_data:
            ax.text(
                0.5,
                0.5,
                "Climatology data missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=_FONT_BASE,
            )
            return
        climatology_metric = metric_data["Climatology"]
        for model_name in model_order:
            if model_name == "Climatology":
                continue
            if model_name not in metric_data:
                continue
            forecast_metric = metric_data[model_name]
            if is_positive_oriented:
                y = calculate_skill_score_positive(forecast_metric, climatology_metric)
            else:
                y = calculate_skill_score_negative(forecast_metric, climatology_metric)
            ax.plot(
                latitudes,
                y,
                label=model_name,
                color=model_color_map[model_name],
                linewidth=2,
            )
        ax.axhline(
            0,
            color=model_color_map["Climatology"],
            linestyle="-",
            linewidth=2,
            label="Climatology",
        )
    else:
        for model_name in model_order:
            if model_name not in metric_data:
                continue
            ax.plot(
                latitudes,
                metric_data[model_name],
                label=model_name,
                color=model_color_map[model_name],
                linewidth=2,
            )

    ax.set_ylabel(ylabel, fontsize=_FONT_BASE)
    ax.set_xlabel("Latitude (degrees)", fontsize=_FONT_BASE)
    if panel_label is not None:
        ax.set_title(
            f"{panel_label}) {metric_title}",
            fontsize=_FONT_TITLE,
            loc="left",
        )
    else:
        ax.set_title(metric_title, fontsize=_FONT_TITLE)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=_FONT_TICK)

    if ylim is not None:
        ax.set_ylim(ylim)

    if not plot_skill_score:
        if metric_name == "acc":
            ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        elif metric_name in ("cma", "cid"):
            ax.axhline(0.5, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
    ax.set_xticklabels(
        [
            "80°S",
            "60°S",
            "40°S",
            "20°S",
            "Equator",
            "20°N",
            "40°N",
            "60°N",
            "80°N",
        ]
    )


def _center_bottom_panel(
    ax_bottom: plt.Axes, ax_row_left: plt.Axes, ax_row_right: plt.Axes
) -> None:
    """After ``tight_layout``, slide SEEPS to the row center (unchanged size)."""
    pos = ax_bottom.get_position()
    ref = ax_row_left.get_position()
    row_center = 0.5 * (ax_row_left.get_position().x0 + ax_row_right.get_position().x1)
    ax_bottom.set_position([row_center - ref.width / 2, pos.y0, ref.width, pos.height])


def _add_legend_beside_seeps(seeps_ax: plt.Axes) -> None:
    """Legend immediately right of SEEPS, outside the data area."""
    handles, labels = seeps_ax.get_legend_handles_labels()
    if not handles:
        return
    seeps_ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        bbox_transform=seeps_ax.transAxes,
        borderaxespad=0.0,
        frameon=True,
        fontsize=_FONT_LEGEND,
    )


def plot_precipitation_metrics(
    input_dir: str,
    output_dir: str,
    *,
    lead_times=(24,),
    add_panel_labels: bool = False,
    use_skill_scores: bool = False,
) -> None:
    variable = "total_precipitation_24hr"
    model_color_map = _style_context()
    model_order = ["GraphCast", "IFS HRES", "Persistence", "Climatology"]

    metrics, filename_suffix = _metric_configs(use_skill_scores)
    panel_labels = ["a", "b", "c", "d", "e"]

    os.makedirs(output_dir, exist_ok=True)

    for lead_time in lead_times:
        print(f"\nGenerating precipitation plot for {lead_time}h lead time...")

        fig, axes = plt.subplots(3, 2, figsize=(22, 20))
        axes_by_metric = {
            "cma": axes[0, 0],
            "cid": axes[0, 1],
            "seeps": axes[1, 0],
            "rmse": axes[2, 0],
            "acc": axes[2, 1],
        }
        axes[1, 1].set_visible(False)

        latitudes: np.ndarray | None = None

        for idx, metric_info in enumerate(metrics):
            metric_name, metric_title, ylabel, is_positive_oriented, ylim = metric_info
            ax = axes_by_metric[metric_name]

            metric_data = load_metric_data(input_dir, metric_name, variable, lead_time)
            if not metric_data:
                print(f"  Warning: No data found for {metric_name} at {lead_time}h")
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=_FONT_BASE,
                )
                continue

            if latitudes is None:
                num_lats = len(next(iter(metric_data.values())))
                latitudes = np.linspace(-90, 90, num_lats)

            _draw_metric_panel(
                ax,
                metric_name=metric_name,
                metric_title=metric_title,
                ylabel=ylabel,
                is_positive_oriented=is_positive_oriented,
                ylim=ylim,
                metric_data=metric_data,
                latitudes=latitudes,
                model_order=model_order,
                model_color_map=model_color_map,
                use_skill_scores=use_skill_scores,
                panel_label=panel_labels[idx] if add_panel_labels else None,
            )

        plt.tight_layout()
        seeps_ax = axes_by_metric.get("seeps")
        if seeps_ax is not None:
            _center_bottom_panel(seeps_ax, axes[0, 0], axes[0, 1])
            _add_legend_beside_seeps(seeps_ax)

        output_file_pdf = os.path.join(
            output_dir,
            f"precipitation_{filename_suffix}_lead{lead_time}h.pdf",
        )
        plt.savefig(output_file_pdf, bbox_inches="tight", dpi=300)
        print(f"  Saved PDF: {output_file_pdf}")
        plt.close()

    print("\nAll precipitation plots completed!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precipitation metrics (CMA/CID top, SEEPS centered row 2, RMSE/ACC bottom)"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./fct_data",
        help="Base directory containing *_results subdirectories (default: ./fct_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Output directory for plots (default: ./plots)",
    )
    parser.add_argument(
        "--lead_times",
        type=int,
        nargs="+",
        default=[24],
        help="Lead times in hours (default: 24)",
    )
    parser.add_argument(
        "--add_panel_labels",
        action="store_true",
        help="Left-aligned titles with panel prefix, e.g. a) CMA Skill Score",
    )
    parser.add_argument(
        "--skill_scores",
        action="store_true",
        help="Plot skill scores instead of raw metrics",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    plot_precipitation_metrics(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lead_times=args.lead_times,
        add_panel_labels=args.add_panel_labels,
        use_skill_scores=args.skill_scores,
    )


if __name__ == "__main__":
    main()
