import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

matplotlib.use("Agg")

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(to_builtin(data), f, indent=2)


def plot_learning_curve(history: pd.DataFrame, metric: str, output_path: Path) -> bool:
    if metric not in history.columns:
        return False
    x = history.get("_step", pd.Series(range(len(history))))
    y = history[metric].astype(float)
    if y.dropna().empty:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    y_clean = y.dropna()
    x_clean = x[y.notna()]

    sns.lineplot(x=x_clean, y=y_clean, marker="o", markersize=8, linewidth=2.5, ax=ax)
    ax.set_title(
        f"{metric.split('/')[-1].replace('_', ' ').title()} Over Training",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        metric.split("/")[-1].replace("_", " ").title(), fontsize=14, fontweight="bold"
    )

    # Set appropriate y-axis limits based on data range
    y_min, y_max = y_clean.min(), y_clean.max()
    y_range = y_max - y_min
    if y_range < 0.01:  # Very small range, likely all zeros or near-constant
        ax.set_ylim(-0.05, max(0.5, y_max + 0.05))
    else:
        margin = y_range * 0.15
        ax.set_ylim(max(0, y_min - margin), min(1.0, y_max + margin))

    # Add final value annotation
    final_val = y_clean.iloc[-1]
    final_x = x_clean.iloc[-1]
    ax.annotate(
        f"Final: {final_val:.3f}",
        xy=(final_x, final_val),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5),
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_domain_bars(summary: Dict[str, Any], output_path: Path) -> bool:
    domain_specs = {
        "SST-2": ["eval/sst2_accuracy_mean", "eval/sst2_accuracy"],
        "Yelp": ["eval/yelp_accuracy_mean", "eval/yelp_accuracy"],
        "IMDB": ["eval/imdb_accuracy_mean", "eval/imdb_accuracy"],
    }
    values = []
    domains = []
    for domain, keys in domain_specs.items():
        val = None
        for key in keys:
            if key in summary:
                val = summary.get(key)
                break
        if val is not None:
            domains.append(domain)
            values.append(float(val))
    if not values:
        return False

    data = pd.DataFrame({"domain": domains, "accuracy": values})
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = sns.barplot(x="domain", y="accuracy", data=data, ax=ax, palette="Set2")

    # Set appropriate y-axis limits
    max_val = max(values)
    if max_val < 0.1:  # All values very small
        ax.set_ylim(0, 0.5)
    else:
        ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for i, (v, bar) in enumerate(zip(values, bars.patches)):
        height = bar.get_height()
        label_y = height + 0.02 if height > 0.05 else 0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_title(
        "Per-Domain Constrained Accuracy", fontsize=16, fontweight="bold", pad=15
    )
    ax.set_xlabel("Domain", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_gate_failures(summary: Dict[str, Any], output_path: Path) -> bool:
    gate_keys = [k for k in summary.keys() if k.startswith("gate_fail/")]
    if not gate_keys:
        return False
    data = {
        "gate": [k.split("/")[-1].replace("_", " ").title() for k in gate_keys],
        "count": [summary[k] for k in gate_keys],
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = sns.barplot(
        x="gate", y="count", data=pd.DataFrame(data), ax=ax, palette="viridis"
    )

    for i, v in enumerate(data["count"]):
        ax.text(
            i,
            v + max(data["count"]) * 0.02,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(
        "First Failed Gate Distribution", fontsize=16, fontweight="bold", pad=15
    )
    ax.set_xlabel("Gate Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_confusion_matrix(
    summary: Dict[str, Any], domain: str, output_path: Path
) -> bool:
    keys = {
        "tp": summary.get(f"confusion/{domain}_tp"),
        "tn": summary.get(f"confusion/{domain}_tn"),
        "fp": summary.get(f"confusion/{domain}_fp"),
        "fn": summary.get(f"confusion/{domain}_fn"),
    }
    if any(v is None for v in keys.values()):
        return False

    matrix = np.array([[keys["tn"], keys["fp"]], [keys["fn"], keys["tp"]]], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "Count"},
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=2,
        linecolor="white",
    )

    ax.set_title(
        f"Confusion Matrix - {domain.upper()}", fontsize=16, fontweight="bold", pad=15
    )
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def primary_metric_name() -> str:
    return "eval/robust_constrained_accuracy_mean"


def is_minimization_metric(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(token in lowered for token in ["loss", "perplexity", "error"])


def extract_primary_metric(
    summary: Dict[str, Any], history: pd.DataFrame
) -> Tuple[str, Optional[float]]:
    primary = primary_metric_name()
    if primary in summary:
        return primary, float(summary[primary])
    fallback = "eval/robust_constrained_accuracy"
    if fallback in summary:
        return fallback, float(summary[fallback])
    if fallback in history.columns and len(history[fallback].dropna()) > 0:
        return fallback, float(history[fallback].dropna().iloc[-1])
    return primary, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--run_ids", type=str, default=None)
    parser.add_argument("results_dir_pos", type=str, nargs="?", default=None)
    parser.add_argument("run_ids_pos", type=str, nargs="?", default=None)
    args = parser.parse_args()

    # Handle both positional and named arguments
    results_dir_arg = args.results_dir or args.results_dir_pos
    run_ids_arg = args.run_ids or args.run_ids_pos

    if not results_dir_arg or not run_ids_arg:
        parser.error("Both results_dir and run_ids are required")

    results_dir = Path(results_dir_arg)
    ensure_dir(results_dir)

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()
    run_ids = json.loads(run_ids_arg)

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    run_primary_scores: Dict[str, float] = {}
    generated_paths: List[Path] = []
    per_run_seed_scores: Dict[str, List[float]] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history(samples=100000)
        summary = run.summary._json_dict
        config = dict(run.config)

        run_dir = results_dir / run_id
        ensure_dir(run_dir)

        metrics_path = run_dir / "metrics.json"
        save_json(
            metrics_path,
            {
                "history": history.to_dict(orient="list"),
                "summary": summary,
                "config": config,
            },
        )
        generated_paths.append(metrics_path)

        curve_path = run_dir / f"{run_id}_learning_curve_robust.pdf"
        if plot_learning_curve(history, "eval/robust_constrained_accuracy", curve_path):
            generated_paths.append(curve_path)

        worst_curve_path = run_dir / f"{run_id}_learning_curve_worst_slice.pdf"
        if plot_learning_curve(
            history, "eval/worst_slice_constrained_accuracy", worst_curve_path
        ):
            generated_paths.append(worst_curve_path)

        domain_path = run_dir / f"{run_id}_domain_accuracy.pdf"
        if plot_domain_bars(summary, domain_path):
            generated_paths.append(domain_path)

        gate_path = run_dir / f"{run_id}_gate_failures.pdf"
        if plot_gate_failures(summary, gate_path):
            generated_paths.append(gate_path)

        for domain in ["sst2", "yelp", "imdb"]:
            conf_path = run_dir / f"{run_id}_confusion_{domain}.pdf"
            if plot_confusion_matrix(summary, domain, conf_path):
                generated_paths.append(conf_path)

        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                aggregated_metrics.setdefault(key, {})[run_id] = float(value)

        primary_key, primary_value = extract_primary_metric(summary, history)
        if primary_value is not None:
            run_primary_scores[run_id] = float(primary_value)

        if (
            "eval/robust_constrained_accuracy" in history.columns
            and "seed" in history.columns
        ):
            seed_scores = (
                history[["eval/robust_constrained_accuracy", "seed"]]
                .dropna()
                .groupby("seed")
                .mean()["eval/robust_constrained_accuracy"]
                .tolist()
            )
            if seed_scores:
                per_run_seed_scores[run_id] = seed_scores

    comparison_dir = results_dir / "comparison"
    ensure_dir(comparison_dir)

    proposed_scores = {k: v for k, v in run_primary_scores.items() if "proposed" in k}
    baseline_scores = {
        k: v
        for k, v in run_primary_scores.items()
        if "comparative" in k or "baseline" in k
    }

    maximize = not is_minimization_metric(primary_metric_name())

    if proposed_scores:
        best_proposed_run = (
            max(proposed_scores.items(), key=lambda kv: kv[1])
            if maximize
            else min(proposed_scores.items(), key=lambda kv: kv[1])
        )
    else:
        best_proposed_run = (None, float("nan"))

    if baseline_scores:
        best_baseline_run = (
            max(baseline_scores.items(), key=lambda kv: kv[1])
            if maximize
            else min(baseline_scores.items(), key=lambda kv: kv[1])
        )
    else:
        best_baseline_run = (None, float("nan"))

    gap = float("nan")
    if best_proposed_run[0] and best_baseline_run[0]:
        base_value = best_baseline_run[1]
        if base_value != 0:
            raw_gap = (best_proposed_run[1] - base_value) / base_value * 100
            gap = -raw_gap if is_minimization_metric(primary_metric_name()) else raw_gap

    aggregated_path = comparison_dir / "aggregated_metrics.json"
    save_json(
        aggregated_path,
        {
            "primary_metric": "accuracy",
            "metrics": aggregated_metrics,
            "best_proposed": {
                "run_id": best_proposed_run[0],
                "value": best_proposed_run[1],
            },
            "best_baseline": {
                "run_id": best_baseline_run[0],
                "value": best_baseline_run[1],
            },
            "gap": gap,
        },
    )
    generated_paths.append(aggregated_path)

    if run_primary_scores:
        fig, ax = plt.subplots(figsize=(12, 7))
        df = pd.DataFrame(
            {
                "run_id": list(run_primary_scores.keys()),
                "score": list(run_primary_scores.values()),
            }
        )

        # Shorten run_id labels for readability
        df["label"] = df["run_id"].apply(
            lambda x: x.replace("comparative-1-", "baseline-").replace(
                "proposed-", "proposed-"
            )
        )

        # Use color to distinguish baseline vs proposed
        colors = [
            "#1f77b4" if "comparative" in rid or "baseline" in rid else "#ff7f0e"
            for rid in df["run_id"]
        ]

        bars = ax.bar(
            range(len(df)),
            df["score"],
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Set appropriate y-axis limits
        max_score = df["score"].max()
        if max_score < 0.1:
            ax.set_ylim(0, 0.5)
        else:
            ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for i, (idx, row) in enumerate(df.iterrows()):
            score = row["score"]
            label_y = score + 0.02 if score > 0.05 else 0.05
            ax.text(
                i,
                label_y,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=11)
        ax.set_title(
            "Primary Metric Comparison Across All Runs",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Run ID", fontsize=14, fontweight="bold")
        ax.set_ylabel("Robust Constrained Accuracy", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", edgecolor="black", label="Baseline", alpha=0.8),
            Patch(facecolor="#ff7f0e", edgecolor="black", label="Proposed", alpha=0.8),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.9
        )

        plt.tight_layout()
        bar_path = comparison_dir / "comparison_accuracy_bar_chart.pdf"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        generated_paths.append(bar_path)

    if per_run_seed_scores:
        fig, ax = plt.subplots(figsize=(12, 7))
        rows = []
        for run_id, scores in per_run_seed_scores.items():
            label = run_id.replace("comparative-1-", "baseline-").replace(
                "proposed-", "proposed-"
            )
            for score in scores:
                rows.append({"run_id": run_id, "label": label, "score": score})
        df = pd.DataFrame(rows)

        # Create color palette
        unique_runs = df["run_id"].unique()
        palette = [
            "#1f77b4" if "comparative" in rid or "baseline" in rid else "#ff7f0e"
            for rid in unique_runs
        ]

        sns.boxplot(
            x="label",
            y="score",
            data=df,
            ax=ax,
            palette=palette,
            linewidth=2,
            fliersize=8,
        )

        # Set appropriate y-axis limits
        max_score = df["score"].max()
        if max_score < 0.1:
            ax.set_ylim(-0.05, 0.5)
        else:
            ax.set_ylim(
                max(0, df["score"].min() - 0.1), min(1.0, df["score"].max() + 0.1)
            )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
        ax.set_title(
            "Seed-Level Robust Accuracy Distribution",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Run ID", fontsize=14, fontweight="bold")
        ax.set_ylabel("Robust Constrained Accuracy", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", edgecolor="black", label="Baseline", alpha=0.8),
            Patch(facecolor="#ff7f0e", edgecolor="black", label="Proposed", alpha=0.8),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.9
        )

        plt.tight_layout()
        box_path = comparison_dir / "comparison_accuracy_box_plot.pdf"
        plt.savefig(box_path, dpi=300, bbox_inches="tight")
        plt.close()
        generated_paths.append(box_path)

    if per_run_seed_scores and proposed_scores and baseline_scores:
        proposed_values = []
        baseline_values = []
        for run_id, scores in per_run_seed_scores.items():
            if run_id in proposed_scores:
                proposed_values.extend(scores)
            if run_id in baseline_scores:
                baseline_values.extend(scores)

        # Only create plot if we have sufficient data for statistical test
        if (
            proposed_values
            and baseline_values
            and len(proposed_values) > 1
            and len(baseline_values) > 1
        ):
            # Check if there's any variance
            prop_std = np.std(proposed_values)
            base_std = np.std(baseline_values)

            if prop_std > 1e-10 or base_std > 1e-10:  # Some variance exists
                t_stat, p_val = stats.ttest_ind(
                    proposed_values, baseline_values, equal_var=False
                )

                fig, ax = plt.subplots(figsize=(10, 7))
                ax.axis("off")

                # Create nice formatted text
                text_content = [
                    f"Statistical Significance Test",
                    f"",
                    f"Test: Welch's t-test (unequal variances)",
                    f"",
                    f"Proposed Method:",
                    f"  • Sample size: {len(proposed_values)}",
                    f"  • Mean: {np.mean(proposed_values):.4f}",
                    f"  • Std Dev: {np.std(proposed_values):.4f}",
                    f"",
                    f"Baseline Method:",
                    f"  • Sample size: {len(baseline_values)}",
                    f"  • Mean: {np.mean(baseline_values):.4f}",
                    f"  • Std Dev: {np.std(baseline_values):.4f}",
                    f"",
                    f"Results:",
                    f"  • t-statistic: {t_stat:.4f}",
                    f"  • p-value: {p_val:.4e}",
                    f"",
                ]

                if p_val < 0.001:
                    text_content.append("  ✓ Highly significant (p < 0.001)")
                elif p_val < 0.01:
                    text_content.append("  ✓ Very significant (p < 0.01)")
                elif p_val < 0.05:
                    text_content.append("  ✓ Significant (p < 0.05)")
                else:
                    text_content.append("  ✗ Not significant (p ≥ 0.05)")

                y_pos = 0.95
                for line in text_content:
                    if (
                        line.startswith("Statistical")
                        or line.startswith("Test:")
                        or line.startswith("Results:")
                    ):
                        weight = "bold"
                        size = 14
                    elif line.startswith("Proposed") or line.startswith("Baseline"):
                        weight = "bold"
                        size = 12
                    else:
                        weight = "normal"
                        size = 11

                    ax.text(
                        0.1,
                        y_pos,
                        line,
                        fontsize=size,
                        fontweight=weight,
                        verticalalignment="top",
                        family="monospace",
                    )
                    y_pos -= 0.045

                plt.tight_layout()
                sig_path = comparison_dir / "comparison_significance_test.pdf"
                plt.savefig(sig_path, dpi=300, bbox_inches="tight")
                plt.close()
                generated_paths.append(sig_path)

    table_metrics = [
        "eval/robust_constrained_accuracy_mean",
        "eval/worst_slice_constrained_accuracy_mean",
        "eval/ood_regression_rate",
        "train/accepted_steps_mean",
        "train/candidates_evaluated_mean",
    ]
    table_data = {
        k: aggregated_metrics[k] for k in table_metrics if k in aggregated_metrics
    }
    if table_data:
        table_df = pd.DataFrame(table_data).T

        # Shorten column names for better readability
        col_mapping = {
            "comparative-1-distilbert-sst2-imdb": "Baseline\nDistilBERT",
            "comparative-1-flan-t5-small-imdb": "Baseline\nFlan-T5",
            "proposed-distilbert-sst2-imdb": "Proposed\nDistilBERT",
            "proposed-flan-t5-small-imdb": "Proposed\nFlan-T5",
        }
        table_df.columns = [col_mapping.get(c, c) for c in table_df.columns]

        # Shorten row names
        row_mapping = {
            "eval/robust_constrained_accuracy_mean": "Robust Accuracy",
            "eval/worst_slice_constrained_accuracy_mean": "Worst Slice Accuracy",
            "eval/ood_regression_rate": "OOD Regression Rate",
            "train/accepted_steps_mean": "Accepted Steps",
            "train/candidates_evaluated_mean": "Candidates Evaluated",
        }
        table_df.index = [row_mapping.get(idx, idx) for idx in table_df.index]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis("off")

        # Format cell values
        cell_text = [
            [f"{val:.4f}" if isinstance(val, (int, float)) else str(val) for val in row]
            for row in table_df.values
        ]

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=table_df.index,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
            rowLoc="left",
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 2.5)

        # Style the header row
        for i in range(len(table_df.columns)):
            cell = tbl[(0, i)]
            cell.set_facecolor("#4CAF50")
            cell.set_text_props(weight="bold", color="white", fontsize=12)

        # Style the row labels
        for i in range(len(table_df.index)):
            cell = tbl[(i + 1, -1)]
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold", fontsize=11)

        # Alternate row colors for better readability
        for i in range(len(table_df.index)):
            for j in range(len(table_df.columns)):
                cell = tbl[(i + 1, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#F5F5F5")
                else:
                    cell.set_facecolor("white")

        ax.set_title(
            "Aggregated Metrics Comparison", fontsize=18, fontweight="bold", pad=20
        )

        plt.tight_layout()
        table_path = comparison_dir / "comparison_metrics_table.pdf"
        plt.savefig(table_path, dpi=300, bbox_inches="tight")
        plt.close()
        generated_paths.append(table_path)

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
