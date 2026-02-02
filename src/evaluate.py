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
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=x, y=y, marker="o")
    plt.title(f"{metric} over time")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.annotate(f"final={y.dropna().iloc[-1]:.3f}", (x.iloc[-1], y.dropna().iloc[-1]))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def plot_domain_bars(summary: Dict[str, Any], output_path: Path) -> bool:
    domain_specs = {
        "sst2": ["eval/sst2_accuracy_mean", "eval/sst2_accuracy"],
        "yelp": ["eval/yelp_accuracy_mean", "eval/yelp_accuracy"],
        "imdb": ["eval/imdb_accuracy_mean", "eval/imdb_accuracy"],
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
    data = {"domain": domains, "accuracy": values}
    plt.figure(figsize=(6, 4))
    sns.barplot(x="domain", y="accuracy", data=pd.DataFrame(data))
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.title("Per-domain constrained accuracy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def plot_gate_failures(summary: Dict[str, Any], output_path: Path) -> bool:
    gate_keys = [k for k in summary.keys() if k.startswith("gate_fail/")]
    if not gate_keys:
        return False
    data = {"gate": [k.split("/")[-1] for k in gate_keys], "count": [summary[k] for k in gate_keys]}
    plt.figure(figsize=(6, 4))
    sns.barplot(x="gate", y="count", data=pd.DataFrame(data))
    for i, v in enumerate(data["count"]):
        plt.text(i, v + 0.1, f"{v:.0f}", ha="center")
    plt.title("First failed gate distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def plot_confusion_matrix(summary: Dict[str, Any], domain: str, output_path: Path) -> bool:
    keys = {
        "tp": summary.get(f"confusion/{domain}_tp"),
        "tn": summary.get(f"confusion/{domain}_tn"),
        "fp": summary.get(f"confusion/{domain}_fp"),
        "fn": summary.get(f"confusion/{domain}_fn"),
    }
    if any(v is None for v in keys.values()):
        return False
    matrix = np.array([[keys["tn"], keys["fp"]], [keys["fn"], keys["tp"]]], dtype=float)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        xticklabels=["neg", "pos"],
        yticklabels=["neg", "pos"],
    )
    plt.title(f"Confusion matrix ({domain})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def primary_metric_name() -> str:
    return "eval/robust_constrained_accuracy_mean"


def is_minimization_metric(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(token in lowered for token in ["loss", "perplexity", "error"])


def extract_primary_metric(summary: Dict[str, Any], history: pd.DataFrame) -> Tuple[str, Optional[float]]:
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
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()
    run_ids = json.loads(args.run_ids)

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
        if plot_learning_curve(history, "eval/worst_slice_constrained_accuracy", worst_curve_path):
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

        if "eval/robust_constrained_accuracy" in history.columns and "seed" in history.columns:
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
        k: v for k, v in run_primary_scores.items() if "comparative" in k or "baseline" in k
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
            "best_proposed": {"run_id": best_proposed_run[0], "value": best_proposed_run[1]},
            "best_baseline": {"run_id": best_baseline_run[0], "value": best_baseline_run[1]},
            "gap": gap,
        },
    )
    generated_paths.append(aggregated_path)

    if run_primary_scores:
        plt.figure(figsize=(8, 4))
        df = pd.DataFrame({"run_id": list(run_primary_scores.keys()), "score": list(run_primary_scores.values())})
        sns.barplot(x="run_id", y="score", data=df)
        plt.xticks(rotation=30, ha="right")
        for i, row in df.iterrows():
            plt.text(i, row["score"] + 0.01, f"{row['score']:.3f}", ha="center")
        plt.title("Primary metric comparison")
        plt.tight_layout()
        bar_path = comparison_dir / "comparison_accuracy_bar_chart.pdf"
        plt.savefig(bar_path)
        plt.close()
        generated_paths.append(bar_path)

    if per_run_seed_scores:
        plt.figure(figsize=(8, 4))
        rows = []
        for run_id, scores in per_run_seed_scores.items():
            for score in scores:
                rows.append({"run_id": run_id, "score": score})
        df = pd.DataFrame(rows)
        sns.boxplot(x="run_id", y="score", data=df)
        plt.xticks(rotation=30, ha="right")
        plt.title("Seed-level robust accuracy distribution")
        plt.tight_layout()
        box_path = comparison_dir / "comparison_accuracy_box_plot.pdf"
        plt.savefig(box_path)
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
        if proposed_values and baseline_values:
            t_stat, p_val = stats.ttest_ind(proposed_values, baseline_values, equal_var=False)
            plt.figure(figsize=(6, 4))
            plt.axis("off")
            plt.text(0.1, 0.6, f"t-statistic: {t_stat:.3f}")
            plt.text(0.1, 0.4, f"p-value: {p_val:.3e}")
            plt.title("Proposed vs Baseline significance")
            plt.tight_layout()
            sig_path = comparison_dir / "comparison_significance_test.pdf"
            plt.savefig(sig_path)
            plt.close()
            generated_paths.append(sig_path)

    table_metrics = [
        "eval/robust_constrained_accuracy_mean",
        "eval/worst_slice_constrained_accuracy_mean",
        "eval/ood_regression_rate",
        "train/accepted_steps_mean",
        "train/candidates_evaluated_mean",
    ]
    table_data = {k: aggregated_metrics[k] for k in table_metrics if k in aggregated_metrics}
    if table_data:
        table_df = pd.DataFrame(table_data).T
        plt.figure(figsize=(10, 4))
        plt.axis("off")
        tbl = plt.table(
            cellText=table_df.values,
            rowLabels=table_df.index,
            colLabels=table_df.columns,
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        plt.tight_layout()
        table_path = comparison_dir / "comparison_metrics_table.pdf"
        plt.savefig(table_path)
        plt.close()
        generated_paths.append(table_path)

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
