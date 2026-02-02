import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model as model_lib
import preprocess


def set_cache_env(cache_dir: str) -> None:
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def apply_mode_overrides(cfg: Any) -> Any:
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "optuna") or cfg.optuna is None:
        cfg.optuna = OmegaConf.create({"n_trials": 0, "search_spaces": []})
    if not hasattr(cfg.training, "max_batches"):
        cfg.training.max_batches = None
    if not hasattr(cfg.training, "max_eval_batches"):
        cfg.training.max_eval_batches = None
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.batch_size = min(cfg.training.batch_size, 8)
        cfg.training.prompt_optimization.iterations = min(
            1, int(cfg.training.prompt_optimization.iterations)
        )
        cfg.training.prompt_optimization.candidates_per_iter = min(
            2, int(cfg.training.prompt_optimization.candidates_per_iter)
        )
        cfg.training.max_batches = 2
        cfg.training.max_eval_batches = 2
        max_examples = cfg.training.batch_size * cfg.training.max_eval_batches
        cfg.dataset.splits.opt_size = min(cfg.dataset.splits.opt_size, max_examples)
        cfg.dataset.splits.audit_total = min(cfg.dataset.splits.audit_total, max_examples)
        cfg.dataset.splits.audit_disc = cfg.dataset.splits.audit_total // 2
        cfg.dataset.splits.audit_conf = (
            cfg.dataset.splits.audit_total - cfg.dataset.splits.audit_disc
        )
        cfg.dataset.splits.eval_size = min(cfg.dataset.splits.eval_size, max_examples)
        if hasattr(cfg.training, "prompt_optimization"):
            cfg.training.prompt_optimization.discovery_size = cfg.dataset.splits.audit_disc
            cfg.training.prompt_optimization.confirmation_size = cfg.dataset.splits.audit_conf
        if isinstance(cfg.seeds, list) and cfg.seeds:
            cfg.seeds = [cfg.seeds[0]]
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return cfg


def normalize_prompt_cfg(cfg: Any) -> float:
    prompt_cfg = cfg.training.prompt_optimization
    if hasattr(prompt_cfg, "pi_fluency_nll") and prompt_cfg.pi_fluency_nll is not None:
        nll_max = float(prompt_cfg.pi_fluency_nll)
    elif hasattr(prompt_cfg, "ppl_max") and prompt_cfg.ppl_max is not None:
        nll_max = float(math.log(max(float(prompt_cfg.ppl_max), 1e-12)))
    else:
        nll_max = 4.0
    prompt_cfg.nll_max = float(nll_max)
    prompt_cfg.pi_fluency_nll = float(nll_max)
    prompt_cfg.ppl_max = float(math.exp(nll_max))
    return float(nll_max)


def build_wandb(cfg: Any) -> Optional[Any]:
    if cfg.wandb.mode == "disabled":
        return None
    run_id = getattr(cfg, "run_id", None) or getattr(cfg, "run", None)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
        mode=cfg.wandb.mode,
    )
    return run


def log_metrics(data: Dict[str, Any], step: Optional[int], run: Optional[Any]) -> None:
    if run is None:
        return
    wandb.log(data, step=step)


def assert_nonzero_gradients(model: nn.Module) -> None:
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads, "No trainable parameters found."
    assert all(g is not None for g in grads), "Missing gradients detected."
    grad_norm = sum(float(g.detach().abs().sum().item()) for g in grads)
    assert grad_norm > 0.0, "Gradients are zero; optimizer step would be ineffective."


def train_auxiliary_predictor(
    evaluator: model_lib.ConstrainedEvaluator,
    d_opt: List[Dict[str, Any]],
    base_prompt: str,
    cfg: Any,
    device: torch.device,
    run: Optional[Any],
    global_step: int,
) -> Tuple[model_lib.AuxiliarySuccessPredictor, int]:
    tau = float(cfg.training.prompt_optimization.tau_similarity)
    nll_max = float(cfg.training.prompt_optimization.nll_max)
    features = []
    labels = []
    for ex in d_opt:
        result = evaluator.constrained_success(
            base_prompt, ex["text"], ex["label"], tau, nll_max
        )
        labels.append(result["success"])
        emb = evaluator.embedding(ex["text"]).detach().cpu()
        features.append(emb)
    if not features:
        raise RuntimeError("No features available for auxiliary predictor training.")

    X = torch.stack(features).to(device)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1).to(device)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    model = model_lib.AuxiliarySuccessPredictor(input_dim=X.shape[-1]).to(device)
    model.train()

    optimizer_cls = torch.optim.AdamW if cfg.training.optimizer.lower() == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    max_batches = cfg.training.max_batches

    for epoch in range(cfg.training.epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            if epoch == 0 and step == 0:
                assert batch_x.shape[0] == batch_y.shape[0], "Input/label batch mismatch."
                assert batch_y.ndim == 2, "Labels must be 2D for BCEWithLogitsLoss."
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            grads = torch.autograd.grad(
                loss, [p for p in model.parameters() if p.requires_grad], create_graph=False
            )
            for param, grad in zip(model.parameters(), grads):
                param.grad = grad
            assert_nonzero_gradients(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log_metrics(
                {"aux/train_loss": float(loss.item()), "aux/epoch": epoch},
                step=global_step,
                run=run,
            )
            global_step += 1
    model.eval()
    return model, global_step


def setup_models(cfg: Any, device: torch.device) -> model_lib.ConstrainedEvaluator:
    rewriter_name = cfg.models.rewriter_name
    sentiment_name = cfg.models.sentiment_name
    if cfg.model.role == "rewriter":
        rewriter_name = cfg.model.name
    elif cfg.model.role == "sentiment_evaluator":
        sentiment_name = cfg.model.name

    max_new_tokens = int(getattr(cfg.model, "max_new_tokens", 80))

    evaluator = model_lib.ConstrainedEvaluator(
        rewriter_name=rewriter_name,
        sentiment_name=sentiment_name,
        similarity_name=cfg.models.similarity_name,
        fluency_name=cfg.models.fluency_name,
        device=device,
        max_new_tokens=max_new_tokens,
        cache_dir=cfg.cache_dir,
    )

    assert evaluator.rewriter.tokenizer.pad_token_id is not None, "Rewriter pad_token_id missing."
    assert evaluator.sentiment.model.config.num_labels >= 2, "Sentiment model has invalid label count."
    assert evaluator.rewriter.model.config.vocab_size > 0, "Rewriter vocab size invalid."
    assert evaluator.rewriter.max_new_tokens > 0, "Rewriter max_new_tokens must be positive."
    assert evaluator.fluency.tokenizer.pad_token_id is not None, "Fluency tokenizer pad_token_id missing."
    return evaluator


def is_regression(final_value: float, base_value: float, epsilon: float) -> bool:
    return (
        math.isfinite(final_value)
        and math.isfinite(base_value)
        and final_value < base_value - epsilon
    )


def run_seed(
    seed: int,
    cfg: Any,
    evaluator: model_lib.ConstrainedEvaluator,
    run: Optional[Any],
    global_step: int,
) -> Tuple[Dict[str, Any], Counter, Dict[str, Dict[str, int]], int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    evaluator.clear_cache()
    data = preprocess.prepare_datasets(cfg, seed)
    d_opt = data["d_opt"]
    a_disc = data["audit_disc"]
    a_conf = data["audit_conf"]
    eval_sets = data["eval_sets"]
    eval_probes = data["eval_probes"]

    base_prompt = cfg.prompt.base
    tau = float(cfg.training.prompt_optimization.tau_similarity)
    nll_max = float(cfg.training.prompt_optimization.nll_max)

    _, global_step = train_auxiliary_predictor(
        evaluator, d_opt, base_prompt, cfg, evaluator.device, run, global_step
    )

    max_eval = None
    if cfg.training.max_eval_batches is not None:
        max_eval = cfg.training.batch_size * cfg.training.max_eval_batches

    base_metrics = model_lib.evaluate_prompt(
        base_prompt,
        eval_sets,
        evaluator,
        tau,
        nll_max,
        max_examples=max_eval,
    )

    optimization_result = model_lib.optimize_prompt(
        method_name=cfg.method,
        base_prompt=base_prompt,
        evaluator=evaluator,
        d_opt=d_opt,
        a_disc=a_disc,
        a_conf=a_conf,
        prompt_cfg=cfg.training.prompt_optimization,
        log_fn=lambda data, step: log_metrics(data, step, run),
        global_step=global_step,
        max_gate_examples=max_eval,
    )
    final_prompt = optimization_result["final_prompt"]
    global_step = optimization_result["global_step"]
    gate_fail_counts = optimization_result["gate_fail_counts"]

    final_metrics, confusions = model_lib.evaluate_prompt(
        final_prompt,
        eval_sets,
        evaluator,
        tau,
        nll_max,
        return_confusion=True,
        max_examples=max_eval,
    )

    robust_domains = ["sst2", "yelp", "imdb"]
    robust_acc = min(final_metrics.get(domain, 0.0) for domain in robust_domains)

    combined_eval = model_lib.flatten_eval_sets(eval_sets) + eval_probes
    worst_slice_acc = model_lib.worst_slice_accuracy(
        final_prompt,
        combined_eval,
        evaluator,
        tau,
        nll_max,
        int(cfg.training.prompt_optimization.slice_min_support),
        max_examples=max_eval,
    )

    epsilon = 0.01
    final_yelp = float(final_metrics.get("yelp", math.nan))
    final_imdb = float(final_metrics.get("imdb", math.nan))
    base_yelp = float(base_metrics.get("yelp", math.nan))
    base_imdb = float(base_metrics.get("imdb", math.nan))
    ood_regression = int(
        is_regression(final_yelp, base_yelp, epsilon)
        or is_regression(final_imdb, base_imdb, epsilon)
    )

    log_metrics(
        {
            "seed": seed,
            "eval/robust_constrained_accuracy": robust_acc,
            "eval/worst_slice_constrained_accuracy": worst_slice_acc,
            "eval/ood_regression": ood_regression,
            "eval/sst2_accuracy": final_metrics.get("sst2", math.nan),
            "eval/yelp_accuracy": final_metrics.get("yelp", math.nan),
            "eval/imdb_accuracy": final_metrics.get("imdb", math.nan),
            "eval/base_sst2_accuracy": base_metrics.get("sst2", math.nan),
            "eval/base_yelp_accuracy": base_metrics.get("yelp", math.nan),
            "eval/base_imdb_accuracy": base_metrics.get("imdb", math.nan),
            "train/accepted_steps": optimization_result["accepted_steps"],
            "train/candidates_evaluated": optimization_result["candidates_evaluated"],
        },
        step=global_step,
        run=run,
    )
    global_step += 1

    metrics = {
        "seed": seed,
        "robust_constrained_accuracy": robust_acc,
        "worst_slice_constrained_accuracy": worst_slice_acc,
        "ood_regression": ood_regression,
        "per_domain": final_metrics,
        "base_domain": base_metrics,
        "accepted_steps": optimization_result["accepted_steps"],
        "candidates_evaluated": optimization_result["candidates_evaluated"],
        "final_prompt": final_prompt,
    }
    return metrics, gate_fail_counts, confusions, global_step


def run_optuna(cfg: Any, evaluator: model_lib.ConstrainedEvaluator) -> Dict[str, Any]:
    if not hasattr(cfg, "optuna") or cfg.optuna is None or cfg.optuna.n_trials <= 0:
        return {}
    if not getattr(cfg.optuna, "search_spaces", None):
        return {}
    import optuna

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.set_struct(trial_cfg, False)
        for space in trial_cfg.optuna.search_spaces:
            name = space.param_name
            if space.distribution_type == "uniform":
                value = trial.suggest_float(name, float(space.low), float(space.high))
            elif space.distribution_type == "loguniform":
                value = trial.suggest_float(name, float(space.low), float(space.high), log=True)
            elif space.distribution_type == "int":
                value = trial.suggest_int(name, int(space.low), int(space.high))
            elif space.distribution_type == "categorical":
                value = trial.suggest_categorical(name, list(space.choices))
            else:
                raise ValueError(f"Unknown distribution: {space.distribution_type}")
            if name == "pi_fluency_nll":
                trial_cfg.training.prompt_optimization.pi_fluency_nll = value
            else:
                trial_cfg.training.prompt_optimization[name] = value

        trial_cfg.training.prompt_optimization.iterations = 1
        trial_cfg.training.prompt_optimization.candidates_per_iter = 2
        trial_cfg.training.max_eval_batches = 1
        trial_cfg.training.batch_size = min(trial_cfg.training.batch_size, 8)
        trial_cfg.dataset.splits.opt_size = min(trial_cfg.dataset.splits.opt_size, 40)
        trial_cfg.dataset.splits.audit_total = min(trial_cfg.dataset.splits.audit_total, 80)
        trial_cfg.dataset.splits.audit_disc = trial_cfg.dataset.splits.audit_total // 2
        trial_cfg.dataset.splits.audit_conf = (
            trial_cfg.dataset.splits.audit_total - trial_cfg.dataset.splits.audit_disc
        )
        trial_cfg.dataset.splits.eval_size = min(trial_cfg.dataset.splits.eval_size, 80)
        trial_cfg.training.prompt_optimization.slice_min_support = min(
            int(trial_cfg.training.prompt_optimization.slice_min_support), 4
        )
        trial_cfg.seeds = [0]
        trial_cfg.wandb.mode = "disabled"

        normalize_prompt_cfg(trial_cfg)

        evaluator.clear_cache()
        data = preprocess.prepare_datasets(trial_cfg, seed=0)
        d_opt = data["d_opt"]
        a_disc = data["audit_disc"]
        a_conf = data["audit_conf"]
        eval_sets = data["eval_sets"]

        result = model_lib.optimize_prompt(
            method_name=trial_cfg.method,
            base_prompt=trial_cfg.prompt.base,
            evaluator=evaluator,
            d_opt=d_opt,
            a_disc=a_disc,
            a_conf=a_conf,
            prompt_cfg=trial_cfg.training.prompt_optimization,
            log_fn=None,
            global_step=0,
            max_gate_examples=trial_cfg.training.batch_size * trial_cfg.training.max_eval_batches,
        )
        final_prompt = result["final_prompt"]
        final_metrics = model_lib.evaluate_prompt(
            final_prompt,
            eval_sets,
            evaluator,
            trial_cfg.training.prompt_optimization.tau_similarity,
            trial_cfg.training.prompt_optimization.nll_max,
        )
        return float(
            min(
                final_metrics.get("sst2", 0.0),
                final_metrics.get("yelp", 0.0),
                final_metrics.get("imdb", 0.0),
            )
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.optuna.n_trials, n_jobs=1)
    return study.best_params


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: Any) -> None:
    if cfg.run is None or str(cfg.run) == "???":
        raise ValueError("Missing run id. Provide run=<run_id>.")
    cfg = apply_mode_overrides(cfg)
    set_cache_env(cfg.cache_dir)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = setup_models(cfg, device)

    normalize_prompt_cfg(cfg)

    best_params = run_optuna(cfg, evaluator)
    for key, value in best_params.items():
        if key == "pi_fluency_nll":
            cfg.training.prompt_optimization.pi_fluency_nll = value
        else:
            cfg.training.prompt_optimization[key] = value

    normalize_prompt_cfg(cfg)

    run = build_wandb(cfg)
    if run is None:
        print("WandB disabled.")
    else:
        print(f"WandB URL: {run.url}")

    seed_metrics: List[Dict[str, Any]] = []
    aggregated_gate_fails: Counter = Counter()
    aggregated_confusions: Dict[str, Counter] = defaultdict(Counter)
    global_step = 0

    for seed in cfg.seeds:
        metrics, gate_fail_counts, confusions, global_step = run_seed(
            seed, cfg, evaluator, run, global_step
        )
        seed_metrics.append(metrics)
        aggregated_gate_fails.update(gate_fail_counts)
        for domain, counts in confusions.items():
            aggregated_confusions[domain].update(counts)

    robust_values = [m["robust_constrained_accuracy"] for m in seed_metrics]
    worst_slice_values = [m["worst_slice_constrained_accuracy"] for m in seed_metrics]
    ood_values = [m["ood_regression"] for m in seed_metrics]
    accepted_values = [m["accepted_steps"] for m in seed_metrics]
    candidates_values = [m["candidates_evaluated"] for m in seed_metrics]

    best_idx = int(np.argmax(robust_values)) if robust_values else 0
    best_prompt = seed_metrics[best_idx]["final_prompt"] if seed_metrics else cfg.prompt.base

    if run is not None:
        wandb.summary["eval/robust_constrained_accuracy_mean"] = float(np.mean(robust_values))
        wandb.summary["eval/robust_constrained_accuracy_std"] = float(np.std(robust_values))
        wandb.summary["eval/robust_constrained_accuracy_best"] = float(np.max(robust_values))
        wandb.summary["eval/worst_slice_constrained_accuracy_mean"] = float(
            np.mean(worst_slice_values)
        )
        wandb.summary["eval/worst_slice_constrained_accuracy_std"] = float(
            np.std(worst_slice_values)
        )
        wandb.summary["eval/ood_regression_rate"] = float(np.mean(ood_values))
        wandb.summary["train/accepted_steps_mean"] = float(np.mean(accepted_values))
        wandb.summary["train/candidates_evaluated_mean"] = float(np.mean(candidates_values))
        wandb.summary["prompt/base"] = cfg.prompt.base
        wandb.summary["prompt/best"] = best_prompt
        wandb.summary["best_epoch"] = 0

        for domain in ["sst2", "yelp", "imdb"]:
            domain_values = [m["per_domain"].get(domain, math.nan) for m in seed_metrics]
            wandb.summary[f"eval/{domain}_accuracy_mean"] = float(np.nanmean(domain_values))
            wandb.summary[f"eval/{domain}_accuracy_std"] = float(np.nanstd(domain_values))
            wandb.summary[f"eval/{domain}_accuracy_best"] = float(np.nanmax(domain_values))

        for gate, count in aggregated_gate_fails.items():
            key = f"gate_fail/{gate}"
            wandb.summary[key] = float(count)

        for domain, counts in aggregated_confusions.items():
            for key, value in counts.items():
                wandb.summary[f"confusion/{domain}_{key}"] = float(value)

        wandb.finish()


if __name__ == "__main__":
    main()
