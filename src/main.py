import subprocess
import sys
from pathlib import Path
from typing import Any, List

import hydra
from omegaconf import OmegaConf


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


def format_override(key: str, value: Any) -> str:
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, list):
        rendered = ",".join(str(v) for v in value)
        return f"{key}=[{rendered}]"
    return f"{key}={value}"


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: Any) -> None:
    if cfg.run is None or str(cfg.run) == "???":
        raise ValueError("Missing run id. Provide run=<run_id>.")
    
    # Disable struct mode before merging to allow new keys
    OmegaConf.set_struct(cfg, False)
    
    # Load run-specific config and merge it
    from pathlib import Path
    run_config_path = Path(__file__).parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if run_config_path.exists():
        run_cfg = OmegaConf.load(run_config_path)
        cfg = OmegaConf.merge(cfg, run_cfg)
    
    cfg = apply_mode_overrides(cfg)

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    overrides: List[str] = [
        f"run={cfg.run}",
        f"mode={cfg.mode}",
        f"results_dir={cfg.results_dir}",
        f"wandb.mode={cfg.wandb.mode}",
        f"optuna.n_trials={cfg.optuna.n_trials}",
        format_override("training.epochs", cfg.training.epochs),
        format_override("training.batch_size", cfg.training.batch_size),
        format_override("training.max_batches", cfg.training.max_batches),
        format_override("training.max_eval_batches", cfg.training.max_eval_batches),
        format_override("training.prompt_optimization.iterations", cfg.training.prompt_optimization.iterations),
        format_override(
            "training.prompt_optimization.candidates_per_iter",
            cfg.training.prompt_optimization.candidates_per_iter,
        ),
        format_override("dataset.splits.opt_size", cfg.dataset.splits.opt_size),
        format_override("dataset.splits.audit_total", cfg.dataset.splits.audit_total),
        format_override("dataset.splits.audit_disc", cfg.dataset.splits.audit_disc),
        format_override("dataset.splits.audit_conf", cfg.dataset.splits.audit_conf),
        format_override("dataset.splits.eval_size", cfg.dataset.splits.eval_size),
        format_override("seeds", cfg.seeds),
    ]

    if hasattr(cfg.training.prompt_optimization, "discovery_size"):
        overrides.append(
            format_override(
                "training.prompt_optimization.discovery_size",
                cfg.training.prompt_optimization.discovery_size,
            )
        )
    if hasattr(cfg.training.prompt_optimization, "confirmation_size"):
        overrides.append(
            format_override(
                "training.prompt_optimization.confirmation_size",
                cfg.training.prompt_optimization.confirmation_size,
            )
        )

    cmd = [sys.executable, "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
