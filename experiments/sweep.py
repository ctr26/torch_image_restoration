"""
Optimizer & Scheduler Sweep
============================

Extended sweep:
  - Optimizer sweep:  10 optimizers × warmup=50 (best from RL seeding)
  - Scheduler sweep:  best optimizer × 6 schedulers

Usage:
    cd ~/projects/ctr26/torch_image_restoration
    python -m experiments.sweep               # both sweeps
    python -m experiments.sweep --opt-only    # optimizer sweep only
    python -m experiments.sweep --sched-only  # scheduler sweep only
"""

import sys
import logging
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import (
    load_test_image,
    generate_psf,
    generate_observation,
    train_deconvolution,
    ExperimentResult,
)
from experiments.optimizers import OPTIMIZER_CONFIGS
from experiments.schedulers import get_scheduler, SCHEDULER_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed experiment settings (matching existing RL-seeding setup)
# ---------------------------------------------------------------------------
IMAGE_CFG   = {"name": "astronaut", "downsample_factor": 4, "colorspace": "gray"}
PSF_CFG     = {"type": "gaussian", "size": [64, 64], "sigma": 2.0, "gain": 1000}
NOISE_CFG   = {"type": "poisson", "snr_db": 30}
DEVICE      = "cpu"
MAX_ITERS   = 500
BEST_WARMUP = 50   # winner from RL-seeding sweep

# Optimizers to sweep (names map to OPTIMIZER_CONFIGS keys)
OPTIMIZER_NAMES = [
    "sgd",           # SGD no momentum
    "sgd_momentum",  # SGD momentum=0.9  ✓ (existing)
    "adam",          # Adam              ✓ (existing)
    "adamw",         # AdamW             ✓ (existing)
    "radam",         # RAdam
    "nadam",         # NAdam
    "adagrad",       # Adagrad
    "rmsprop",       # RMSprop
    "adadelta",      # Adadelta
    "lbfgs",         # L-BFGS
]

# Schedulers to sweep
SCHEDULER_NAMES = [
    "constant",              # None (constant LR)  ✓ (existing)
    "cosine",                # CosineAnnealingLR
    "step",                  # StepLR every 50 iters
    "reduce_on_plateau",     # ReduceLROnPlateau
    "one_cycle",             # OneCycleLR
    "cosine_warm_restarts",  # CosineAnnealingWarmRestarts
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load ground truth, PSF, and observed image once."""
    gt   = load_test_image(IMAGE_CFG)
    psf  = generate_psf(PSF_CFG)
    obs  = generate_observation(gt, psf, NOISE_CFG)
    return gt, psf, obs


def result_to_row(result: ExperimentResult, extra: dict) -> dict:
    """Flatten an ExperimentResult to a flat dict for CSV."""
    return {
        "experiment_name":     result.experiment_name,
        "optimizer":           extra.get("optimizer", ""),
        "scheduler":           extra.get("scheduler", ""),
        "rl_warmup":           extra.get("rl_warmup", BEST_WARMUP),
        "lr":                  extra.get("lr", ""),
        "final_psnr":          result.final_psnr,
        "final_ssim":          result.final_ssim,
        "final_mse":           result.final_mse,
        "final_mae":           result.final_mae,
        "iterations":          result.iterations,
        "converged":           result.converged,
        "convergence_iter":    result.convergence_iteration,
        "total_time_s":        result.total_time,
        "time_per_iter_ms":    result.time_per_iteration * 1000,
        "timestamp":           result.timestamp,
    }


# ---------------------------------------------------------------------------
# Optimizer Sweep
# ---------------------------------------------------------------------------

def run_optimizer_sweep(gt, psf, obs, results_dir: Path) -> pd.DataFrame:
    """10 optimizers × warmup=50, constant LR scheduler."""
    log.info("=" * 70)
    log.info(f"OPTIMIZER SWEEP — {len(OPTIMIZER_NAMES)} optimizers × warmup={BEST_WARMUP}")
    log.info("=" * 70)

    rows = []

    for opt_name in OPTIMIZER_NAMES:
        cfg = OPTIMIZER_CONFIGS[opt_name]
        lr  = cfg["default_lr"]
        log.info(f"\n--- {opt_name} (lr={lr}) ---")

        try:
            result = train_deconvolution(
                observed=obs,
                psf=psf,
                ground_truth=gt,
                optimizer_name=opt_name,
                scheduler_name="constant",
                lr=lr,
                max_iterations=MAX_ITERS,
                device=DEVICE,
                rl_warmup_iterations=BEST_WARMUP,
                use_rl_derived_lr=False,
                experiment_name=f"opt_sweep_{opt_name}_w{BEST_WARMUP}",
                convergence_threshold=1e-5,
                early_stopping_patience=50,
            )
            row = result_to_row(result, {
                "optimizer": opt_name,
                "scheduler": "constant",
                "rl_warmup": BEST_WARMUP,
                "lr": lr,
            })
            log.info(
                f"  PSNR={result.final_psnr:.3f}  SSIM={result.final_ssim:.4f}"
                f"  iters={result.iterations}  t={result.total_time:.1f}s"
            )
        except Exception as e:
            log.error(f"  FAILED: {e}")
            row = {
                "experiment_name": f"opt_sweep_{opt_name}_w{BEST_WARMUP}",
                "optimizer": opt_name,
                "scheduler": "constant",
                "rl_warmup": BEST_WARMUP,
                "lr": lr,
                "final_psnr": float("nan"),
                "final_ssim": float("nan"),
                "error": str(e),
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    out = results_dir / "optimizer_sweep_results.csv"
    df.to_csv(out, index=False)
    log.info(f"\nOptimizer sweep saved → {out}")

    # Summary
    valid = df.dropna(subset=["final_psnr"]).sort_values("final_psnr", ascending=False)
    log.info("\nOptimizer Ranking (by PSNR):")
    log.info(valid[["optimizer", "final_psnr", "final_ssim", "iterations", "total_time_s"]].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Scheduler Sweep
# ---------------------------------------------------------------------------

def run_scheduler_sweep(gt, psf, obs, best_optimizer: str, results_dir: Path) -> pd.DataFrame:
    """6 schedulers × best optimizer × warmup=50."""
    log.info("\n" + "=" * 70)
    log.info(f"SCHEDULER SWEEP — {len(SCHEDULER_NAMES)} schedulers × optimizer={best_optimizer} × warmup={BEST_WARMUP}")
    log.info("=" * 70)

    cfg = OPTIMIZER_CONFIGS[best_optimizer]
    lr  = cfg["default_lr"]
    rows = []

    for sched_name in SCHEDULER_NAMES:
        log.info(f"\n--- {sched_name} ---")

        # Build extra kwargs needed for certain schedulers
        sched_kwargs = {}
        if sched_name == "step":
            sched_kwargs = {"step_size": 50, "gamma": 0.5}   # decay every 50 iters
        elif sched_name == "cosine":
            sched_kwargs = {"T_max": MAX_ITERS}
        elif sched_name == "cosine_warm_restarts":
            sched_kwargs = {"T_0": 50, "T_mult": 2}
        elif sched_name == "one_cycle":
            sched_kwargs = {"max_lr": lr * 10, "total_steps": MAX_ITERS}
        elif sched_name == "reduce_on_plateau":
            sched_kwargs = {"mode": "min", "factor": 0.5, "patience": 10}

        try:
            # Monkey-patch get_scheduler call for this run by using override_kwargs
            # The runner calls get_scheduler(scheduler_name, optimizer) with no extra kwargs.
            # We need to handle this by temporarily patching the SCHEDULER_CONFIGS.
            import experiments.schedulers as sched_module
            orig_params = sched_module.SCHEDULER_CONFIGS[sched_name]["params"].copy()
            sched_module.SCHEDULER_CONFIGS[sched_name]["params"].update(sched_kwargs)

            result = train_deconvolution(
                observed=obs,
                psf=psf,
                ground_truth=gt,
                optimizer_name=best_optimizer,
                scheduler_name=sched_name,
                lr=lr,
                max_iterations=MAX_ITERS,
                device=DEVICE,
                rl_warmup_iterations=BEST_WARMUP,
                use_rl_derived_lr=False,
                experiment_name=f"sched_sweep_{best_optimizer}_{sched_name}_w{BEST_WARMUP}",
                convergence_threshold=1e-5,
                early_stopping_patience=50,
            )
            # Restore original params
            sched_module.SCHEDULER_CONFIGS[sched_name]["params"] = orig_params

            row = result_to_row(result, {
                "optimizer": best_optimizer,
                "scheduler": sched_name,
                "rl_warmup": BEST_WARMUP,
                "lr": lr,
            })
            log.info(
                f"  PSNR={result.final_psnr:.3f}  SSIM={result.final_ssim:.4f}"
                f"  iters={result.iterations}  t={result.total_time:.1f}s"
            )
        except Exception as e:
            log.error(f"  FAILED: {e}")
            import experiments.schedulers as sched_module
            sched_module.SCHEDULER_CONFIGS[sched_name]["params"] = orig_params
            row = {
                "experiment_name": f"sched_sweep_{best_optimizer}_{sched_name}_w{BEST_WARMUP}",
                "optimizer": best_optimizer,
                "scheduler": sched_name,
                "rl_warmup": BEST_WARMUP,
                "lr": lr,
                "final_psnr": float("nan"),
                "final_ssim": float("nan"),
                "error": str(e),
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    out = results_dir / "scheduler_sweep_results.csv"
    df.to_csv(out, index=False)
    log.info(f"\nScheduler sweep saved → {out}")

    # Summary
    valid = df.dropna(subset=["final_psnr"]).sort_values("final_psnr", ascending=False)
    log.info("\nScheduler Ranking (by PSNR):")
    log.info(valid[["scheduler", "final_psnr", "final_ssim", "iterations", "total_time_s"]].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt-only",   action="store_true", help="Run optimizer sweep only")
    parser.add_argument("--sched-only", action="store_true", help="Run scheduler sweep only")
    parser.add_argument("--best-optimizer", type=str, default=None,
                        help="Force best optimizer for scheduler sweep (skips opt sweep)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    gt, psf, obs = load_data()
    log.info(f"  image shape: {gt.shape}, PSF shape: {psf.shape}")

    best_optimizer = args.best_optimizer or "sgd_momentum"  # prior best from RL seeding

    if not args.sched_only:
        df_opt = run_optimizer_sweep(gt, psf, obs, results_dir)
        # Pick best optimizer by PSNR
        valid = df_opt.dropna(subset=["final_psnr"])
        if not valid.empty:
            best_optimizer = valid.sort_values("final_psnr", ascending=False).iloc[0]["optimizer"]
            log.info(f"\n★ Best optimizer: {best_optimizer} "
                     f"(PSNR={valid.sort_values('final_psnr', ascending=False).iloc[0]['final_psnr']:.3f})")

    if not args.opt_only:
        df_sched = run_scheduler_sweep(gt, psf, obs, best_optimizer, results_dir)
        valid = df_sched.dropna(subset=["final_psnr"])
        if not valid.empty:
            best_row = valid.sort_values("final_psnr", ascending=False).iloc[0]
            log.info(f"\n★ Best scheduler: {best_row['scheduler']} "
                     f"(PSNR={best_row['final_psnr']:.3f})")
            log.info(f"\n🏆 WINNER: optimizer={best_optimizer}, "
                     f"scheduler={best_row['scheduler']}, "
                     f"PSNR={best_row['final_psnr']:.3f}, "
                     f"SSIM={best_row['final_ssim']:.4f}")


if __name__ == "__main__":
    main()
