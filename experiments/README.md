# Optimizer & Scheduler Experiments

This directory contains a comprehensive experimental framework for testing optimization strategies for image deconvolution.

## Overview

The framework tests:
- **10 optimizers**: SGD, SGD+momentum, Adam, AdamW, RMSprop, LBFGS, Adagrad, Adadelta, NAdam, RAdam
- **8+ schedulers**: Constant, StepLR, ExponentialLR, CosineAnnealing, SGDR, ReduceLROnPlateau, CyclicLR, OneCycleLR
- **Theory-driven initialization**: Richardson-Lucy derived learning rates
- **RL-seeding**: Warm-start gradient descent with RL iterations

## Structure

```
experiments/
├── __init__.py              # Package initialization
├── optimizers.py            # Optimizer configurations and factory functions
├── schedulers.py            # Scheduler configurations (including RL-derived LR)
├── config.yaml              # Experiment definitions
├── runner.py                # Experiment orchestration
├── metrics.py               # Evaluation metrics
├── README.md                # This file
└── results/                 # Output directory (created at runtime)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm
```

### 2. Run All Experiments

```bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --output experiments/results
```

### 3. Run Specific Experiment

```bash
python -m experiments.runner --config experiments/config.yaml --experiment optimizer_comparison
```

## Configuration

The `config.yaml` file defines 9 experiment suites:

1. **Optimizer Comparison**: Fixed LR, vary optimizer
2. **Scheduler Comparison**: Fixed optimizer, vary scheduler  
3. **Grid Search**: Optimizer × Scheduler × LR
4. **SNR Sweep**: Noise robustness testing
5. **PSF Complexity**: Different PSF types (Gaussian, aberrated)
6. **Convergence Analysis**: Speed vs quality tradeoff
7. **RL-Seeding**: RL warm-start + gradient descent
8. **RL-Derived LR**: Compare RL-derived vs manual LR
9. **LBFGS Special**: Closure-based optimization

## Key Features

### Richardson-Lucy Derived Learning Rate

The `compute_rl_learning_rate()` function in `schedulers.py` derives an initial learning rate from Richardson-Lucy theory:

```python
from experiments.schedulers import compute_rl_learning_rate

rl_lr = compute_rl_learning_rate(psf, observed_data, n_iterations=10)
print(f"RL-derived LR: {rl_lr:.6e}")
```

**Theory**: RL uses multiplicative updates with implicit step size. We estimate an equivalent gradient descent LR by analyzing update magnitudes.

### RL-Seeding Strategy

Combine the best of both worlds:
1. Run N iterations of Richardson-Lucy (gets good initial estimate)
2. Switch to gradient descent with modern optimizers (refines solution)

```python
# In config.yaml:
rl_seeding:
  rl_warmup_iterations: [0, 5, 10, 20, 50, 100]
  post_rl_optimizers:
    - optimizer: "adam"
      scheduler: "cosine"
```

## Optimizers

All optimizers are configured with sensible defaults based on literature:

| Optimizer | Default LR | Notes |
|-----------|------------|-------|
| SGD | 1e-3 | Baseline |
| SGD+Momentum | 1e-3 | momentum=0.9 |
| Adam | 1e-3 | Most popular |
| AdamW | 1e-3 | Better regularization |
| RMSprop | 1e-3 | Good for non-stationary |
| LBFGS | 1.0 | Requires closure, fewer iterations |
| Adagrad | 1e-2 | Monotonic LR decay |
| Adadelta | 1.0 | No manual LR needed |
| NAdam | 2e-3 | Nesterov + Adam |
| RAdam | 1e-3 | Variance warmup |

See `optimizers.py` for full details.

## Schedulers

| Scheduler | Description |
|-----------|-------------|
| ConstantLR | No decay (baseline) |
| StepLR | Periodic decay |
| ExponentialLR | Smooth exponential decay |
| CosineAnnealingLR | Cosine curve to minimum |
| CosineAnnealingWarmRestarts | SGDR with periodic restarts |
| ReduceLROnPlateau | Adaptive to loss plateau |
| CyclicLR | Oscillate between bounds |
| OneCycleLR | Ramp up then anneal (1cycle policy) |

See `schedulers.py` for full details.

## Metrics

Each experiment computes:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **Convergence speed**: Iterations to reach 95% of final quality
- **Compute time**: Wall-clock time
- **Efficiency**: PSNR gain per second

## Results

Results are saved as CSV files:
```
results/
├── all_results.csv                    # Combined results
├── optimizer_comparison_results.csv
├── scheduler_comparison_results.csv
├── rl_seeding_results.csv
└── ...
```

Each row contains:
- Experiment config (optimizer, scheduler, LR, etc.)
- Final metrics (PSNR, SSIM, MSE, MAE)
- Convergence info (iterations, converged, convergence_iteration)
- Timing (total_time, time_per_iteration)
- Training history (loss, PSNR, LR over iterations)

## Visualization

Generate comparison plots:

```bash
python -m experiments.visualize --input results/all_results.csv --output plots/
```

Plots include:
- Convergence curves (loss/PSNR vs iteration)
- Final metrics bar charts
- SNR robustness curves
- Convergence speed vs quality scatter
- RL-seeding benefit analysis

## Usage Examples

### Test a Single Configuration

```python
from experiments.runner import train_deconvolution
from experiments.optimizers import get_optimizer
from experiments.schedulers import get_scheduler

# Setup data (see runner.py for helpers)
observed = ...  # Blurred+noisy image
psf = ...       # Point spread function
ground_truth = ...  # Clean image

# Train
result = train_deconvolution(
    observed=observed,
    psf=psf,
    ground_truth=ground_truth,
    optimizer_name='adam',
    scheduler_name='cosine',
    lr=1e-3,
    max_iterations=1000,
    device='cuda'
)

print(f"Final PSNR: {result.final_psnr:.2f} dB")
print(f"Converged in {result.iterations} iterations")
```

### Compare Optimizers Programmatically

```python
from experiments.optimizers import OPTIMIZER_CONFIGS

for opt_name in OPTIMIZER_CONFIGS.keys():
    result = train_deconvolution(
        observed=observed,
        psf=psf,
        ground_truth=ground_truth,
        optimizer_name=opt_name,
        scheduler_name='constant',
        lr=1e-3,
        max_iterations=1000
    )
    print(f"{opt_name:15s} | PSNR: {result.final_psnr:.2f} | Time: {result.total_time:.2f}s")
```

### Use RL-Seeding

```python
result = train_deconvolution(
    observed=observed,
    psf=psf,
    ground_truth=ground_truth,
    optimizer_name='adam',
    scheduler_name='cosine',
    lr=1e-3,
    max_iterations=500,
    rl_warmup_iterations=20,  # Run 20 RL iterations first
    use_rl_derived_lr=True     # Use RL-derived LR
)
```

## Theory: Richardson-Lucy Seeding

**Why RL-seeding works:**

1. **RL strength**: Fast initial convergence, good for getting in the right ballpark
2. **RL weakness**: Slow final convergence, no adaptive step size
3. **Gradient descent strength**: Modern optimizers (Adam, etc.) excel at refinement
4. **Gradient descent weakness**: Can converge slowly from random init

**Solution**: Use RL to get a warm start, then refine with gradient descent.

**Empirical question**: How many RL iterations are optimal? (Answered by experiments!)

## Implementation Notes

### Convolution

The current implementation uses PyTorch's `conv2d` for simplicity. For production:
- Use FFT-based convolution for large images
- Implement circulant matrix multiplication (as in `utils.py`)
- Consider GPU-accelerated convolution libraries

### Memory

For large images or batch processing:
- Enable gradient checkpointing
- Use mixed precision (FP16)
- Process tiles instead of full images

### Parallelization

To run experiments in parallel:
```bash
# Run different experiments on different GPUs
CUDA_VISIBLE_DEVICES=0 python -m experiments.runner --experiment optimizer_comparison &
CUDA_VISIBLE_DEVICES=1 python -m experiments.runner --experiment scheduler_comparison &
```

## References

### Optimizers
- Kingma & Ba (2015): "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization" (AdamW)
- Liu & Nocedal (1989): "On the Limited Memory BFGS Method" (LBFGS)
- Liu et al. (2020): "On the Variance of the Adaptive Learning Rate" (RAdam)

### Schedulers
- Smith (2017): "Cyclical Learning Rates for Training Neural Networks"
- Loshchilov & Hutter (2017): "SGDR: Stochastic Gradient Descent with Warm Restarts"
- Smith & Topin (2019): "Super-Convergence" (1cycle policy)

### Richardson-Lucy
- Richardson (1972): "Bayesian-Based Iterative Method of Image Restoration"
- Lucy (1974): "An Iterative Technique for the Rectification of Observed Distributions"

## Contributing

To add a new optimizer:
1. Add factory function in `optimizers.py`
2. Add entry to `OPTIMIZER_CONFIGS`
3. Update this README

To add a new experiment:
1. Define in `config.yaml`
2. Implement handler in `runner.py`
3. Update this README

## License

Same as parent repository.
