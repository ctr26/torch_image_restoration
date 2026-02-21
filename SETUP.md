# Setup Guide for Experiment Infrastructure

This guide explains how to set up and verify the experiment infrastructure for torch_image_restoration.

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm

# 2. Verify setup
bash scripts/setup_experiments.sh

# 3. Run quick test
python scripts/test_experiment.py

# 4. Run experiments
python -m experiments.optimizers
python -m experiments.schedulers
```

## Detailed Setup

### Option 1: Using pip (Universal)

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (Faster)

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option 3: Using conda

```bash
# Create environment from file
conda env create -f environment.yml

# Activate it
conda activate 2021_image_restoration

# Install missing packages
pip install pyyaml pandas tqdm
```

## Dependencies

### Core Requirements

- **torch** (≥2.0.0): Deep learning framework
- **numpy** (≥1.21.0): Numerical arrays
- **scipy** (≥1.7.0): Scientific computing (convolution, filters)
- **scikit-image** (≥0.19.0): Image processing metrics (PSNR, SSIM)
- **pyyaml** (≥6.0): Configuration file parsing
- **pandas** (≥1.3.0): Results data frames
- **matplotlib** (≥3.4.0): Plotting (future visualizations)
- **tqdm** (≥4.62.0): Progress bars

### Optional

- **pyro-ppl** (≥1.8.0): Probabilistic programming (for advanced experiments)

### Development

- **pytest** (≥7.0.0): Testing framework
- **ruff** (≥0.1.0): Linting
- **mypy** (≥1.0.0): Type checking

## Verification

### 1. Check Dependencies

```bash
python -c "import torch, numpy, scipy, skimage, yaml, pandas, matplotlib, tqdm; print('✅ All dependencies OK')"
```

### 2. Verify Experiment Modules

```bash
python -m experiments.optimizers
python -m experiments.schedulers
```

Expected output:
```
Available optimizers: sgd, sgd_momentum, adam, adamw, rmsprop, lbfgs, adagrad, adadelta, nadam, radam
Available schedulers: constant, step, exponential, cosine, sgdr, plateau, cyclic, onecycle
```

### 3. Run Test Suite

```bash
python scripts/test_experiment.py
```

This runs a minimal experiment to verify:
- All optimizers are accessible
- All schedulers are accessible
- RL-derived learning rate computation works
- Full training pipeline executes
- Results directory structure exists

### 4. Run Setup Script

```bash
bash scripts/setup_experiments.sh --install
```

This script:
- Creates results directory structure
- Verifies Python version
- (Optionally) installs dependencies
- Tests all experiment modules
- Provides next steps

## Directory Structure

After setup, you should have:

```
torch_image_restoration/
├── experiments/
│   ├── optimizers.py          # 10 optimizer configurations
│   ├── schedulers.py          # 8+ scheduler configurations
│   ├── runner.py              # Training orchestration
│   ├── metrics.py             # Evaluation metrics
│   ├── config.yaml            # Experiment definitions
│   └── README.md              # Full documentation
├── results/                   # Created by setup
│   ├── optimizer_comparison/
│   ├── scheduler_comparison/
│   ├── grid_search/
│   ├── snr_sweep/
│   ├── psf_sweep/
│   ├── convergence_analysis/
│   ├── rl_seeding/
│   ├── rl_lr_comparison/
│   ├── lbfgs_experiment/
│   ├── plots/
│   └── checkpoints/
├── scripts/
│   ├── setup_experiments.sh   # Setup verification script
│   └── test_experiment.py     # Quick test
├── requirements.txt           # Pip dependencies
├── pyproject.toml            # uv/pip project config
└── SETUP.md                  # This file
```

## Running Experiments

### Test Individual Modules

```bash
# List available optimizers
python -m experiments.optimizers

# List available schedulers  
python -m experiments.schedulers

# Test runner (shows usage)
python -m experiments.runner --help
```

### Run Quick Test

```bash
# Minimal experiment (32×32 image, 10 iterations)
python scripts/test_experiment.py
```

### Run Single Experiment

```bash
# RL-seeding experiment
python -m experiments.runner --experiment rl_seeding --output results/rl_seeding

# Optimizer comparison
python -m experiments.runner --experiment optimizer_comparison --output results/optimizer_comparison
```

### Run All Experiments

```bash
# Full experimental suite (6-8 hours on GPU)
python -m experiments.runner --config experiments/config.yaml --output results/
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Edit `experiments/config.yaml` and change `device: "cuda"` to `device: "cpu"`

### Slow Experiments

**Problem**: Experiments taking too long

**Solution**: Reduce iterations in `config.yaml`:
```yaml
max_iterations: 100  # Instead of 1000
```

### No Space Left on Device

**Problem**: `/tmp` fills up during installation

**Solution**: Install in your home directory:
```bash
cd ~/projects/torch_image_restoration
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. ✅ Dependencies installed
2. ✅ Setup verified (`scripts/setup_experiments.sh`)
3. ✅ Quick test passed (`scripts/test_experiment.py`)
4. → Run experiments (`python -m experiments.runner`)
5. → Analyze results (see `experiments/README.md`)

## References

- Full documentation: `experiments/README.md`
- Quick start: `experiments/QUICKSTART.md`
- Configuration: `experiments/config.yaml`
- Code: `experiments/optimizers.py`, `experiments/schedulers.py`, `experiments/runner.py`

## Support

For issues:
1. Check this guide first
2. Read `experiments/README.md` for detailed documentation
3. Run `python scripts/test_experiment.py` to diagnose
4. File an issue on GitHub with the error message
