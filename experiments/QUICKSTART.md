# Experiments Quick Start

## 🚀 Run Everything (One Command)

```bash
cd ~/projects/ctr26/torch_image_restoration
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm
python -m experiments.runner --config experiments/config.yaml --output results/
```

**Time**: ~6-8 hours on GPU, ~24-48 hours on CPU  
**Output**: `results/all_results.csv` + individual experiment CSVs

---

## 📊 Run Single Experiment

```bash
# Just RL-seeding (the novel contribution)
python -m experiments.runner --experiment rl_seeding --output results/rl_seeding

# Just optimizer comparison
python -m experiments.runner --experiment optimizer_comparison --output results/optimizer_comparison
```

---

## 🧪 Test Framework (Quick)

```bash
# List available optimizers
python -m experiments.optimizers

# List available schedulers
python -m experiments.schedulers

# Test RL-derived LR computation
python -c "from experiments.schedulers import compute_rl_learning_rate; import numpy as np; from scipy.ndimage import gaussian_filter; psf = np.zeros((64,64)); psf[32,32]=1; psf = gaussian_filter(psf, 2.0); data = np.random.rand(128,128); lr = compute_rl_learning_rate(psf, data); print(f'RL-LR: {lr:.6e}')"
```

---

## 📁 File Structure

```
experiments/
├── config.yaml          # 9 experiment definitions
├── optimizers.py        # 10 optimizer configs
├── schedulers.py        # 8+ schedulers + RL-derived LR
├── runner.py            # Training loop & orchestration
├── metrics.py           # PSNR, SSIM, convergence speed
├── README.md            # Full documentation
└── QUICKSTART.md        # This file
```

---

## 🔬 The 9 Experiments

| # | Name | What It Tests | Time |
|---|------|---------------|------|
| 1 | `optimizer_comparison` | 10 optimizers × 5 LRs | 1-2h |
| 2 | `scheduler_comparison` | 8 schedulers | 30m |
| 3 | `grid_search` | optimizer × scheduler × LR | 1h |
| 4 | `snr_sweep` | noise robustness | 30m |
| 5 | `psf_sweep` | PSF complexity | 15m |
| 6 | `convergence_analysis` | speed vs quality | 1h |
| 7 | **`rl_seeding`** | **RL warmup + gradient descent** | **1h** |
| 8 | `rl_lr_comparison` | RL-derived LR validation | 15m |
| 9 | `lbfgs_experiment` | LBFGS closure-based | 15m |

**Total**: ~6-8 hours

---

## 🎯 Key Innovations

### 1. RL-Seeding
Warm-start gradient descent with Richardson-Lucy iterations:
```python
result = train_deconvolution(
    observed=blurred_image,
    psf=point_spread_function,
    ground_truth=clean_image,
    optimizer_name='adam',
    scheduler_name='cosine',
    lr=1e-3,
    rl_warmup_iterations=20  # ← Run 20 RL iterations first
)
```

### 2. RL-Derived Learning Rate
Derive LR from RL theory (no manual tuning):
```python
from experiments.schedulers import compute_rl_learning_rate

rl_lr = compute_rl_learning_rate(psf, observed_data)
# Use this LR instead of manual 1e-3, 1e-4, etc.
```

---

## 📈 Expected Results

After running all experiments, you'll have:
- CSV files with metrics (PSNR, SSIM, convergence speed)
- Answer to: **Best optimizer for deconvolution?**
- Answer to: **Does RL-seeding help?**
- Answer to: **Can we avoid manual LR tuning?**

---

## 🐛 Troubleshooting

**Import errors?**
```bash
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm
```

**CUDA out of memory?**
```python
# In config.yaml, change:
device: "cpu"  # instead of "cuda"
```

**Experiments too slow?**
```python
# In config.yaml, reduce:
max_iterations: 100  # instead of 1000
```

---

## 📚 Documentation

- **Full docs**: `experiments/README.md`
- **Theory**: See RL-seeding and RL-LR sections in README
- **Config reference**: `experiments/config.yaml` (heavily commented)
- **API docs**: Docstrings in all `.py` files

---

## 🏆 Publication Checklist

- [ ] Run all experiments
- [ ] Generate visualizations (TODO: `experiments/visualize.py`)
- [ ] Statistical analysis (ANOVA, t-tests)
- [ ] Write paper (TODO: `experiments/report.py`)
- [ ] Create tutorial notebook

---

## 🔗 Links

- Repo: https://github.com/ctr26/torch_image_restoration
- Issues: https://github.com/ctr26/torch_image_restoration/issues (12 created)
- Summary: `EXPERIMENTS_SUMMARY.md`
