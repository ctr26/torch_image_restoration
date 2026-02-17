# Optimizer & Scheduler Experiments: Implementation Summary

**Date**: 2026-02-16  
**Status**: Framework complete, ready for execution  
**Repository**: https://github.com/ctr26/torch_image_restoration

## What Was Built

A comprehensive experimental framework for testing optimization strategies on image deconvolution tasks.

### Core Components

1. **`experiments/optimizers.py`** (9.2 KB)
   - 10 optimizer configurations (SGD, Adam, AdamW, RMSprop, LBFGS, etc.)
   - Factory functions with sensible defaults
   - Optimizer registry for easy access
   - Benchmark utility

2. **`experiments/schedulers.py`** (15.9 KB)
   - 8+ learning rate schedulers
   - **Novel**: Richardson-Lucy derived learning rate computation
   - Custom `RLSeededScheduler` class
   - Visualization utilities

3. **`experiments/config.yaml`** (12.1 KB)
   - 9 experiment definitions:
     1. Optimizer comparison
     2. Scheduler comparison
     3. Grid search (optimizer × scheduler × LR)
     4. SNR sweep (noise robustness)
     5. PSF complexity sweep
     6. Convergence speed analysis
     7. **RL-seeding experiments**
     8. RL-derived LR validation
     9. LBFGS special case
   - Comprehensive parameter specifications
   - Analysis and reporting configuration

4. **`experiments/runner.py`** (17.4 KB)
   - Experiment orchestration
   - Data generation (images, PSFs, noise)
   - Richardson-Lucy implementation
   - PyTorch training loop with:
     - Early stopping
     - Convergence detection
     - Metrics tracking (PSNR, SSIM, MSE, MAE)
     - Learning rate scheduling
   - Results saving (CSV format)

5. **`experiments/metrics.py`** (2.1 KB)
   - Quality metrics (PSNR, SSIM, MSE, MAE)
   - Convergence speed computation
   - Efficiency metrics

6. **`experiments/README.md`** (9.0 KB)
   - Complete documentation
   - Quick start guide
   - Theory explanations (RL-seeding, RL-derived LR)
   - Usage examples
   - References

7. **`experiments/create_issues.sh`** (18.0 KB)
   - GitHub issue creation script
   - 12 issues covering:
     - Infrastructure setup
     - All 9 experiments
     - Visualization & analysis
     - Documentation & paper

### Key Innovations

#### 1. Richardson-Lucy Seeding
Novel hybrid approach combining:
- RL iterations for warm-start (fast initial convergence)
- Gradient descent for refinement (modern optimizers)

**Hypothesis**: RL gets you in the right ballpark fast, gradient descent refines efficiently.

#### 2. RL-Derived Learning Rate
Theory-driven initialization:
```python
compute_rl_learning_rate(psf, data, n_iterations=10)
```

Derives gradient descent LR from RL update magnitudes. Potentially eliminates manual LR tuning.

#### 3. Systematic Comparison Framework
- **Factorial design**: Test one variable at a time
- **Grid search**: Explore interactions
- **Robustness testing**: SNR sweep, PSF complexity
- **Speed/quality tradeoff**: Convergence analysis

## Experiments Overview

| # | Experiment | Variables | Combinations | Est. Time |
|---|------------|-----------|--------------|-----------|
| 1 | Optimizer comparison | 10 optimizers × 5 LRs | 50 | 1-2 hrs |
| 2 | Scheduler comparison | 8 schedulers | 8 | 30 min |
| 3 | Grid search | 4 opt × 3 sched × 3 LR | 36 | 1 hr |
| 4 | SNR sweep | 4 noise × 4 optimizers | 16 | 30 min |
| 5 | PSF sweep | 4 PSF types | 4 | 15 min |
| 6 | Convergence analysis | 5 optimizers × 2000 iter | 5 | 1 hr |
| 7 | **RL-seeding** | 7 warmup × 3 optimizers | 21 | 1 hr |
| 8 | RL-LR comparison | 4 LR strategies | 4 | 15 min |
| 9 | LBFGS special | 3 configurations | 3 | 15 min |

**Total**: ~147 experimental runs, ~6-8 hours compute time (GPU)

## GitHub Issues Created

12 issues created in repository:

1. **Setup experiment infrastructure** - Dependencies, testing
2. **Run optimizer comparison** - 10 optimizers, 5 LRs
3. **Run scheduler comparison** - 8 schedulers
4. **Run RL-seeding experiments** - Core novelty
5. **Run grid search** - 36 combinations
6. **Run SNR sweep** - Noise robustness
7. **Run PSF sweep** - PSF complexity
8. **Convergence speed analysis** - Speed/quality tradeoff
9. **Test RL-derived LR** - Theory validation
10. **Test LBFGS** - Closure-based optimization
11. **Generate visualizations** - Plots, summary tables
12. **Write paper/documentation** - Publication-ready report

## Expected Deliverables

### Code & Data
- ✅ Experiment framework (complete)
- ⏳ Results CSV files (after execution)
- ⏳ Checkpoints and intermediate images

### Analysis
- ⏳ Convergence curves
- ⏳ Final metrics bar charts
- ⏳ Heatmaps (grid search)
- ⏳ SNR robustness plots
- ⏳ RL-seeding benefit analysis
- ⏳ Statistical tests (ANOVA, t-tests)

### Documentation
- ✅ README with usage guide
- ⏳ Jupyter tutorial notebook
- ⏳ Technical report or paper
- ⏳ Summary tables (top 10 configs, etc.)

## How to Execute

### 1. Setup (Issue #1)
```bash
cd ~/projects/ctr26/torch_image_restoration
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm

# Test
python -m experiments.optimizers
python -m experiments.schedulers
```

### 2. Run All Experiments
```bash
python -m experiments.runner --config experiments/config.yaml --output results/
```

### 3. Run Specific Experiment
```bash
python -m experiments.runner --experiment rl_seeding --output results/rl_seeding
```

### 4. Generate Visualizations (after experiments)
```bash
python -m experiments.visualize --input results/ --output plots/
```

## Scientific Questions Addressed

1. **Which optimizer is best for image deconvolution?**
   - Systematic comparison across 10 optimizers
   - Context-dependent: noise, PSF, speed requirements

2. **Do learning rate schedulers help?**
   - Comparison of 8 scheduling strategies
   - Identify best for this problem

3. **Does RL-seeding improve gradient descent?**
   - **Novel contribution**
   - Optimal warmup iterations
   - Comparison to pure methods

4. **Can we derive LR from theory?**
   - Validate RL-derived LR formula
   - Compare to manual tuning

5. **How robust are methods to noise?**
   - SNR sweep (10-40 dB)
   - Identify robust optimizers

6. **Speed vs quality tradeoff?**
   - Convergence analysis
   - Recommendations for different use cases

## Next Steps

### Immediate (Week 1)
1. ✅ Framework implementation - **DONE**
2. ✅ GitHub issues created - **DONE**
3. ⏳ Run infrastructure setup (Issue #1)
4. ⏳ Test on small example

### Short-term (Week 2-3)
1. ⏳ Run all experiments (Issues #2-#10)
2. ⏳ Generate visualizations (Issue #11)
3. ⏳ Statistical analysis

### Medium-term (Month 1-2)
1. ⏳ Write technical report (Issue #12)
2. ⏳ Create tutorial notebook
3. ⏳ Publish results (paper or blog post)

### Long-term (Month 3+)
1. ⏳ Extend to 3D deconvolution
2. ⏳ Test on real microscopy data
3. ⏳ Integrate with existing deconvolution tools

## Theoretical Contributions

### RL-Seeding (Novel)
**Observation**: RL and gradient descent have complementary strengths.

**Solution**: Hybrid approach
1. Run N RL iterations (warm start)
2. Switch to gradient descent (refinement)

**Empirical validation**: Experiment #7 tests N ∈ {0, 5, 10, 20, 50, 100, 500}

### RL-Derived Learning Rate (Novel)
**Theory**: RL update can be approximated as gradient descent with implicit step size.

**Method**:
1. Run M RL iterations
2. Analyze update magnitudes vs gradient magnitudes
3. Estimate equivalent gradient descent LR: `α = Δx / ∇L`

**Validation**: Experiment #8 compares to manual LR tuning.

## References

### Deconvolution
- Richardson (1972): Bayesian-based iterative method
- Lucy (1974): Maximum likelihood deconvolution

### Optimizers
- Kingma & Ba (2015): Adam
- Loshchilov & Hutter (2019): AdamW
- Liu et al. (2020): RAdam

### Schedulers
- Smith (2017): Cyclical learning rates
- Loshchilov & Hutter (2017): SGDR
- Smith & Topin (2019): 1cycle policy

## Repository Structure

```
torch_image_restoration/
├── pytorch_Hx.py              # Original PyTorch deconvolution
├── pyro.ai.py                 # Pyro probabilistic model
├── utils.py                   # Circulant matrix utilities
├── experiments/               # NEW: Experiment framework
│   ├── __init__.py
│   ├── optimizers.py          # 10 optimizer configs
│   ├── schedulers.py          # 8+ scheduler configs + RL-derived LR
│   ├── config.yaml            # 9 experiment definitions
│   ├── runner.py              # Orchestration and training loop
│   ├── metrics.py             # Quality metrics
│   ├── README.md              # Documentation
│   ├── create_issues.sh       # GitHub issue generator
│   └── results/               # Output directory (created at runtime)
└── EXPERIMENTS_SUMMARY.md     # This file
```

## Contact & Support

- **Repository**: https://github.com/ctr26/torch_image_restoration
- **Issues**: https://github.com/ctr26/torch_image_restoration/issues
- **Questions**: See experiment README.md

## License

Same as parent repository.

---

**Status**: ✅ Framework complete, ready for execution  
**Last updated**: 2026-02-16
