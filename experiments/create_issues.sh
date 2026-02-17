#!/bin/bash
# Create GitHub issues for experiment execution

REPO="ctr26/torch_image_restoration"

echo "Creating GitHub issues for optimizer/scheduler experiments..."

# Issue 1: Infrastructure Setup
gh issue create \
  --repo "$REPO" \
  --title "Setup experiment infrastructure" \
  --label "experiment,infrastructure" \
  --body "## Goal
Setup the infrastructure for running optimizer/scheduler experiments.

## Tasks
- [ ] Verify all dependencies are installed (torch, numpy, scipy, scikit-image, pyyaml, pandas, matplotlib, tqdm)
- [ ] Test experiment runner on small example
- [ ] Setup results directory structure
- [ ] Configure logging
- [ ] Test GPU availability and device selection

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm

# Test basic imports
python -c 'from experiments import optimizers, schedulers; print(\"Success!\")'

# Test optimizer/scheduler listings
python -m experiments.optimizers
python -m experiments.schedulers
\`\`\`

## Success Criteria
- All imports work without errors
- Optimizer and scheduler listings display correctly
- GPU (if available) is detected

## Related
- config.yaml
- experiments/README.md"

# Issue 2: Optimizer Comparison
gh issue create \
  --repo "$REPO" \
  --title "Run optimizer comparison experiment" \
  --label "experiment,optimizer" \
  --body "## Goal
Compare all 10 optimizers with fixed learning rate and constant scheduler.

## Optimizers to Test
- SGD (baseline)
- SGD + momentum
- Adam
- AdamW
- RMSprop
- LBFGS (special handling)
- Adagrad
- Adadelta
- NAdam
- RAdam

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Scheduler: constant
- Learning rates: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- Max iterations: 1000

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment optimizer_comparison --output results/optimizer_comparison
\`\`\`

## Expected Output
- \`results/optimizer_comparison/optimizer_comparison_results.csv\`
- Convergence curves for each optimizer
- Summary table with final PSNR, SSIM, convergence speed

## Success Criteria
- All optimizers complete successfully
- Results saved to CSV
- PSNR > 25 dB for at least one optimizer
- Clear winner identified

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/optimizers.py
- experiments/config.yaml"

# Issue 3: Scheduler Comparison
gh issue create \
  --repo "$REPO" \
  --title "Run scheduler comparison experiment" \
  --label "experiment,scheduler" \
  --body "## Goal
Compare all learning rate schedulers with fixed optimizer (Adam).

## Schedulers to Test
- ConstantLR (baseline)
- StepLR
- ExponentialLR
- CosineAnnealingLR
- CosineAnnealingWarmRestarts (SGDR)
- ReduceLROnPlateau (adaptive)
- CyclicLR
- OneCycleLR (1cycle policy)

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Optimizer: Adam
- Initial LR: 1e-3
- Max iterations: 1000

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment scheduler_comparison --output results/scheduler_comparison
\`\`\`

## Expected Output
- \`results/scheduler_comparison/scheduler_comparison_results.csv\`
- Learning rate curves over time
- Convergence comparison

## Success Criteria
- All schedulers complete successfully
- Clear differences in convergence behavior
- At least one scheduler outperforms constant baseline

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/schedulers.py
- experiments/config.yaml"

# Issue 4: RL-Seeding Experiments
gh issue create \
  --repo "$REPO" \
  --title "Run Richardson-Lucy seeding experiments" \
  --label "experiment,rl-seeding" \
  --body "## Goal
Test the hypothesis that RL warm-start improves gradient descent convergence.

## RL Warm-up Iterations to Test
- 0 (pure gradient descent, baseline)
- 5
- 10
- 20
- 50
- 100
- 500 (pure RL, no gradient descent)

## Post-RL Configurations
1. Adam + CosineAnnealing
2. AdamW + CosineAnnealing
3. SGD+momentum + Constant

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Max iterations: 500 (after RL warmup)

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment rl_seeding --output results/rl_seeding
\`\`\`

## Key Questions
1. What is the optimal number of RL warm-up iterations?
2. Does RL-seeding outperform pure gradient descent?
3. Does RL-seeding outperform pure RL?
4. Which optimizer benefits most from RL-seeding?

## Expected Output
- \`results/rl_seeding/rl_seeding_results.csv\`
- Convergence curves showing RL → gradient descent transition
- Comparison of final PSNR vs RL warmup iterations

## Success Criteria
- Clear identification of optimal RL warmup iterations
- RL-seeding shows improvement over pure methods
- Results publishable in paper

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/schedulers.py (RL-derived LR computation)
- experiments/runner.py (RL implementation)"

# Issue 5: Grid Search
gh issue create \
  --repo "$REPO" \
  --title "Run grid search: optimizer × scheduler × LR" \
  --label "experiment,grid-search" \
  --body "## Goal
Comprehensive grid search over optimizer, scheduler, and learning rate combinations.

## Grid Dimensions
- Optimizers: [sgd_momentum, adam, adamw, radam]
- Schedulers: [constant, cosine, cosine_warm_restarts]
- Learning rates: [1e-4, 1e-3, 1e-2]
- **Total combinations**: 4 × 3 × 3 = 36

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Max iterations: 1000

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment grid_search --output results/grid_search
\`\`\`

## Expected Output
- \`results/grid_search/grid_search_results.csv\`
- Heatmap: optimizer × scheduler (averaged over LR)
- Top 10 configurations ranked by PSNR

## Success Criteria
- All 36 combinations complete successfully
- Clear identification of best configuration
- Interaction effects visible (e.g., certain optimizer+scheduler pairs work especially well)

## Computational Notes
- Estimated time: ~1-2 hours on GPU
- Consider parallelization if possible

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/config.yaml"

# Issue 6: SNR Sweep
gh issue create \
  --repo "$REPO" \
  --title "Run SNR sweep: noise robustness analysis" \
  --label "experiment,robustness" \
  --body "## Goal
Test optimizer robustness across different noise levels (SNR).

## Noise Levels (SNR)
- 40 dB (low noise)
- 30 dB (medium noise)
- 20 dB (high noise)
- 10 dB (very high noise)

## Optimizers to Compare
- Adam
- AdamW
- RAdam
- SGD+momentum

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Scheduler: CosineAnnealingLR
- LR: 1e-3
- Max iterations: 1000

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment snr_sweep --output results/snr_sweep
\`\`\`

## Key Questions
1. Which optimizer is most robust to noise?
2. At what SNR do optimizers start to fail?
3. Does the ranking of optimizers change with noise level?

## Expected Output
- \`results/snr_sweep/snr_sweep_results.csv\`
- Plot: PSNR vs SNR for each optimizer
- Robustness ranking

## Success Criteria
- Clear separation of robust vs fragile optimizers
- All noise levels tested
- Actionable recommendations for noisy data

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/config.yaml"

# Issue 7: PSF Complexity Sweep
gh issue create \
  --repo "$REPO" \
  --title "Run PSF complexity sweep" \
  --label "experiment,psf" \
  --body "## Goal
Test optimizer performance on different PSF types and complexities.

## PSF Types
- Gaussian (σ=1.0, small blur)
- Gaussian (σ=2.0, medium blur)
- Gaussian (σ=4.0, large blur)
- Defocus aberration (simulated)

## Configuration
- Image: astronaut (downsampled 4x)
- Noise: Poisson (30 dB SNR)
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- LR: 1e-3
- Max iterations: 1000

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment psf_sweep --output results/psf_sweep
\`\`\`

## Key Questions
1. How does PSF complexity affect convergence?
2. Do larger blurs require more iterations?
3. Does aberrated PSF behave differently than Gaussian?

## Expected Output
- \`results/psf_sweep/psf_sweep_results.csv\`
- Comparison of convergence across PSF types
- Example restored images for each PSF

## Success Criteria
- All PSF types tested successfully
- Clear understanding of PSF impact on optimization

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/config.yaml
- utils.py (PSF generation)"

# Issue 8: Convergence Speed Analysis
gh issue create \
  --repo "$REPO" \
  --title "Analyze convergence speed vs final quality tradeoff" \
  --label "experiment,analysis" \
  --body "## Goal
Understand the tradeoff between convergence speed and final quality for different optimizers.

## Optimizers to Analyze
- SGD (baseline, slow but steady)
- SGD+momentum
- Adam (fast early, plateaus)
- AdamW
- RAdam (more stable)

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Scheduler: constant
- LR: 1e-3
- Max iterations: 2000 (longer to see asymptotic behavior)

## Metrics to Track
- Time to reach 80%, 90%, 95%, 99% of final PSNR
- PSNR at iterations: 10, 50, 100, 200, 500, 1000, 2000
- Compute time vs PSNR gain

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment convergence_analysis --output results/convergence_analysis
\`\`\`

## Key Questions
1. Which optimizer reaches 95% quality fastest?
2. Which optimizer has best final quality?
3. Is there a clear speed/quality tradeoff?

## Expected Output
- \`results/convergence_analysis/convergence_analysis_results.csv\`
- Scatter plot: convergence speed vs final PSNR
- Recommendation for \"fast prototyping\" vs \"publication quality\"

## Success Criteria
- Clear characterization of speed/quality tradeoff
- Actionable recommendations for different use cases

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/metrics.py (convergence speed computation)"

# Issue 9: RL-Derived Learning Rate
gh issue create \
  --repo "$REPO" \
  --title "Test RL-derived learning rate vs manual tuning" \
  --label "experiment,rl-lr,theory" \
  --body "## Goal
Validate the Richardson-Lucy derived learning rate against manual LR choices.

## LR Strategies
1. **RL-derived**: Computed from RL update magnitudes
2. Manual: 1e-4 (conservative)
3. Manual: 1e-3 (standard)
4. Manual: 1e-2 (aggressive)

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Optimizer: Adam
- Scheduler: constant
- Max iterations: 1000

## Theory
The RL algorithm uses multiplicative updates with implicit step size:
\`\`\`
x_{k+1} = x_k * (H^T(y / (Hx_k))) / (H^T 1)
\`\`\`

We approximate this as gradient descent:
\`\`\`
x_{k+1} ≈ x_k - α * ∇L(x_k)
\`\`\`

And estimate α by analyzing RL update vs gradient magnitudes.

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment rl_lr_comparison --output results/rl_lr_comparison

# Test RL-LR computation in isolation
python -c 'from experiments.schedulers import compute_rl_learning_rate; import numpy as np; from scipy.ndimage import gaussian_filter; psf = gaussian_filter(np.zeros((64,64)), 2.0); data = np.random.rand(128,128); lr = compute_rl_learning_rate(psf, data); print(f\"RL-derived LR: {lr:.6e}\")'
\`\`\`

## Key Questions
1. Is the RL-derived LR competitive with manual tuning?
2. Is it close to the optimal manual LR?
3. Does it save human tuning time?

## Expected Output
- \`results/rl_lr_comparison/rl_lr_comparison_results.csv\`
- Comparison of convergence for each LR strategy
- Theoretical validation of RL-LR formula

## Success Criteria
- RL-derived LR is within 1 dB PSNR of best manual LR
- Theory is validated empirically

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/schedulers.py (compute_rl_learning_rate function)
- Theory in README.md"

# Issue 10: LBFGS Special Case
gh issue create \
  --repo "$REPO" \
  --title "Test LBFGS optimizer (closure-based)" \
  --label "experiment,optimizer,lbfgs" \
  --body "## Goal
Test LBFGS optimizer which requires special handling (closure-based optimization).

## LBFGS Configurations
- LR=1.0, max_iter=20, history_size=100
- LR=0.5, max_iter=20, history_size=100
- LR=1.0, max_iter=10, history_size=50

## Configuration
- Image: astronaut (downsampled 4x)
- PSF: Gaussian (σ=2.0)
- Noise: Poisson (30 dB SNR)
- Max iterations: 100 (LBFGS uses fewer iterations)

## Special Handling
LBFGS requires a closure function that re-evaluates the model:
\`\`\`python
def closure():
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss

optimizer.step(closure)
\`\`\`

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.runner --config experiments/config.yaml --experiment lbfgs_experiment --output results/lbfgs_experiment
\`\`\`

## Key Questions
1. Does LBFGS converge faster (in iterations) than first-order methods?
2. What is the compute cost per iteration vs Adam?
3. Is LBFGS practical for this problem?

## Expected Output
- \`results/lbfgs_experiment/lbfgs_experiment_results.csv\`
- Comparison: LBFGS vs Adam (iterations and time)

## Success Criteria
- LBFGS runs successfully with closure
- Clear understanding of LBFGS tradeoffs

## Dependencies
- #<issue_number_1> (infrastructure setup)

## Related
- experiments/runner.py (LBFGS closure implementation)"

# Issue 11: Visualization & Analysis
gh issue create \
  --repo "$REPO" \
  --title "Generate visualizations and analysis report" \
  --label "analysis,visualization" \
  --body "## Goal
Create comprehensive visualizations and summary report from all experiment results.

## Visualizations to Create
1. **Convergence curves**: Loss/PSNR vs iteration for all experiments
2. **Final metrics bar charts**: Optimizer comparison, scheduler comparison
3. **Heatmaps**: Grid search results (optimizer × scheduler)
4. **SNR robustness curves**: PSNR vs noise level
5. **RL-seeding analysis**: Benefit of RL warmup vs pure methods
6. **Convergence speed scatter**: Speed vs final quality tradeoff
7. **Learning rate curves**: LR over time for different schedulers

## Summary Tables
1. Top 10 configurations (by PSNR)
2. Convergence speed ranking (iterations to 95% quality)
3. Compute efficiency ranking (PSNR gain per second)
4. Robustness ranking (performance across noise levels)

## Statistical Tests
- ANOVA: optimizer effect on PSNR
- Pairwise t-tests: each optimizer vs Adam (baseline)
- Bonferroni correction for multiple comparisons

## Commands
\`\`\`bash
cd ~/projects/ctr26/torch_image_restoration
python -m experiments.visualize --input results/ --output plots/

# Generate summary report
python -m experiments.report --input results/ --output report.pdf
\`\`\`

## Expected Output
- \`plots/\` directory with all figures (PNG, high DPI)
- \`report.pdf\`: Summary report with:
  - Executive summary
  - All visualizations
  - Statistical analysis
  - Recommendations

## Success Criteria
- All plots are publication-quality
- Report is comprehensive and actionable
- Clear recommendations for:
  - Best optimizer (general use)
  - Best scheduler (general use)
  - Best RL-seeding strategy
  - Noise-robust configuration
  - Fast prototyping vs final quality

## Dependencies
- All experiment issues (#2-#10)

## Related
- experiments/visualize.py (to be created)
- experiments/report.py (to be created)"

# Issue 12: Documentation & Paper
gh issue create \
  --repo "$REPO" \
  --title "Write paper/documentation on experimental findings" \
  --label "documentation,paper" \
  --body "## Goal
Document experimental methodology and findings for publication or technical report.

## Outline
1. **Introduction**
   - Problem: Image deconvolution
   - Motivation: Systematic optimizer comparison
   - Novel contribution: RL-seeding strategy

2. **Background**
   - Richardson-Lucy algorithm
   - Gradient descent methods
   - Theory: RL-derived learning rates

3. **Methods**
   - Experimental design (factorial, grid search)
   - Datasets (astronaut, PSFs, noise levels)
   - Optimizers and schedulers tested
   - Metrics (PSNR, SSIM, convergence speed)

4. **Results**
   - Optimizer comparison
   - Scheduler comparison
   - RL-seeding findings
   - SNR robustness
   - Statistical validation

5. **Discussion**
   - Practical recommendations
   - RL-seeding benefits
   - Future work

6. **Conclusion**

## Deliverables
- [ ] Technical report (LaTeX or Markdown)
- [ ] README update with key findings
- [ ] Tutorial notebook (Jupyter)
- [ ] API documentation (docstrings)

## Success Criteria
- Clear, reproducible methodology
- Well-supported conclusions
- Actionable recommendations for practitioners

## Dependencies
- All experiment issues (#2-#10)
- Visualization (#11)

## Related
- experiments/README.md
- All experiment results"

echo ""
echo "✓ All issues created successfully!"
echo ""
echo "Next steps:"
echo "1. Review issues on GitHub"
echo "2. Assign priorities and milestones"
echo "3. Start with infrastructure setup issue"
echo "4. Run experiments in order of dependencies"
