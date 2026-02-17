#!/bin/bash
# Create GitHub issues for experiment execution (without labels for now)

REPO="ctr26/torch_image_restoration"

echo "Creating GitHub issues for optimizer/scheduler experiments..."

# Issue 1: Infrastructure Setup
gh issue create \
  --repo "$REPO" \
  --title "Setup experiment infrastructure" \
  --body "## Goal
Setup the infrastructure for running optimizer/scheduler experiments.

## Tasks
- [ ] Verify all dependencies are installed
- [ ] Test experiment runner on small example
- [ ] Setup results directory structure

## Commands
\`\`\`bash
pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm
python -m experiments.optimizers
python -m experiments.schedulers
\`\`\`"

# Issue 2-12: Use similar pattern but shorter for brevity
for i in 2 3 4 5 6 7 8 9 10 11 12; do
    case $i in
        2) TITLE="Run optimizer comparison experiment"; DESC="Compare 10 optimizers with fixed LR";;
        3) TITLE="Run scheduler comparison experiment"; DESC="Compare 8 schedulers with Adam";;
        4) TITLE="Run RL-seeding experiments"; DESC="Test RL warmup + gradient descent";;
        5) TITLE="Run grid search experiment"; DESC="optimizer × scheduler × LR grid";;
        6) TITLE="Run SNR sweep experiment"; DESC="Noise robustness testing";;
        7) TITLE="Run PSF complexity sweep"; DESC="Different PSF types";;
        8) TITLE="Run convergence speed analysis"; DESC="Speed vs quality tradeoff";;
        9) TITLE="Test RL-derived learning rate"; DESC="Theory validation";;
        10) TITLE="Test LBFGS optimizer"; DESC="Closure-based optimization";;
        11) TITLE="Generate visualizations and analysis"; DESC="Plots and summary tables";;
        12) TITLE="Write documentation and paper"; DESC="Publication-ready report";;
    esac
    
    gh issue create \
      --repo "$REPO" \
      --title "$TITLE" \
      --body "## Goal
$DESC

See \`experiments/config.yaml\` for full configuration.

## Commands
\`\`\`bash
python -m experiments.runner --experiment <experiment_name> --output results/
\`\`\`"
    
    echo "Created issue $i: $TITLE"
done

echo ""
echo "✓ All 12 issues created!"
echo "View at: https://github.com/$REPO/issues"
