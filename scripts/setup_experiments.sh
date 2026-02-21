#!/bin/bash
# Setup script for experiment infrastructure
set -e

echo "🔧 Setting up experiment infrastructure..."

# Create results directory structure
echo "📁 Creating results directory structure..."
mkdir -p results/{optimizer_comparison,scheduler_comparison,grid_search,snr_sweep,psf_sweep,convergence_analysis,rl_seeding,rl_lr_comparison,lbfgs_experiment}
mkdir -p results/plots
mkdir -p results/checkpoints

echo "✅ Directory structure created:"
tree -L 2 results/ || ls -la results/

# Verify Python version
echo ""
echo "🐍 Checking Python version..."
python --version

# Install dependencies (if requested)
if [ "$1" == "--install" ]; then
    echo ""
    echo "📦 Installing dependencies..."
    
    # Try uv first, fall back to pip
    if command -v uv &> /dev/null; then
        echo "Using uv..."
        uv pip install -r requirements.txt
    else
        echo "Using pip..."
        pip install -r requirements.txt
    fi
fi

# Verify imports
echo ""
echo "🔍 Verifying dependencies..."
python -c "
import sys
import importlib.util

required_modules = [
    'torch',
    'numpy',
    'scipy',
    'skimage',
    'yaml',
    'pandas',
    'matplotlib',
    'tqdm'
]

missing = []
for module in required_modules:
    # Handle special cases
    module_name = 'scikit-image' if module == 'skimage' else module
    module_import = module
    
    spec = importlib.util.find_spec(module_import)
    if spec is None:
        missing.append(module_name)
        print(f'❌ {module_name} - NOT FOUND')
    else:
        print(f'✅ {module_name} - OK')

if missing:
    print(f'\n⚠️  Missing modules: {", ".join(missing)}')
    print('Run: pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm')
    sys.exit(1)
else:
    print('\n✅ All dependencies installed!')
"

# Test experiment modules
echo ""
echo "🧪 Testing experiment modules..."
python -c "
from experiments import optimizers, schedulers, runner, metrics
print('✅ experiments.optimizers - OK')
print('✅ experiments.schedulers - OK')
print('✅ experiments.runner - OK')
print('✅ experiments.metrics - OK')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run quick test: python scripts/test_experiment.py"
echo "  2. Run single experiment: python -m experiments.runner --experiment rl_seeding"
echo "  3. Run all experiments: python -m experiments.runner --config experiments/config.yaml"
