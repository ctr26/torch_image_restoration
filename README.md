# Torch Image Restoration

PyTorch-based image restoration framework with comprehensive optimizer and scheduler experiments.

## Features

- **10 Optimizers**: SGD, SGD+Momentum, Adam, AdamW, RMSprop, LBFGS, Adagrad, Adadelta, NAdam, RAdam
- **8+ Schedulers**: Constant, StepLR, Exponential, Cosine Annealing, SGDR, ReduceLROnPlateau, CyclicLR, OneCycleLR
- **Richardson-Lucy Integration**: Theory-driven learning rate initialization
- **RL-Seeding Strategy**: Warm-start gradient descent with RL iterations
- **Comprehensive Metrics**: PSNR, SSIM, MSE, MAE, convergence analysis

## Quick Start

### 1. Installation

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

Run the verification script to ensure everything is configured correctly:

```bash
python verify_setup.py
```

This will check:
- All dependencies are installed
- Experiment modules can be imported
- Directory structure is correct
- Basic functionality works

### 3. Run Experiments

Test individual modules:

```bash
# List available optimizers
python -m experiments.optimizers

# List available schedulers
python -m experiments.schedulers

# Run full experiment suite
python -m experiments.runner --config experiments/config.yaml
```

## Project Structure

```
torch_image_restoration/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── verify_setup.py             # Setup verification script
├── environment.yml             # Conda environment (legacy)
├── experiments/                # Experiment framework
│   ├── __init__.py
│   ├── optimizers.py          # Optimizer configurations
│   ├── schedulers.py          # Scheduler configurations
│   ├── runner.py              # Experiment orchestration
│   ├── metrics.py             # Evaluation metrics
│   ├── config.yaml            # Experiment definitions
│   └── README.md              # Detailed experiment docs
├── results/                    # Experiment results (CSV files)
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
└── utils.py                    # Helper functions
```

## Documentation

- **[Experiment Framework Documentation](experiments/README.md)**: Detailed guide to optimizers, schedulers, and experiment configurations
- **[Quick Start Guide](experiments/QUICKSTART.md)**: Get started quickly with common use cases

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-image
- PyYAML, Pandas, Matplotlib, tqdm

See `requirements.txt` for complete list.

## Dependencies Installation

### Using pip (recommended)

```bash
pip install -r requirements.txt
```

### Using conda (alternative)

```bash
conda env create -f environment.yml
conda activate 2021_image_restoration
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_optimizers.py

# Run with coverage
pytest --cov=experiments tests/
```

## Usage Example

```python
from experiments.runner import train_deconvolution
from experiments.optimizers import get_optimizer
from experiments.schedulers import get_scheduler
import torch

# Setup data
observed = ...  # Blurred+noisy image (torch.Tensor)
psf = ...       # Point spread function (torch.Tensor)
ground_truth = ...  # Clean image (torch.Tensor)

# Train with Adam + Cosine Annealing
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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/my-feature`)
7. Create a Pull Request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{torch_image_restoration,
  title = {Torch Image Restoration},
  author = {[Your name]},
  year = {2026},
  url = {https://github.com/ctr26/torch_image_restoration}
}
```

## References

### Core Methods
- Richardson (1972): "Bayesian-Based Iterative Method of Image Restoration"
- Lucy (1974): "An Iterative Technique for the Rectification of Observed Distributions"

### Optimizers
- Kingma & Ba (2015): "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization" (AdamW)
- Liu et al. (2020): "On the Variance of the Adaptive Learning Rate" (RAdam)

### Learning Rate Schedules
- Smith (2017): "Cyclical Learning Rates for Training Neural Networks"
- Loshchilov & Hutter (2017): "SGDR: Stochastic Gradient Descent with Warm Restarts"
- Smith & Topin (2019): "Super-Convergence" (1cycle policy)

## Contact

[Add your contact information or link to issues page]
