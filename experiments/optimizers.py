"""
Optimizer Configurations
========================

Comprehensive suite of PyTorch optimizers for image restoration experiments.
Each optimizer is configured with sensible defaults based on literature and practice.

References:
- Kingma & Ba (2015): Adam optimizer
- Loshchilov & Hutter (2019): AdamW, decoupled weight decay
- Liu & Nocedal (1989): LBFGS
- Dozat (2016): NAdam
- Liu et al. (2020): RAdam (Rectified Adam)
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Callable


# =============================================================================
# Optimizer Factory Functions
# =============================================================================

def get_sgd(params, lr: float = 1e-3, **kwargs) -> optim.SGD:
    """Vanilla SGD - baseline optimizer."""
    return optim.SGD(params, lr=lr, **kwargs)


def get_sgd_momentum(params, lr: float = 1e-3, momentum: float = 0.9, **kwargs) -> optim.SGD:
    """SGD with momentum (Polyak 1964)."""
    return optim.SGD(params, lr=lr, momentum=momentum, **kwargs)


def get_adam(params, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8, **kwargs) -> optim.Adam:
    """Adam - adaptive moment estimation."""
    return optim.Adam(params, lr=lr, betas=betas, eps=eps, **kwargs)


def get_adamw(params, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, **kwargs) -> optim.AdamW:
    """AdamW - Adam with decoupled weight decay regularization."""
    return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)


def get_rmsprop(params, lr: float = 1e-3, alpha=0.99, eps=1e-8, momentum=0, **kwargs) -> optim.RMSprop:
    """RMSprop - root mean square propagation."""
    return optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, momentum=momentum, **kwargs)


def get_lbfgs(params, lr: float = 1.0, max_iter=20, history_size=100, **kwargs) -> optim.LBFGS:
    """
    L-BFGS - limited-memory Broyden–Fletcher–Goldfarb–Shanno.
    
    Note: LBFGS requires closure-based optimization. Use with caution in 
    mini-batch settings. Best for full-batch optimization on small problems.
    """
    return optim.LBFGS(params, lr=lr, max_iter=max_iter, history_size=history_size, **kwargs)


def get_adagrad(params, lr: float = 1e-2, lr_decay=0, eps=1e-10, **kwargs) -> optim.Adagrad:
    """Adagrad - adaptive gradient algorithm."""
    return optim.Adagrad(params, lr=lr, lr_decay=lr_decay, eps=eps, **kwargs)


def get_adadelta(params, lr: float = 1.0, rho=0.9, eps=1e-6, **kwargs) -> optim.Adadelta:
    """Adadelta - adaptive learning rate method (extension of Adagrad)."""
    return optim.Adadelta(params, lr=lr, rho=rho, eps=eps, **kwargs)


def get_nadam(params, lr: float = 2e-3, betas=(0.9, 0.999), eps=1e-8, **kwargs) -> optim.NAdam:
    """NAdam - Nesterov-accelerated Adam."""
    return optim.NAdam(params, lr=lr, betas=betas, eps=eps, **kwargs)


def get_radam(params, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8, **kwargs) -> optim.RAdam:
    """RAdam - Rectified Adam (variance warmup)."""
    return optim.RAdam(params, lr=lr, betas=betas, eps=eps, **kwargs)


# =============================================================================
# Optimizer Registry
# =============================================================================

OPTIMIZER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'sgd': {
        'name': 'SGD',
        'factory': get_sgd,
        'default_lr': 1e-3,
        'description': 'Vanilla stochastic gradient descent (baseline)',
        'params': {},
        'notes': 'Simple baseline. May require careful LR tuning.',
    },
    
    'sgd_momentum': {
        'name': 'SGD+Momentum',
        'factory': get_sgd_momentum,
        'default_lr': 1e-3,
        'description': 'SGD with Polyak momentum (0.9)',
        'params': {'momentum': 0.9},
        'notes': 'Classic choice. Smooths gradient updates.',
    },
    
    'adam': {
        'name': 'Adam',
        'factory': get_adam,
        'default_lr': 1e-3,
        'description': 'Adaptive moment estimation',
        'params': {'betas': (0.9, 0.999), 'eps': 1e-8},
        'notes': 'Most popular adaptive method. Good default choice.',
    },
    
    'adamw': {
        'name': 'AdamW',
        'factory': get_adamw,
        'default_lr': 1e-3,
        'description': 'Adam with decoupled weight decay',
        'params': {'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-2},
        'notes': 'Better regularization than Adam. Recommended for modern work.',
    },
    
    'rmsprop': {
        'name': 'RMSprop',
        'factory': get_rmsprop,
        'default_lr': 1e-3,
        'description': 'Root mean square propagation',
        'params': {'alpha': 0.99, 'eps': 1e-8},
        'notes': 'Good for non-stationary objectives (e.g., RL).',
    },
    
    'lbfgs': {
        'name': 'L-BFGS',
        'factory': get_lbfgs,
        'default_lr': 1.0,
        'description': 'Limited-memory BFGS (quasi-Newton)',
        'params': {'max_iter': 20, 'history_size': 100},
        'notes': 'Requires closure. Best for small problems with full batch.',
        'requires_closure': True,
    },
    
    'adagrad': {
        'name': 'Adagrad',
        'factory': get_adagrad,
        'default_lr': 1e-2,
        'description': 'Adaptive gradient algorithm',
        'params': {'lr_decay': 0, 'eps': 1e-10},
        'notes': 'Accumulates squared gradients. LR decays monotonically.',
    },
    
    'adadelta': {
        'name': 'Adadelta',
        'factory': get_adadelta,
        'default_lr': 1.0,
        'description': 'Extension of Adagrad with windowed accumulation',
        'params': {'rho': 0.9, 'eps': 1e-6},
        'notes': 'No manual LR required. More robust than Adagrad.',
    },
    
    'nadam': {
        'name': 'NAdam',
        'factory': get_nadam,
        'default_lr': 2e-3,
        'description': 'Nesterov-accelerated Adam',
        'params': {'betas': (0.9, 0.999), 'eps': 1e-8},
        'notes': 'Adam + Nesterov momentum. Can converge faster than Adam.',
    },
    
    'radam': {
        'name': 'RAdam',
        'factory': get_radam,
        'default_lr': 1e-3,
        'description': 'Rectified Adam (variance warmup)',
        'params': {'betas': (0.9, 0.999), 'eps': 1e-8},
        'notes': 'Fixes early-training instability in Adam. More robust.',
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_optimizer(name: str, params, lr: float = None, **override_kwargs) -> optim.Optimizer:
    """
    Get an optimizer by name with optional parameter overrides.
    
    Args:
        name: Optimizer name from OPTIMIZER_CONFIGS
        params: Model parameters to optimize
        lr: Learning rate (uses default_lr if None)
        **override_kwargs: Override default parameters
    
    Returns:
        Configured optimizer instance
    
    Example:
        >>> model = MyModel()
        >>> opt = get_optimizer('adam', model.parameters(), lr=1e-4)
    """
    if name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_CONFIGS.keys())}")
    
    config = OPTIMIZER_CONFIGS[name]
    factory = config['factory']
    
    # Use provided LR or default
    lr = lr if lr is not None else config['default_lr']
    
    # Merge default params with overrides
    opt_params = {**config['params'], **override_kwargs}
    
    return factory(params, lr=lr, **opt_params)


def list_optimizers() -> None:
    """Print all available optimizers with descriptions."""
    print("Available Optimizers:")
    print("=" * 80)
    for key, config in OPTIMIZER_CONFIGS.items():
        print(f"\n{key:15s} | {config['name']}")
        print(f"{'':15s} | LR: {config['default_lr']}")
        print(f"{'':15s} | {config['description']}")
        print(f"{'':15s} | {config['notes']}")
    print("\n" + "=" * 80)


def benchmark_optimizers():
    """
    Benchmark all optimizers on a simple quadratic problem.
    Useful for sanity-checking configurations.
    """
    import time
    print("Benchmarking optimizers on quadratic: f(x) = x^2")
    print("=" * 80)
    
    results = {}
    for name in OPTIMIZER_CONFIGS.keys():
        # Skip LBFGS for simple benchmark (requires closure)
        if name == 'lbfgs':
            continue
            
        x = torch.tensor([10.0], requires_grad=True)
        opt = get_optimizer(name, [x])
        
        start = time.time()
        for _ in range(100):
            opt.zero_grad()
            loss = x ** 2
            loss.backward()
            opt.step()
        elapsed = time.time() - start
        
        results[name] = {
            'final_x': x.item(),
            'time': elapsed * 1000,  # ms
        }
        
        print(f"{name:15s} | final x: {x.item():8.4f} | time: {elapsed*1000:6.2f}ms")
    
    print("=" * 80)
    return results


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    list_optimizers()
    print("\n")
    benchmark_optimizers()
