"""
Learning Rate Scheduler Configurations
========================================

Comprehensive suite of LR schedulers for image restoration experiments.
Includes theory-driven initialization from Richardson-Lucy (RL) algorithm.

References:
- Richardson (1972): Bayesian-based iterative method
- Lucy (1974): Maximum likelihood deconvolution
- Smith (2017): Cyclical learning rates (CLR)
- Loshchilov & Hutter (2017): SGDR with warm restarts
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any, Callable, Optional
import numpy as np


# =============================================================================
# Richardson-Lucy Derived Learning Rate
# =============================================================================

def compute_rl_learning_rate(psf: np.ndarray, data: np.ndarray, 
                            n_iterations: int = 10) -> float:
    """
    Derive initial learning rate from Richardson-Lucy theory.
    
    The RL algorithm uses multiplicative updates with implicit step size.
    We estimate an equivalent gradient descent LR by analyzing the update magnitude.
    
    Args:
        psf: Point spread function (convolution kernel)
        data: Observed blurred image
        n_iterations: Number of RL iterations to analyze
    
    Returns:
        Estimated learning rate for gradient-based methods
    
    Theory:
        RL update: x_{k+1} = x_k * (H^T(y / (Hx_k))) / (H^T 1)
        This can be approximated as: x_{k+1} ≈ x_k - α * ∇L(x_k)
        We estimate α by comparing RL updates to gradient magnitudes.
    """
    from scipy.signal import convolve2d
    from scipy.ndimage import correlate
    
    # Initialize with uniform estimate
    x = np.ones_like(data) * data.mean()
    
    # Run a few RL iterations and track update sizes
    update_ratios = []
    
    for i in range(n_iterations):
        # Forward model: Hx
        Hx = convolve2d(x, psf, mode='same')
        
        # Avoid division by zero
        Hx = np.maximum(Hx, 1e-10)
        
        # RL correction term
        ratio = data / Hx
        correction = correlate(ratio, psf, mode='constant')
        
        # Normalization (H^T 1)
        norm = correlate(np.ones_like(data), psf, mode='constant')
        norm = np.maximum(norm, 1e-10)
        
        # RL update
        x_new = x * correction / norm
        
        # Estimate equivalent gradient step size
        update = x_new - x
        grad_estimate = -update / (x + 1e-10)  # Approximate gradient
        
        # Ratio of update magnitude to gradient magnitude
        update_mag = np.abs(update).mean()
        grad_mag = np.abs(grad_estimate).mean()
        
        if grad_mag > 1e-10:
            alpha = update_mag / (grad_mag + 1e-10)
            update_ratios.append(alpha)
        
        x = x_new
    
    # Return median of estimated step sizes
    if len(update_ratios) > 0:
        return float(np.median(update_ratios))
    else:
        return 1e-3  # Fallback default


# =============================================================================
# Scheduler Factory Functions
# =============================================================================

def get_constant_lr(optimizer, **kwargs) -> lr_scheduler.ConstantLR:
    """Constant learning rate (baseline)."""
    return lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)


def get_step_lr(optimizer, step_size: int = 100, gamma: float = 0.5, **kwargs) -> lr_scheduler.StepLR:
    """Step decay: multiply LR by gamma every step_size epochs."""
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_exponential_lr(optimizer, gamma: float = 0.95, **kwargs) -> lr_scheduler.ExponentialLR:
    """Exponential decay: lr = lr * gamma^epoch."""
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def get_cosine_annealing_lr(optimizer, T_max: int = 100, eta_min: float = 0, **kwargs) -> lr_scheduler.CosineAnnealingLR:
    """Cosine annealing: smooth decay from initial LR to eta_min over T_max epochs."""
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


def get_cosine_annealing_warm_restarts(optimizer, T_0: int = 10, T_mult: int = 2, 
                                       eta_min: float = 0, **kwargs) -> lr_scheduler.CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts (SGDR)."""
    return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)


def get_reduce_on_plateau(optimizer, mode: str = 'min', factor: float = 0.5, 
                         patience: int = 10, threshold: float = 1e-4, **kwargs) -> lr_scheduler.ReduceLROnPlateau:
    """Reduce LR when metric plateaus."""
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                                         patience=patience, threshold=threshold)


def get_cyclic_lr(optimizer, base_lr: float = 1e-4, max_lr: float = 1e-2, 
                 step_size_up: int = 50, mode: str = 'triangular', **kwargs) -> lr_scheduler.CyclicLR:
    """Cyclical learning rates (CLR) - oscillate between base and max LR."""
    return lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, 
                                step_size_up=step_size_up, mode=mode)


def get_one_cycle_lr(optimizer, max_lr: float = 1e-2, total_steps: int = 1000, 
                    pct_start: float = 0.3, anneal_strategy: str = 'cos', **kwargs) -> lr_scheduler.OneCycleLR:
    """1cycle policy: ramp up to max_lr, then anneal down."""
    return lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, 
                                  pct_start=pct_start, anneal_strategy=anneal_strategy)


# =============================================================================
# Custom RL-Seeded Scheduler
# =============================================================================

class RLSeededScheduler:
    """
    Custom scheduler that starts with RL-derived learning rate,
    then optionally switches to another scheduler after warmup.
    
    This implements the RL-seeding strategy: use RL theory to initialize,
    then refine with gradient descent.
    """
    
    def __init__(self, optimizer, rl_lr: float, warmup_iters: int = 0,
                 post_scheduler: Optional[lr_scheduler._LRScheduler] = None):
        """
        Args:
            optimizer: PyTorch optimizer
            rl_lr: Learning rate derived from Richardson-Lucy
            warmup_iters: Number of iterations to use RL-derived LR
            post_scheduler: Optional scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.rl_lr = rl_lr
        self.warmup_iters = warmup_iters
        self.post_scheduler = post_scheduler
        self.current_iter = 0
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rl_lr
    
    def step(self, *args, **kwargs):
        """Step the scheduler."""
        self.current_iter += 1
        
        if self.current_iter < self.warmup_iters:
            # Keep RL-derived LR during warmup
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.rl_lr
        elif self.post_scheduler is not None:
            # Switch to post-warmup scheduler
            self.post_scheduler.step(*args, **kwargs)
    
    def get_last_lr(self):
        """Get current learning rate."""
        if self.current_iter < self.warmup_iters:
            return [self.rl_lr] * len(self.optimizer.param_groups)
        elif self.post_scheduler is not None:
            return self.post_scheduler.get_last_lr()
        else:
            return [pg['lr'] for pg in self.optimizer.param_groups]


# =============================================================================
# Scheduler Registry
# =============================================================================

SCHEDULER_CONFIGS: Dict[str, Dict[str, Any]] = {
    'constant': {
        'name': 'ConstantLR',
        'factory': get_constant_lr,
        'description': 'Constant learning rate (baseline)',
        'params': {},
        'notes': 'No decay. Good baseline for comparison.',
    },
    
    'step': {
        'name': 'StepLR',
        'factory': get_step_lr,
        'description': 'Step decay every N epochs',
        'params': {'step_size': 100, 'gamma': 0.5},
        'notes': 'Simple periodic decay. Easy to tune.',
    },
    
    'exponential': {
        'name': 'ExponentialLR',
        'factory': get_exponential_lr,
        'description': 'Exponential decay',
        'params': {'gamma': 0.95},
        'notes': 'Smooth continuous decay. Can be too aggressive.',
    },
    
    'cosine': {
        'name': 'CosineAnnealingLR',
        'factory': get_cosine_annealing_lr,
        'description': 'Cosine annealing to minimum',
        'params': {'T_max': 100, 'eta_min': 0},
        'notes': 'Smooth decay. Popular for training neural networks.',
    },
    
    'cosine_warm_restarts': {
        'name': 'CosineAnnealingWarmRestarts',
        'factory': get_cosine_annealing_warm_restarts,
        'description': 'Cosine annealing with periodic restarts (SGDR)',
        'params': {'T_0': 10, 'T_mult': 2, 'eta_min': 0},
        'notes': 'Periodic restarts can escape local minima. Very effective.',
    },
    
    'reduce_on_plateau': {
        'name': 'ReduceLROnPlateau',
        'factory': get_reduce_on_plateau,
        'description': 'Reduce LR when loss plateaus',
        'params': {'mode': 'min', 'factor': 0.5, 'patience': 10},
        'notes': 'Adaptive to training dynamics. Requires metric monitoring.',
        'requires_metric': True,
    },
    
    'cyclic': {
        'name': 'CyclicLR',
        'factory': get_cyclic_lr,
        'description': 'Cyclical learning rates (triangular)',
        'params': {'base_lr': 1e-4, 'max_lr': 1e-2, 'step_size_up': 50, 'mode': 'triangular'},
        'notes': 'Oscillates between bounds. Can improve generalization.',
    },
    
    'one_cycle': {
        'name': 'OneCycleLR',
        'factory': get_one_cycle_lr,
        'description': '1cycle policy: ramp up then anneal',
        'params': {'max_lr': 1e-2, 'total_steps': 1000, 'pct_start': 0.3},
        'notes': 'Highly effective. Used in fast.ai. Requires total_steps.',
        'requires_total_steps': True,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_scheduler(name: str, optimizer, **override_kwargs) -> lr_scheduler._LRScheduler:
    """
    Get a scheduler by name with optional parameter overrides.
    
    Args:
        name: Scheduler name from SCHEDULER_CONFIGS
        optimizer: PyTorch optimizer instance
        **override_kwargs: Override default parameters
    
    Returns:
        Configured scheduler instance
    
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = get_scheduler('cosine', optimizer, T_max=200)
    """
    if name not in SCHEDULER_CONFIGS:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(SCHEDULER_CONFIGS.keys())}")
    
    config = SCHEDULER_CONFIGS[name]
    factory = config['factory']
    
    # Merge default params with overrides
    sched_params = {**config['params'], **override_kwargs}
    
    return factory(optimizer, **sched_params)


def get_rl_seeded_scheduler(optimizer, psf: np.ndarray, data: np.ndarray,
                           warmup_iters: int = 10, post_scheduler_name: str = 'cosine',
                           **scheduler_kwargs) -> RLSeededScheduler:
    """
    Create RL-seeded scheduler with optional post-warmup scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        psf: Point spread function
        data: Observed data
        warmup_iters: Number of iterations to use RL-derived LR
        post_scheduler_name: Scheduler to use after warmup (or None)
        **scheduler_kwargs: Additional params for post-scheduler
    
    Returns:
        RLSeededScheduler instance
    """
    # Compute RL-derived learning rate
    rl_lr = compute_rl_learning_rate(psf, data)
    
    # Create post-warmup scheduler if specified
    post_scheduler = None
    if post_scheduler_name is not None:
        post_scheduler = get_scheduler(post_scheduler_name, optimizer, **scheduler_kwargs)
    
    return RLSeededScheduler(optimizer, rl_lr, warmup_iters, post_scheduler)


def list_schedulers() -> None:
    """Print all available schedulers with descriptions."""
    print("Available Learning Rate Schedulers:")
    print("=" * 80)
    for key, config in SCHEDULER_CONFIGS.items():
        print(f"\n{key:20s} | {config['name']}")
        print(f"{'':20s} | {config['description']}")
        print(f"{'':20s} | {config['notes']}")
    print("\n" + "=" * 80)


def plot_scheduler_curves(optimizer_class=optim.SGD, initial_lr: float = 0.1, 
                         epochs: int = 100, save_path: Optional[str] = None):
    """
    Visualize learning rate schedules over time.
    
    Args:
        optimizer_class: Optimizer class to use (just for testing)
        initial_lr: Initial learning rate
        epochs: Number of epochs to simulate
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt
    
    # Dummy model parameter
    param = torch.tensor([1.0], requires_grad=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, config) in enumerate(SCHEDULER_CONFIGS.items()):
        if i >= len(axes):
            break
            
        # Skip schedulers that require special handling
        if config.get('requires_metric') or config.get('requires_total_steps'):
            axes[i].text(0.5, 0.5, f"{config['name']}\n(requires special args)", 
                        ha='center', va='center')
            axes[i].set_title(name)
            continue
        
        # Create optimizer and scheduler
        opt = optimizer_class([param], lr=initial_lr)
        sched = get_scheduler(name, opt)
        
        # Track LR over epochs
        lrs = []
        for epoch in range(epochs):
            lrs.append(opt.param_groups[0]['lr'])
            
            # Dummy step
            opt.zero_grad()
            loss = param ** 2
            loss.backward()
            opt.step()
            sched.step()
        
        # Plot
        axes[i].plot(lrs, linewidth=2)
        axes[i].set_title(f"{name}: {config['name']}")
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Learning Rate')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scheduler curves to: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    list_schedulers()
    
    print("\n" + "="*80)
    print("Testing RL-derived learning rate computation...")
    print("="*80)
    
    # Create synthetic test case
    from scipy.ndimage import gaussian_filter
    psf_test = np.zeros((64, 64))
    psf_test[32, 32] = 1.0
    psf_test = gaussian_filter(psf_test, sigma=2.0)
    data_test = np.random.rand(128, 128)
    
    rl_lr = compute_rl_learning_rate(psf_test, data_test, n_iterations=10)
    print(f"Estimated RL-derived learning rate: {rl_lr:.6e}")
    
    print("\n" + "="*80)
    print("Generating scheduler visualization...")
    print("="*80)
    # plot_scheduler_curves(epochs=100, save_path='scheduler_curves.png')
