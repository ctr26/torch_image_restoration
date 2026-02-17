"""
Experiment Runner
==================

Orchestrates execution of optimizer/scheduler experiments defined in config.yaml.
Handles data loading, model setup, training loops, and metrics collection.
"""

import torch
import numpy as np
import yaml
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from tqdm import tqdm

# Local imports
from .optimizers import get_optimizer, OPTIMIZER_CONFIGS
from .schedulers import get_scheduler, get_rl_seeded_scheduler, compute_rl_learning_rate, SCHEDULER_CONFIGS


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ExperimentResult:
    """Container for single experiment results."""
    experiment_name: str
    config: Dict[str, Any]
    
    # Final metrics
    final_psnr: float
    final_ssim: float
    final_mse: float
    final_mae: float
    
    # Convergence metrics
    iterations: int
    converged: bool
    convergence_iteration: Optional[int]
    
    # Timing
    total_time: float
    time_per_iteration: float
    
    # Training history
    loss_history: List[float]
    psnr_history: List[float]
    lr_history: List[float]
    
    # Additional metadata
    timestamp: str
    device: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return asdict(self)


# =============================================================================
# Data Generation
# =============================================================================

def load_test_image(config: Dict[str, Any]) -> np.ndarray:
    """Load test image based on configuration."""
    from skimage import data, color
    from skimage.transform import rescale
    from skimage.util import img_as_float
    
    name = config['name']
    downsample = config.get('downsample_factor', 1)
    colorspace = config.get('colorspace', 'gray')
    
    # Load image
    if name == 'astronaut':
        img = data.astronaut()
    elif name == 'cameraman' or name == 'camera':
        img = data.camera()
    elif name == 'text':
        img = data.text()
    else:
        raise ValueError(f"Unknown image: {name}")
    
    # Normalize to [0, 1] float (handles uint8, uint16, etc.)
    img = img_as_float(img)
    
    # Convert to grayscale if needed
    if colorspace == 'gray' and img.ndim == 3:
        img = color.rgb2gray(img)
    
    # Downsample
    if downsample != 1:
        img = rescale(img, 1.0 / downsample, anti_aliasing=True)
    
    return img


def generate_psf(config: Dict[str, Any]) -> np.ndarray:
    """Generate PSF based on configuration."""
    psf_type = config['type']
    size = config['size']
    gain = config.get('gain', 1000)
    
    psf = np.zeros(size)
    center = (size[0] // 2, size[1] // 2)
    
    if psf_type == 'gaussian':
        sigma = config['sigma']
        psf[center] = gain
        psf = gaussian_filter(psf, sigma=sigma)
        psf = psf / psf.sum()  # Normalize to sum=1
    
    elif psf_type == 'defocus':
        # Simulated defocus (disk)
        radius = config['radius']
        y, x = np.ogrid[:size[0], :size[1]]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        psf[dist <= radius] = gain
        psf = psf / psf.sum()  # Normalize
    
    else:
        raise ValueError(f"Unknown PSF type: {psf_type}")
    
    return psf


def add_noise(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Add noise to image based on configuration."""
    noise_type = config['type']
    
    if noise_type == 'poisson':
        # Poisson noise (photon counting)
        # Scale up for realistic photon counts, apply noise, scale back
        peak_photons = config.get('peak_photons', 1000)  # configurable SNR
        rng = np.random.default_rng()
        scaled = image * peak_photons
        noisy = rng.poisson(scaled).astype(float) / peak_photons
        return noisy
    
    elif noise_type == 'gaussian':
        snr_db = config['snr_db']
        signal_power = np.mean(image ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(*image.shape) * np.sqrt(noise_power)
        return image + noise
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def generate_observation(ground_truth: np.ndarray, psf: np.ndarray, noise_config: Dict[str, Any]) -> np.ndarray:
    """Generate blurred and noisy observation from ground truth."""
    # Convolve with PSF
    blurred = convolve2d(ground_truth, psf, mode='same', boundary='symm')
    
    # Scale to target SNR if specified
    if 'snr_db' in noise_config:
        snr_linear = 10 ** (noise_config['snr_db'] / 10)
        blurred = blurred * snr_linear / (blurred.max() + 1e-10)
    
    # Add noise
    observed = add_noise(blurred, noise_config)
    
    return np.maximum(observed, 0)  # Ensure non-negative


# =============================================================================
# Richardson-Lucy Implementation
# =============================================================================

def richardson_lucy_iteration(image: np.ndarray, psf: np.ndarray, 
                              estimate: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Single Richardson-Lucy iteration."""
    # Forward model
    convolved = convolve2d(estimate, psf, mode='same', boundary='symm')
    
    # Avoid division by zero
    convolved = np.maximum(convolved, epsilon)
    
    # Ratio
    ratio = image / convolved
    
    # Backproject
    psf_flipped = np.flip(psf)
    correction = convolve2d(ratio, psf_flipped, mode='same', boundary='symm')
    
    # Update
    estimate_new = estimate * correction
    
    return estimate_new


def run_richardson_lucy(image: np.ndarray, psf: np.ndarray, 
                       n_iterations: int) -> np.ndarray:
    """Run Richardson-Lucy deconvolution."""
    # Initialize with uniform estimate
    estimate = np.ones_like(image) * image.mean()
    
    for i in range(n_iterations):
        estimate = richardson_lucy_iteration(image, psf, estimate)
    
    return estimate


# =============================================================================
# PyTorch Training Loop
# =============================================================================

class DeconvolutionModel(torch.nn.Module):
    """Simple deconvolution model: learnable image."""
    
    def __init__(self, image_shape: Tuple[int, int], device: str = 'cpu'):
        super().__init__()
        self.image = torch.nn.Parameter(
            torch.randn(image_shape, device=device, dtype=torch.float32)
        )
    
    def forward(self):
        return self.image


def train_deconvolution(
    observed: np.ndarray,
    psf: np.ndarray,
    ground_truth: np.ndarray,
    optimizer_name: str,
    scheduler_name: str,
    lr: float,
    max_iterations: int,
    device: str = 'cpu',
    rl_warmup_iterations: int = 0,
    use_rl_derived_lr: bool = False,
    convergence_threshold: float = 1e-4,
    early_stopping_patience: int = 50,
    **kwargs
) -> ExperimentResult:
    """
    Train deconvolution model with specified optimizer and scheduler.
    
    Args:
        observed: Blurred+noisy observation
        psf: Point spread function
        ground_truth: Clean image (for metrics)
        optimizer_name: Name of optimizer to use
        scheduler_name: Name of scheduler to use
        lr: Learning rate
        max_iterations: Maximum training iterations
        device: 'cpu' or 'cuda'
        rl_warmup_iterations: Number of RL iterations before gradient descent
        use_rl_derived_lr: Use RL-derived LR instead of provided lr
        convergence_threshold: Stop if loss change < threshold
        early_stopping_patience: Stop if no improvement for N iterations
        **kwargs: Additional experiment metadata
    
    Returns:
        ExperimentResult object
    """
    start_time = time.time()
    
    # RL warm-start if requested
    if rl_warmup_iterations > 0:
        logging.info(f"Running {rl_warmup_iterations} RL warm-up iterations...")
        initial_estimate = run_richardson_lucy(observed, psf, rl_warmup_iterations)
    else:
        initial_estimate = np.ones_like(observed) * observed.mean()
    
    # Compute RL-derived LR if requested
    if use_rl_derived_lr:
        lr = compute_rl_learning_rate(psf, observed, n_iterations=10)
        logging.info(f"Using RL-derived learning rate: {lr:.6e}")
    
    # Setup PyTorch model
    model = DeconvolutionModel(observed.shape, device=device)
    
    # Initialize with RL estimate
    with torch.no_grad():
        model.image.data = torch.tensor(initial_estimate, device=device, dtype=torch.float32)
    
    # Setup optimizer
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
    
    # Setup scheduler
    if scheduler_name == 'rl_seeded':
        # Custom RL-seeded scheduler
        scheduler = get_rl_seeded_scheduler(
            optimizer, psf, observed, 
            warmup_iters=rl_warmup_iterations,
            post_scheduler_name='cosine'
        )
    else:
        scheduler = get_scheduler(scheduler_name, optimizer)
    
    # Convert data to tensors
    observed_tensor = torch.tensor(observed, device=device, dtype=torch.float32)
    psf_tensor = torch.tensor(psf, device=device, dtype=torch.float32)
    ground_truth_tensor = torch.tensor(ground_truth, device=device, dtype=torch.float32)
    
    # Training history
    loss_history = []
    psnr_history = []
    lr_history = []
    
    # Early stopping trackers
    best_loss = float('inf')
    patience_counter = 0
    converged = False
    convergence_iteration = None
    
    # Training loop
    pbar = tqdm(range(max_iterations), desc=f"{optimizer_name}/{scheduler_name}")
    
    for iteration in pbar:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: convolve estimate with PSF
        estimate = model()
        
        # FFT-based convolution for exact 'same' output
        # Zero-pad both to avoid circular convolution artifacts
        target_h, target_w = observed_tensor.shape
        fft_h, fft_w = target_h + psf_tensor.shape[0], target_w + psf_tensor.shape[1]
        
        est_padded = torch.zeros(fft_h, fft_w, device=device)
        est_padded[:target_h, :target_w] = estimate
        psf_padded = torch.zeros(fft_h, fft_w, device=device)
        psf_padded[:psf_tensor.shape[0], :psf_tensor.shape[1]] = psf_tensor
        
        # FFT convolution
        est_fft = torch.fft.fft2(est_padded)
        psf_fft = torch.fft.fft2(psf_padded)
        conv_fft = est_fft * psf_fft
        convolved_full = torch.fft.ifft2(conv_fft).real
        
        # Extract 'same' region
        start_h, start_w = psf_tensor.shape[0] // 2, psf_tensor.shape[1] // 2
        convolved = convolved_full[start_h:start_h + target_h, start_w:start_w + target_w]
        
        # Loss: MSE between observation and forward model
        loss = torch.nn.functional.mse_loss(convolved, observed_tensor)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        if optimizer_name == 'lbfgs':
            # LBFGS requires closure
            def closure():
                optimizer.zero_grad()
                estimate = model()
                convolved = torch.nn.functional.conv2d(
                    estimate.unsqueeze(0).unsqueeze(0),
                    psf_tensor.unsqueeze(0).unsqueeze(0),
                    padding=(psf_tensor.shape[0]//2, psf_tensor.shape[1]//2)
                ).squeeze()
                loss = torch.nn.functional.mse_loss(convolved, observed_tensor)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer.step()
        
        # Scheduler step
        if scheduler_name != 'reduce_on_plateau':
            scheduler.step()
        else:
            scheduler.step(loss)
        
        # Compute metrics
        with torch.no_grad():
            current_estimate = model.image.cpu().numpy()
            current_estimate = np.clip(current_estimate, 0, 1)  # Clamp to valid range
            psnr = peak_signal_noise_ratio(ground_truth, current_estimate, data_range=1.0)
            
            # Record history
            loss_history.append(loss.item())
            psnr_history.append(psnr)
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'PSNR': f'{psnr:.2f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Early stopping check
        if loss.item() < best_loss - convergence_threshold:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            converged = True
            convergence_iteration = iteration
            logging.info(f"Early stopping at iteration {iteration}")
            break
    
    total_time = time.time() - start_time
    
    # Final metrics
    with torch.no_grad():
        final_estimate = model.image.cpu().numpy()
        final_estimate = np.clip(final_estimate, 0, 1)  # Clamp to valid range
        final_psnr = peak_signal_noise_ratio(ground_truth, final_estimate, data_range=1.0)
        final_ssim = structural_similarity(ground_truth, final_estimate, data_range=1.0)
        final_mse = np.mean((ground_truth - final_estimate) ** 2)
        final_mae = np.mean(np.abs(ground_truth - final_estimate))
    
    # Create result object
    result = ExperimentResult(
        experiment_name=kwargs.get('experiment_name', 'unknown'),
        config=kwargs,
        final_psnr=final_psnr,
        final_ssim=final_ssim,
        final_mse=final_mse,
        final_mae=final_mae,
        iterations=len(loss_history),
        converged=converged,
        convergence_iteration=convergence_iteration,
        total_time=total_time,
        time_per_iteration=total_time / len(loss_history) if len(loss_history) > 0 else 0,
        loss_history=loss_history,
        psnr_history=psnr_history,
        lr_history=lr_history,
        timestamp=datetime.now().isoformat(),
        device=device
    )
    
    return result


# =============================================================================
# Experiment Orchestration
# =============================================================================

def run_experiment_suite(config_path: str = 'config.yaml', 
                        output_dir: str = './results') -> pd.DataFrame:
    """
    Run all experiments defined in config file.
    
    Args:
        config_path: Path to config.yaml
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all results
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config['global']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Run each experiment
    experiments = config['experiments']
    
    for exp_name, exp_config in experiments.items():
        if not exp_config.get('enabled', True):
            logging.info(f"Skipping disabled experiment: {exp_name}")
            continue
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Running experiment: {exp_name}")
        logging.info(f"Description: {exp_config['description']}")
        logging.info(f"{'='*80}\n")
        
        # Dispatch to specific experiment handler
        if exp_name == 'optimizer_comparison':
            results = run_optimizer_comparison(exp_config, config)
        elif exp_name == 'scheduler_comparison':
            results = run_scheduler_comparison(exp_config, config)
        elif exp_name == 'rl_seeding':
            results = run_rl_seeding(exp_config, config)
        else:
            logging.warning(f"Unknown experiment type: {exp_name}")
            continue
        
        all_results.extend(results)
        
        # Save intermediate results
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(output_path / f'{exp_name}_results.csv', index=False)
        logging.info(f"Saved results to: {output_path / f'{exp_name}_results.csv'}")
    
    # Combine all results
    df_all = pd.DataFrame([r.to_dict() for r in all_results])
    df_all.to_csv(output_path / 'all_results.csv', index=False)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"All experiments complete!")
    logging.info(f"Results saved to: {output_path / 'all_results.csv'}")
    logging.info(f"{'='*80}\n")
    
    return df_all


def run_optimizer_comparison(exp_config: Dict, global_config: Dict) -> List[ExperimentResult]:
    """Run optimizer comparison experiment."""
    # TODO: Implement
    logging.info("Optimizer comparison experiment not yet implemented")
    return []


def run_scheduler_comparison(exp_config: Dict, global_config: Dict) -> List[ExperimentResult]:
    """Run scheduler comparison experiment."""
    # TODO: Implement
    logging.info("Scheduler comparison experiment not yet implemented")
    return []


def run_rl_seeding(exp_config: Dict, global_config: Dict) -> List[ExperimentResult]:
    """Run RL-seeding experiment: test RL warm-start followed by gradient descent."""
    logging.info("Running RL-seeding experiment...")
    results = []
    
    # Load test image and PSF
    image_config = next(i for i in global_config['data']['images'] if i['name'] == exp_config['image'])
    psf_config = next(p for p in global_config['data']['psfs'] if p['name'] == exp_config['psf'])
    noise_config = next(n for n in global_config['data']['noise'] if n['name'] == exp_config['noise'])
    
    ground_truth = load_test_image(image_config)
    psf = generate_psf(psf_config)
    observed = generate_observation(ground_truth, psf, noise_config)
    
    device = global_config['global']['device']
    max_iterations = exp_config.get('max_iterations', 500)
    
    # Sweep over RL warmup iterations
    rl_warmup_list = exp_config.get('rl_warmup_iterations', [0, 10, 50, 100])
    post_rl_configs = exp_config.get('post_rl_optimizers', [
        {'optimizer': 'adam', 'scheduler': 'cosine', 'lr': 1e-3}
    ])
    
    total_runs = len(rl_warmup_list) * len(post_rl_configs)
    logging.info(f"Total runs: {total_runs} ({len(rl_warmup_list)} warmups × {len(post_rl_configs)} optimizers)")
    
    for rl_warmup in tqdm(rl_warmup_list, desc="RL warmup sweep"):
        for opt_config in post_rl_configs:
            try:
                result = train_deconvolution(
                    observed=observed,
                    psf=psf,
                    ground_truth=ground_truth,
                    optimizer_name=opt_config['optimizer'],
                    scheduler_name=opt_config['scheduler'],
                    lr=opt_config['lr'],
                    max_iterations=max_iterations,
                    device=device,
                    rl_warmup_iterations=rl_warmup,
                    use_rl_derived_lr=exp_config.get('use_rl_derived_lr', False),
                    experiment_name=f"rl_seeding_warmup{rl_warmup}_{opt_config['optimizer']}",
                    rl_warmup=rl_warmup,
                )
                results.append(result)
                logging.info(f"RL warmup={rl_warmup}, {opt_config['optimizer']}: PSNR={result.final_psnr:.2f}")
            except Exception as e:
                logging.error(f"Failed: rl_warmup={rl_warmup}, {opt_config['optimizer']}: {e}")
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimizer/scheduler experiments')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Run specific experiment (default: all)')
    
    args = parser.parse_args()
    
    # Run experiments
    df_results = run_experiment_suite(args.config, args.output)
    
    # Print summary
    print("\nTop 10 configurations by PSNR:")
    print(df_results.nlargest(10, 'final_psnr')[
        ['experiment_name', 'final_psnr', 'final_ssim', 'total_time']
    ])
