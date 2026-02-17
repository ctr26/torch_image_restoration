"""
Image Restoration Metrics
==========================

Comprehensive metrics for evaluating deconvolution quality.
"""

import numpy as np
from typing import Dict, Any
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)


def compute_all_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute all quality metrics for restored image.
    
    Args:
        estimate: Restored image
        ground_truth: Original clean image
    
    Returns:
        Dictionary of metric values
    """
    data_range = ground_truth.max() - ground_truth.min()
    
    metrics = {
        'psnr': peak_signal_noise_ratio(ground_truth, estimate, data_range=data_range),
        'ssim': structural_similarity(ground_truth, estimate, data_range=data_range),
        'mse': mean_squared_error(ground_truth, estimate),
        'mae': np.mean(np.abs(ground_truth - estimate)),
        'rmse': np.sqrt(mean_squared_error(ground_truth, estimate)),
    }
    
    return metrics


def compute_convergence_speed(psnr_history: list, threshold: float = 0.95) -> int:
    """
    Compute number of iterations to reach threshold % of final PSNR.
    
    Args:
        psnr_history: List of PSNR values over iterations
        threshold: Fraction of final PSNR to reach (e.g., 0.95 = 95%)
    
    Returns:
        Iteration index where threshold is reached
    """
    if len(psnr_history) == 0:
        return -1
    
    final_psnr = psnr_history[-1]
    target_psnr = threshold * final_psnr
    
    for i, psnr in enumerate(psnr_history):
        if psnr >= target_psnr:
            return i
    
    return len(psnr_history)  # Never reached threshold


def compute_efficiency_metric(psnr_gain: float, time: float) -> float:
    """
    Compute efficiency: PSNR gain per second.
    
    Args:
        psnr_gain: Improvement in PSNR from initial to final
        time: Total time in seconds
    
    Returns:
        PSNR gain per second
    """
    if time <= 0:
        return 0.0
    return psnr_gain / time
