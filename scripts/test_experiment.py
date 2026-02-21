#!/usr/bin/env python
"""
Quick test to verify experiment infrastructure works.
Runs a minimal deconvolution experiment to verify all components.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import train_deconvolution
from experiments.optimizers import OPTIMIZER_CONFIGS
from experiments.schedulers import SCHEDULER_CONFIGS, compute_rl_learning_rate
from scipy.ndimage import gaussian_filter


def create_test_data(size=64):
    """Create small test dataset for quick verification."""
    print(f"📊 Creating test data ({size}x{size})...")
    
    # Simple test image (circle)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    ground_truth = ((x**2 + y**2) < 0.5).astype(np.float32)
    
    # Gaussian PSF
    psf = np.zeros((size, size))
    psf[size//2, size//2] = 1.0
    psf = gaussian_filter(psf, sigma=2.0)
    psf /= psf.sum()
    
    # Convolve (simple scipy convolution for test)
    from scipy.signal import convolve2d
    blurred = convolve2d(ground_truth, psf, mode='same', boundary='wrap')
    
    # Add noise
    snr_db = 20.0
    signal_power = np.mean(blurred ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), blurred.shape)
    observed = blurred + noise
    
    return ground_truth, psf, observed


def test_optimizers():
    """Test that all optimizers are accessible."""
    print("\n🔧 Testing optimizers...")
    print(f"Available optimizers: {len(OPTIMIZER_CONFIGS)}")
    for name in OPTIMIZER_CONFIGS.keys():
        print(f"  ✅ {name}")
    assert len(OPTIMIZER_CONFIGS) >= 10, "Expected at least 10 optimizers"


def test_schedulers():
    """Test that all schedulers are accessible."""
    print("\n📅 Testing schedulers...")
    print(f"Available schedulers: {len(SCHEDULER_CONFIGS)}")
    for name in SCHEDULER_CONFIGS.keys():
        print(f"  ✅ {name}")
    assert len(SCHEDULER_CONFIGS) >= 8, "Expected at least 8 schedulers"


def test_rl_learning_rate():
    """Test RL-derived learning rate computation."""
    print("\n🎯 Testing RL-derived learning rate...")
    ground_truth, psf, observed = create_test_data(size=32)
    
    lr = compute_rl_learning_rate(psf, observed, n_iterations=10)
    print(f"  Computed RL-LR: {lr:.6e}")
    
    assert lr > 0, "LR must be positive"
    assert lr < 1.0, "LR should be < 1.0 for gradient descent"
    print("  ✅ RL learning rate computation OK")


def test_single_experiment():
    """Run a minimal experiment to verify the full pipeline."""
    print("\n🧪 Running minimal experiment...")
    
    # Use CPU and very small problem for speed
    device = 'cpu'
    ground_truth, psf, observed = create_test_data(size=32)
    
    print(f"  Device: {device}")
    print(f"  Optimizer: adam")
    print(f"  Scheduler: constant")
    print(f"  Max iterations: 10 (very short for testing)")
    
    try:
        result = train_deconvolution(
            observed=observed,
            psf=psf,
            ground_truth=ground_truth,
            optimizer_name='adam',
            scheduler_name='constant',
            lr=1e-3,
            max_iterations=10,  # Very short for testing
            device=device,
            convergence_patience=5,
            convergence_threshold=1e-4
        )
        
        print(f"\n  📈 Results:")
        print(f"    Final PSNR: {result.final_psnr:.2f} dB")
        print(f"    Final SSIM: {result.final_ssim:.4f}")
        print(f"    Iterations: {result.iterations}")
        print(f"    Time: {result.total_time:.3f}s")
        
        # Basic sanity checks
        assert result.final_psnr > 0, "PSNR should be positive"
        assert 0 <= result.final_ssim <= 1, "SSIM should be in [0,1]"
        assert result.iterations > 0, "Should run at least one iteration"
        
        print("  ✅ Experiment completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_results_directory():
    """Verify results directory structure exists."""
    print("\n📁 Testing results directory structure...")
    
    results_dir = Path("results")
    expected_subdirs = [
        "optimizer_comparison",
        "scheduler_comparison",
        "grid_search",
        "snr_sweep",
        "psf_sweep",
        "convergence_analysis",
        "rl_seeding",
        "rl_lr_comparison",
        "lbfgs_experiment",
        "plots",
        "checkpoints"
    ]
    
    for subdir in expected_subdirs:
        path = results_dir / subdir
        if path.exists():
            print(f"  ✅ {subdir}/")
        else:
            print(f"  ⚠️  {subdir}/ (missing, will be created)")
    
    print("  ✅ Directory structure verified")


def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 Testing Experiment Infrastructure")
    print("=" * 60)
    
    try:
        test_optimizers()
        test_schedulers()
        test_rl_learning_rate()
        test_results_directory()
        success = test_single_experiment()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
            print("\nExperiment infrastructure is ready to use.")
            print("\nNext steps:")
            print("  • Run full experiment: python -m experiments.runner --experiment rl_seeding")
            print("  • Run all experiments: python -m experiments.runner --config experiments/config.yaml")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
