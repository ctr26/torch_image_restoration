"""Sanity tests to catch silent failures early."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_image_range():
    """Images must be in [0, 1] range."""
    from experiments.runner import load_test_image
    
    for name in ['camera', 'astronaut']:
        config = {'name': name, 'colorspace': 'gray', 'downsample_factor': 4}
        img = load_test_image(config)
        assert img.min() >= 0, f"{name}: min={img.min()}"
        assert img.max() <= 1, f"{name}: max={img.max()}"
        assert img.dtype == np.float64, f"{name}: dtype={img.dtype}"
    print("✓ test_image_range passed")


def test_psf_normalized():
    """PSF must sum to 1."""
    from experiments.runner import generate_psf
    
    for psf_type in ['gaussian', 'defocus']:
        if psf_type == 'gaussian':
            config = {'type': 'gaussian', 'size': (21, 21), 'sigma': 2, 'gain': 1000}
        else:
            config = {'type': 'defocus', 'size': (21, 21), 'radius': 5, 'gain': 1000}
        
        psf = generate_psf(config)
        assert abs(psf.sum() - 1.0) < 1e-6, f"{psf_type}: sum={psf.sum()}"
    print("✓ test_psf_normalized passed")


def test_observation_range():
    """Observation must stay in reasonable range after convolution + noise."""
    from experiments.runner import load_test_image, generate_psf, generate_observation
    
    img = load_test_image({'name': 'camera', 'colorspace': 'gray', 'downsample_factor': 4})
    psf = generate_psf({'type': 'gaussian', 'size': (21, 21), 'sigma': 2, 'gain': 1000})
    obs = generate_observation(img, psf, {'type': 'poisson', 'peak_photons': 1000})
    
    # Should be roughly [0, 1] - allow some overshoot from noise
    assert obs.min() >= -0.1, f"obs min={obs.min()}"
    assert obs.max() <= 1.5, f"obs max={obs.max()}"
    print("✓ test_observation_range passed")


def test_psnr_positive():
    """PSNR should be positive for reasonable reconstructions."""
    from skimage.metrics import peak_signal_noise_ratio
    
    gt = np.random.rand(64, 64)
    # Simulated "reconstruction" - similar but not identical
    recon = gt + np.random.randn(64, 64) * 0.1
    recon = np.clip(recon, 0, 1)  # Must clamp!
    
    psnr = peak_signal_noise_ratio(gt, recon, data_range=1.0)
    assert psnr > 0, f"PSNR should be positive, got {psnr}"
    print(f"✓ test_psnr_positive passed (PSNR={psnr:.1f}dB)")


if __name__ == '__main__':
    test_image_range()
    test_psf_normalized()
    test_observation_range()
    test_psnr_positive()
    print("\n✅ All sanity tests passed!")
