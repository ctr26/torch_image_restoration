# Fast-Fail Experiment Notes (2026-02-17)

## What Works
- ✅ FFT-based convolution for exact 'same' output
- ✅ RL-seeding experiment runs (21 configs: 7 warmups × 3 optimizers)
- ✅ Adam/AdamW converge with RL-derived learning rate
- ✅ Early stopping triggers correctly

## Issues Found

### 1. Data Scaling
**Problem**: PSNR is negative (-50 to -52 dB)
**Cause**: Data not normalized to [0,1] range; loss values in billions
**Fix**: Normalize ground truth and observed images before training

### 2. SGD Explodes
**Problem**: SGD with momentum goes to inf→NaN in ~25 iterations
**Cause**: RL-derived LR (~0.16) too high for vanilla SGD
**Finding**: RL-derived LR works for adaptive optimizers (Adam, AdamW) but needs reduction for SGD (divide by 10-100)

### 3. Conv Shape (Fixed)
**Problem**: `conv2d` output 129x129 for 128x128 input
**Fix**: Use FFT-based convolution with proper padding/cropping

## Next Steps
1. [x] Add data normalization (scale to [0,1]) - FIXED: img_as_float in load_test_image
2. [x] Fix Poisson noise for [0,1] images - FIXED: scale by peak_photons
3. [ ] Add optimizer-specific LR scaling (SGD: LR/100)
4. [ ] Run full experiment suite
5. [ ] Generate figures

## Key Insight
RL-derived LR is a good starting point for adaptive optimizers but not universal.
This is a paper finding: "RL-LR optimal for Adam-family, requires scaling for SGD-family"
