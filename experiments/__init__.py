"""
Torch Image Restoration Experiments
====================================

This package contains experimental frameworks for testing different optimization
strategies for image deconvolution tasks.
"""

from .optimizers import OPTIMIZER_CONFIGS
from .schedulers import SCHEDULER_CONFIGS

__all__ = ['OPTIMIZER_CONFIGS', 'SCHEDULER_CONFIGS']
