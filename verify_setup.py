#!/usr/bin/env python3
"""
Verify experiment infrastructure setup.

This script checks:
1. All required dependencies are installed
2. Experiment modules can be imported
3. Results directory structure exists
4. Basic module functionality works
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 80)
    print("Checking dependencies...")
    print("=" * 80)
    
    required_packages = [
        'torch',
        'numpy',
        'scipy',
        'skimage',  # scikit-image
        'yaml',     # pyyaml
        'pandas',
        'matplotlib',
        'tqdm',
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} - MISSING")
    
    print(f"\nInstalled: {len(installed)}/{len(required_packages)}")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print("pip install torch numpy scipy scikit-image pyyaml pandas matplotlib tqdm")
        return False
    
    return True


def check_modules():
    """Check if experiment modules can be imported."""
    print("\n" + "=" * 80)
    print("Checking experiment modules...")
    print("=" * 80)
    
    modules = [
        'experiments',
        'experiments.optimizers',
        'experiments.schedulers',
        'experiments.runner',
        'experiments.metrics',
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            failed.append((module, str(e)))
            print(f"✗ {module} - FAILED: {e}")
    
    if failed:
        print(f"\nFailed to import {len(failed)} modules")
        return False
    
    print(f"\nAll {len(modules)} modules imported successfully")
    return True


def check_directory_structure():
    """Check if required directories exist."""
    print("\n" + "=" * 80)
    print("Checking directory structure...")
    print("=" * 80)
    
    required_dirs = [
        'experiments',
        'results',
        'tests',
        'scripts',
    ]
    
    missing = []
    
    for dirname in required_dirs:
        path = Path(dirname)
        if path.exists():
            print(f"✓ {dirname}/")
        else:
            missing.append(dirname)
            print(f"✗ {dirname}/ - MISSING")
    
    if missing:
        print(f"\nCreating missing directories: {', '.join(missing)}")
        for dirname in missing:
            Path(dirname).mkdir(parents=True, exist_ok=True)
            print(f"  Created {dirname}/")
    
    return True


def test_basic_functionality():
    """Test basic module functionality."""
    print("\n" + "=" * 80)
    print("Testing basic functionality...")
    print("=" * 80)
    
    try:
        # Test optimizer listing
        from experiments.optimizers import OPTIMIZER_CONFIGS
        print(f"✓ Found {len(OPTIMIZER_CONFIGS)} optimizer configurations")
        
        # Test scheduler listing
        from experiments.schedulers import SCHEDULER_CONFIGS
        print(f"✓ Found {len(SCHEDULER_CONFIGS)} scheduler configurations")
        
        # Test that we can create an optimizer
        import torch
        param = torch.tensor([1.0], requires_grad=True)
        from experiments.optimizers import get_optimizer
        opt = get_optimizer('adam', [param])
        print(f"✓ Successfully created optimizer: {type(opt).__name__}")
        
        # Test that we can create a scheduler
        from experiments.schedulers import get_scheduler
        sched = get_scheduler('cosine', opt, T_max=100)
        print(f"✓ Successfully created scheduler: {type(sched).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("EXPERIMENT INFRASTRUCTURE VERIFICATION")
    print("=" * 80)
    
    results = {
        'Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
    }
    
    # Only check modules and functionality if dependencies are installed
    if results['Dependencies']:
        results['Module Imports'] = check_modules()
        if results['Module Imports']:
            results['Basic Functionality'] = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All checks passed! Infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Run experiments: python -m experiments.optimizers")
        print("  2. Run schedulers: python -m experiments.schedulers")
        print("  3. Run full suite: python -m experiments.runner --config experiments/config.yaml")
        return 0
    else:
        print("\n✗ Some checks failed. Please resolve the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
