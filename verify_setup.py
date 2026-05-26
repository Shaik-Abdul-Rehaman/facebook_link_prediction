#!/usr/bin/env python3
"""
Project setup verification script
Checks if all files are in place and environment is ready
"""

import os
import sys
from pathlib import Path

def check_file(path, file_type="file"):
    """Check if a file or directory exists"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {file_type}: {path}")
    return exists

def main():
    print("=" * 60)
    print("LINK PREDICTION PROJECT - SETUP VERIFICATION")
    print("=" * 60)

    base_dir = Path(__file__).parent
    all_good = True

    # Check Python files
    print("\n[1] Python Source Files")
    print("-" * 60)
    python_files = [
        'config.py',
        'train.py',
        'run_api.py',
        'test_api.py',
        'verify_setup.py',
        'src/__init__.py',
        'src/api.py',
        'src/data_processor.py',
        'src/feature_engineer.py',
        'src/model_trainer.py'
    ]
    for file in python_files:
        if not check_file(base_dir / file):
            all_good = False

    # Check Documentation
    print("\n[2] Documentation Files")
    print("-" * 60)
    docs = [
        'README.md',
        'GETTING_STARTED.md',
        'QUICKREF.md',
        'PROJECT_SUMMARY.md',
        'INDEX.md'
    ]
    for file in docs:
        if not check_file(base_dir / file):
            all_good = False

    # Check Configuration Files
    print("\n[3] Configuration Files")
    print("-" * 60)
    config_files = [
        'requirements.txt',
        '.env.example',
        'Dockerfile',
        'docker-compose.yml',
        '.gitignore'
    ]
    for file in config_files:
        if not check_file(base_dir / file):
            all_good = False

    # Check Input Data
    print("\n[4] Input Data")
    print("-" * 60)
    if check_file(base_dir / 'train.csv'):
        size_mb = os.path.getsize(base_dir / 'train.csv') / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print("⚠ WARNING: train.csv not found")
        print("  → Required for training. Place in project root.")
        all_good = False

    # Check Dependencies
    print("\n[5] Python Dependencies")
    print("-" * 60)
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'sklearn',
        'networkx',
        'scipy',
        'xgboost'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package if package != 'sklearn' else 'sklearn')
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
            all_good = False

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("  → Run: pip install -r requirements.txt")

    # Check Directories (will be created)
    print("\n[6] Required Directories")
    print("-" * 60)
    directories = ['src', 'data', 'models', 'features']
    for dir_name in directories:
        dir_path = base_dir / dir_name
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}/")
        else:
            print(f"⚠ {dir_name}/ - Will be created on first run")

    # Summary
    print("\n" + "=" * 60)
    if all_good and not missing:
        print("STATUS: ✓ ALL CHECKS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run training:    python train.py")
        print("2. Start API:       python run_api.py")
        print("3. Test API:        python test_api.py")
        return 0
    else:
        print("STATUS: ✗ ISSUES FOUND")
        print("=" * 60)
        if missing:
            print("\nFix missing packages:")
            print(f"  pip install {' '.join(missing)}")
        else:
            print("\nMissing files detected. Check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
