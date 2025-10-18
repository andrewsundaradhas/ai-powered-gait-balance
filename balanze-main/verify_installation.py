"""
Verification script to check if all components are working correctly.
Run this script after installation to verify everything is set up properly.
"""
import sys
import os
import importlib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'scikit-learn',
    'opencv-python',
    'mediapipe',
    'fastapi',
    'uvicorn',
    'python-multipart',
    'python-dotenv',
    'joblib',
    'matplotlib',
    'seaborn',
    'plotly',
    'pydantic',
    'python-jose[cryptography]',
    'passlib[bcrypt]',
    'python-multipart',
    'aiofiles',
    'pytest',
    'pytest-cov'
]

def check_python_version():
    """Check Python version."""
    logger.info(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    return True

def check_imports():
    """Check if all required packages are installed."""
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package.split('[')[0] if '[' in package else package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is not installed")
    
    if missing_packages:
        logger.error("\nMissing packages. Install them using:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        "app",
        "src",
        "src/api",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "data/raw",
        "models",
        "reports/figures"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            all_exist = False
        else:
            logger.info(f"✓ Directory exists: {dir_path}")
    
    if not all_exist:
        logger.warning("\nSome directories are missing. Consider creating them.")
    
    return all_exist

def check_models():
    """Check if required models exist."""
    required_models = [
        "models/gait_classifier.pkl",
        "models/balance_classifier.pkl",
        "models/disorder_classifier.pkl"
    ]
    
    all_exist = True
    for model_path in required_models:
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            all_exist = False
        else:
            logger.info(f"✓ Model exists: {model_path}")
    
    if not all_exist:
        logger.warning("\nSome models are missing. Train them using 'python scripts/train_models.py'")
    
    return all_exist

def main():
    """Run all checks."""
    logger.info("="*50)
    logger.info("Starting verification...")
    logger.info("="*50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Package Imports", check_imports()),
        ("Directory Structure", check_directories()),
        ("Model Files", check_models())
    ]
    
    logger.info("\n" + "="*50)
    logger.info("Verification Summary:")
    logger.info("="*50)
    
    all_passed = True
    for check_name, passed in checks:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{check_name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n✅ All checks passed! The system is ready for deployment.")
    else:
        logger.warning("\n❌ Some checks failed. Please address the issues above before deployment.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
