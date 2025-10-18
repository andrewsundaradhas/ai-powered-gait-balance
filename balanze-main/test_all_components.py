"""
Comprehensive test script to verify all components of the Gait Analysis System.
Run this script to ensure everything is working as expected.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_results.log")
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Constants
TEST_DATA_DIR = Path("tests/test_data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def test_imports():
    """Test that all required modules can be imported."""
    modules = [
        "numpy",
        "pandas",
        "sklearn",
        "cv2",
        "mediapipe",
        "fastapi",
        "pydantic",
        "joblib",
        "matplotlib",
        "seaborn",
        "plotly",
        "src.features.advanced_feature_engineer",
        "src.models.model_manager",
        "src.visualization.plot_utils",
        "app.main"
    ]
    
    for module in modules:
        try:
            __import__(module)
            logger.info(f"✓ Successfully imported: {module}")
        except ImportError as e:
            logger.error(f"✗ Failed to import {module}: {e}")
            raise

def test_data_validation():
    """Test data validation functionality."""
    try:
        from src.data.data_validator import DataValidator, GaitDataSchema, DataVersion
        
        # Test DataVersion
        version = DataVersion(major=1, minor=0, patch=0)
        assert str(version) == "1.0.0"
        
        # Test GaitDataSchema
        sample_data = {
            "left_hip_angle": 25.5,
            "right_hip_angle": 24.8,
            "left_knee_angle": 15.2,
            "right_knee_angle": 16.0,
            "cadence": 110.5,
            "stride_length": 1.2
        }
        validated = GaitDataSchema(**sample_data)
        assert validated.left_hip_angle == 25.5
        
        logger.info("✓ Data validation tests passed")
        return True
    except ImportError as e:
        logger.error("Data validation module not found: %s", e)
        pytest.skip("Data validation module not available")
    except Exception as e:
        logger.error("Data validation tests failed: %s", e)
        raise

def test_model_loading():
    """Test that all required models can be loaded."""
    try:
        import joblib
        from src.models.model_manager import ModelManager, ModelType
        
        models = {
            "gait_classifier.pkl": ModelType.GAIT,
            "balance_classifier.pkl": ModelType.BALANCE,
            "disorder_classifier.pkl": ModelType.DISORDER
        }
        
        manager = ModelManager()
        all_models_loaded = True
        
        for model_file, model_type in models.items():
            model_path = MODELS_DIR / model_file
            if not model_path.exists():
                logger.warning("Model file not found: %s", model_path)
                all_models_loaded = False
                continue
                
            model = manager.load_model(model_path, model_type)
            assert model is not None
            logger.info("✓ Successfully loaded model: %s", model_file)
        
        if all_models_loaded:
            logger.info("✓ All models loaded successfully")
        else:
            logger.warning("⚠ Some models were not found")
        
        return all_models_loaded
    except ImportError as e:
        logger.error("Required modules not found: %s", e)
        pytest.skip("Required modules not available")
    except Exception as e:
        logger.error("Model loading tests failed: %s", e)
        raise

def test_visualization():
    """Test visualization functions."""
    try:
        from src.visualization.plot_utils import GaitVisualizer
        
        # Create test data
        joint_angles = {
            "left_hip": np.sin(np.linspace(0, 2 * np.pi, 100)) * 30 + 20,
            "right_hip": np.sin(np.linspace(0, 2 * np.pi, 100) + np.pi) * 30 + 20,
            "left_knee": np.sin(np.linspace(0, 2 * np.pi, 100) + 0.5) * 15 + 10,
            "right_knee": np.sin(np.linspace(0, 2 * np.pi, 100) + 0.5 + np.pi) * 15 + 10,
        }
        
        # Create output directory if it doesn't exist
        output_dir = REPORTS_DIR / "test_figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = GaitVisualizer(output_dir=str(output_dir))
        
        # Test plotting functions
        fig = visualizer.plot_gait_cycle(joint_angles, sample_rate=30.0)
        assert fig is not None
        
        # Save test figure
        output_path = output_dir / "test_gait_cycle.html"
        fig.write_html(str(output_path))
        assert output_path.exists()
        
        logger.info("✓ Visualization tests passed")
        return True
    except ImportError as e:
        logger.error("Visualization modules not found: %s", e)
        pytest.skip("Visualization modules not available")
    except Exception as e:
        logger.error("Visualization tests failed: %s", e)
        raise

def test_api_endpoints():
    """Test FastAPI endpoints."""
    try:
        from app.main import app
        
        client = TestClient(app)
        
        # Test health check endpoint
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
        
        # Test authentication
        response = client.post(
            "/api/token",
            data={"username": "admin", "password": "admin"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 200
        token = response.json().get("access_token")
        assert token is not None
        
        # Test protected endpoint
        response = client.get(
            "/api/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        logger.info("✓ API endpoint tests passed")
        return True
    except ImportError as e:
        logger.error("FastAPI modules not found: %s", e)
        pytest.skip("FastAPI modules not available")
    except Exception as e:
        logger.error("API endpoint tests failed: %s", e)
        raise

def run_all_tests():
    """Run all test functions and return results."""
    test_functions = [
        ("Import Tests", test_imports),
        ("Data Validation Tests", test_data_validation),
        ("Model Loading Tests", test_model_loading),
        ("Visualization Tests", test_visualization),
        ("API Endpoint Tests", test_api_endpoints)
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in test_functions:
        try:
            logger.info("\n%s", "=" * 50)
            logger.info("Running %s...", name)
            logger.info("%s", "=" * 50)
            
            result = test_func()
            status = "PASSED" if result else "FAILED"
            results[name] = status
            
            if not result:
                all_passed = False
                
            logger.info("\n%s: %s", name, status)
            
        except Exception as e:
            logger.error("Error in %s: %s", name, str(e), exc_info=True)
            results[name] = "ERROR"
            all_passed = False
    
    # Print summary
    logger.info("\n%s", "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("%s", "=" * 50)
    
    for test, status in results.items():
        logger.info("%s: %s", test, status)
    
    if all_passed:
        logger.info("\n✅ All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed. Please check the logs for details.")
    
    return all_passed

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(REPORTS_DIR / "test_figures", exist_ok=True)
    
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
