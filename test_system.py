#!/usr/bin/env python3
"""
Test script for the English-French Neural Machine Translation System.

This script tests the basic functionality of the system components.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        logger.error(f"❌ Failed to import PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        logger.error(f"❌ Failed to import NumPy: {e}")
        return False
    
    try:
        import sklearn
        logger.info(f"✅ Scikit-learn imported successfully (version: {sklearn.__version__})")
    except ImportError as e:
        logger.error(f"❌ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        from rouge_score import rouge_scorer
        logger.info("✅ Rouge-score imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Rouge-score: {e}")
        return False
    
    try:
        import sacrebleu
        logger.info("✅ Sacrebleu imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Sacrebleu: {e}")
        return False
    
    return True


def test_project_structure():
    """Test that the project structure is correct."""
    logger.info("Testing project structure...")
    
    project_root = Path(__file__).parent
    required_files = [
        "translator.py",
        "config.py",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "data",
        "saved_models"
    ]
    
    # Check required files
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            logger.info(f"✅ Found {file_name}")
        else:
            logger.error(f"❌ Missing {file_name}")
            return False
    
    # Check required directories
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            logger.info(f"✅ Found directory {dir_name}")
        else:
            logger.warning(f"⚠️  Directory {dir_name} not found (will be created)")
    
    return True


def test_data_files():
    """Test that data files exist and are accessible."""
    logger.info("Testing data files...")
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        logger.warning("⚠️  Data directory not found")
        return True
    
    # Look for CSV files
    csv_files = list(data_dir.rglob("*.csv"))
    if csv_files:
        for csv_file in csv_files:
            file_size = csv_file.stat().st_size
            logger.info(f"✅ Found data file: {csv_file.name} ({file_size / (1024*1024):.1f} MB)")
    else:
        logger.warning("⚠️  No CSV data files found in data directory")
    
    return True


def test_model_files():
    """Test that model files exist and are accessible."""
    logger.info("Testing model files...")
    
    project_root = Path(__file__).parent
    models_dir = project_root / "saved_models"
    
    if not models_dir.exists():
        logger.warning("⚠️  Saved models directory not found")
        return True
    
    # Look for model files
    model_files = list(models_dir.rglob("*.pkl")) + list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth"))
    if model_files:
        for model_file in model_files:
            file_size = model_file.stat().st_size
            logger.info(f"✅ Found model file: {model_file.name} ({file_size / (1024*1024):.1f} MB)")
    else:
        logger.warning("⚠️  No model files found in saved_models directory")
    
    return True


def test_basic_functionality():
    """Test basic functionality of the system components."""
    logger.info("Testing basic functionality...")
    
    try:
        # Test configuration
        from config import get_device, get_model_path, get_dictionary_paths
        logger.info("✅ Configuration module imported successfully")
        
        device = get_device()
        logger.info(f"✅ Device detection working: {device}")
        
        model_path = get_model_path()
        logger.info(f"✅ Model path generation working: {model_path}")
        
        dict_paths = get_dictionary_paths()
        logger.info(f"✅ Dictionary paths generation working: {dict_paths}")
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False
    
    try:
        # Test basic functions from translator
        from translator import summarize_document, calculate_rouge_scores
        
        # Test summarization
        test_doc = "This is a test document with multiple sentences. It should be summarized properly."
        summary = summarize_document(test_doc, max_chars=50)
        logger.info(f"✅ Summarization working: '{summary}'")
        
        # Test ROUGE calculation
        refs = ["This is a test document."]
        hyps = ["This is a test document."]
        scores = calculate_rouge_scores(hyps, refs)
        logger.info(f"✅ ROUGE calculation working: {scores}")
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False
    
    return True


def test_gpu_availability():
    """Test GPU availability if PyTorch is available."""
    logger.info("Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            
            # Test basic GPU operations
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                logger.info("✅ Basic GPU operations working")
            except Exception as e:
                logger.warning(f"⚠️  GPU operations failed: {e}")
        else:
            logger.info("ℹ️  No GPU available, will use CPU")
            
    except ImportError:
        logger.warning("⚠️  PyTorch not available, skipping GPU test")
    
    return True


def main():
    """Run all tests."""
    logger.info("🧪 Running system tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Project Structure Test", test_project_structure),
        ("Data Files Test", test_data_files),
        ("Model Files Test", test_model_files),
        ("Basic Functionality Test", test_basic_functionality),
        ("GPU Availability Test", test_gpu_availability),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
