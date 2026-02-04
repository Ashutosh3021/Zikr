# TTS-STT System Implementation
# Following the technical specification framework

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Version information
__version__ = "1.0.0"
__author__ = "TTS-STT Development Team"

# Project structure constants
PROJECT_ROOT = project_root
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

# Create directory structure
for directory in [CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, TESTS_DIR, SRC_DIR]:
    directory.mkdir(exist_ok=True)

# Import core modules
try:
    from src.utils.logger import setup_logger
    from src.utils.config import load_config
    
    # Initialize logging
    logger = setup_logger(__name__)
    logger.info(f"TTS-STT System v{__version__} initialized")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Please ensure all dependencies are installed")

# System configuration
SYSTEM_CONFIG = {
    "cuda_available": False,
    "gpu_count": 0,
    "default_device": "cpu",
    "max_workers": 4
}

# Check for CUDA availability
try:
    import torch
    if torch.cuda.is_available():
        SYSTEM_CONFIG["cuda_available"] = True
        SYSTEM_CONFIG["gpu_count"] = torch.cuda.device_count()
        SYSTEM_CONFIG["default_device"] = "cuda"
        logger.info(f"CUDA available: {SYSTEM_CONFIG['gpu_count']} GPU(s) detected")
    else:
        logger.info("CUDA not available, using CPU")
except ImportError:
    logger.warning("PyTorch not available, cannot check CUDA status")

# Environment validation
REQUIRED_ENV_VARS = [
    "PYTHONPATH",
    # Add other required environment variables here
]

def validate_environment():
    """Validate system environment meets minimum requirements"""
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if var not in os.environ:
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    logger.info("Environment validation passed")
    return True

# Initialize system
if __name__ == "__main__":
    validate_environment()
    print(f"TTS-STT System v{__version__}")
    print(f"Root directory: {PROJECT_ROOT}")
    print(f"System config: {SYSTEM_CONFIG}")