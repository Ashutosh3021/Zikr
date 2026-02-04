"""
Logging utilities for the TTS-STT system.
Implements structured logging with different levels and output formats.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and configuration.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        console_output: Whether to output to console (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_system_logger() -> logging.Logger:
    """Get the main system logger."""
    return setup_logger("tts_stt.system")

def get_model_logger() -> logging.Logger:
    """Get logger for model components."""
    return setup_logger("tts_stt.model")

def get_data_logger() -> logging.Logger:
    """Get logger for data processing components."""
    return setup_logger("tts_stt.data")

def get_api_logger() -> logging.Logger:
    """Get logger for API components."""
    return setup_logger("tts_stt.api")

# Pre-configured loggers
system_logger = get_system_logger()
model_logger = get_model_logger()
data_logger = get_data_logger()
api_logger = get_api_logger()

class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    def __init__(self, logger_name: Optional[str] = None):
        if logger_name:
            self.logger = setup_logger(logger_name)
        else:
            self.logger = get_system_logger()
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)