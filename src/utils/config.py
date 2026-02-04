"""
Configuration management for the TTS-STT system.
Handles loading, validation, and management of system configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict

from .logger import get_system_logger

logger = get_system_logger()

@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    # TTS Model Configuration
    tts_encoder_layers: int = 12
    tts_encoder_heads: int = 8
    tts_encoder_dim: int = 512
    tts_vocab_size: int = 50000
    tts_max_length: int = 1024
    
    # STT Model Configuration
    stt_encoder_layers: int = 12
    stt_encoder_heads: int = 8
    stt_encoder_dim: int = 512
    stt_decoder_layers: int = 6
    stt_decoder_heads: int = 8
    stt_vocab_size: int = 10000
    
    # Common Configuration
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Hardware
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Scheduling
    scheduler_type: str = "cosine"
    scheduler_warmup_ratio: float = 0.1
    
    # Regularization
    gradient_clipping: float = 1.0
    label_smoothing: float = 0.1

@dataclass
class DataConfig:
    """Configuration for data processing."""
    # Audio parameters
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    
    # Text parameters
    max_text_length: int = 200
    max_audio_length: int = 8192
    
    # Preprocessing
    normalize_audio: bool = True
    trim_silence: bool = True
    silence_threshold: float = 0.01

@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    device: str = "cuda"  # Will be updated based on availability
    batch_size: int = 1
    max_input_length: int = 200
    enable_caching: bool = True
    cache_size: int = 1000
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Initialize device based on CUDA availability."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

@dataclass
class SystemConfig:
    """System-level configuration."""
    # Paths
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    config_dir: str = "./config"
    
    # Device settings
    device: str = "cuda"
    seed: int = 42
    
    # Performance
    cache_size: int = 1000
    prefetch_factor: int = 2

class ConfigManager:
    """Manages system configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        self.system_config = SystemConfig()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_file.suffix}")
            
            self._update_config_from_dict(config_data)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration objects from dictionary."""
        if 'model' in config_dict:
            self._update_dataclass_from_dict(self.model_config, config_dict['model'])
        
        if 'training' in config_dict:
            self._update_dataclass_from_dict(self.training_config, config_dict['training'])
        
        if 'data' in config_dict:
            self._update_dataclass_from_dict(self.data_config, config_dict['data'])
        
        if 'system' in config_dict:
            self._update_dataclass_from_dict(self.system_config, config_dict['system'])
    
    def _update_dataclass_from_dict(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file."""
        config_data = {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'data': asdict(self.data_config),
            'system': asdict(self.system_config)
        }
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_data, f, default_flow_style=False)
                elif config_file.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {config_file.suffix}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate model configuration
            assert self.model_config.tts_encoder_dim > 0
            assert self.model_config.stt_encoder_dim > 0
            assert self.model_config.dropout_rate >= 0.0 and self.model_config.dropout_rate <= 1.0
            
            # Validate training configuration
            assert self.training_config.batch_size > 0
            assert self.training_config.learning_rate > 0
            assert self.training_config.gradient_accumulation_steps > 0
            
            # Validate data configuration
            assert self.data_config.sample_rate > 0
            assert self.data_config.n_mels > 0
            assert self.data_config.max_text_length > 0
            
            # Validate system configuration
            assert self.system_config.seed >= 0
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load configuration manager with optional config file."""
    return ConfigManager(config_path)

# Default configuration instance
default_config = ConfigManager()