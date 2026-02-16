#!/usr/bin/env python3
"""
Main training script for TTS-STT system.
Usage: python train.py --model [tts|stt|hifigan] --config config.yaml
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import time

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig, DataConfig, TrainingConfig
from src.utils.logger import setup_logger, get_logger
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel
from src.models.hifigan_vocoder import HiFiGANVocoder
from src.training.training_pipeline import ModelTrainer, TTSLoss, STTLoss, HiFiGANLoss
from src.data.audio_preprocessing import AudioPreprocessor
from src.data.text_preprocessing import TextNormalizer, Tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TTS, STT, or HiFi-GAN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train TTS model
  python train.py --model tts --data-dir ./data/tts --epochs 100
  
  # Train STT model
  python train.py --model stt --data-dir ./data/stt --epochs 100
  
  # Train HiFi-GAN vocoder
  python train.py --model hifigan --data-dir ./data/audio --epochs 1000
  
  # Resume training from checkpoint
  python train.py --model tts --checkpoint ./checkpoints/tts_epoch_50.pt
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['tts', 'stt', 'hifigan'],
                       help='Model to train: tts, stt, or hifigan')
    
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing training data')
    
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Initial learning rate')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    parser.add_argument('--validate-every', type=int, default=1,
                       help='Validate every N epochs')
    
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')
    
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    
    parser.add_argument('--local-rank', type=int, default=0,
                       help='Local rank for distributed training')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def setup_device(args):
    """Setup training device."""
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    
    return device


def load_config(args):
    """Load configuration from file or use defaults."""
    data_config = DataConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Link configs
    model_config.data_config = data_config
    
    # Override with custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Update configs
        if 'data' in custom_config:
            for key, value in custom_config['data'].items():
                if hasattr(data_config, key):
                    setattr(data_config, key, value)
        
        if 'model' in custom_config:
            for key, value in custom_config['model'].items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        if 'training' in custom_config:
            for key, value in custom_config['training'].items():
                if hasattr(training_config, key):
                    setattr(training_config, key, value)
    
    # Override with command line arguments
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.epochs = args.epochs
    training_config.mixed_precision = args.mixed_precision
    training_config.num_workers = args.num_workers
    
    return data_config, model_config, training_config


def create_model(model_type, model_config, device):
    """Create model based on type."""
    logger = get_logger()
    
    if model_type == 'tts':
        model = TTSModel(model_config)
        logger.info(f"Created TTS Model")
    elif model_type == 'stt':
        model = STTModel(model_config)
        logger.info(f"Created STT Model")
    elif model_type == 'hifigan':
        model = HiFiGANVocoder(model_config)
        logger.info(f"Created HiFi-GAN Vocoder")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Log model size
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} ({trainable_params:,} trainable)")
    
    return model


def create_loss_function(model_type, model_config):
    """Create loss function based on model type."""
    if model_type == 'tts':
        return TTSLoss()
    elif model_type == 'stt':
        return STTLoss(model_config.stt_vocab_size)
    elif model_type == 'hifigan':
        return HiFiGANLoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dataloaders(model_type, data_dir, data_config, training_config):
    """Create training and validation dataloaders."""
    logger = get_logger()
    
    # This is a placeholder - in a real implementation, you would load your actual datasets
    logger.info(f"Creating dataloaders for {model_type}")
    logger.info(f"Data directory: {data_dir}")
    
    # Return dummy dataloaders for now
    # In production, replace with actual dataset loading
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Return dummy data
            return {
                'text': 'dummy text',
                'audio': torch.randn(16000),
                'mel': torch.randn(80, 100)
            }
    
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(100)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, model_type):
    """Train for one epoch."""
    logger = get_logger()
    model.train()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        # In a real implementation, unpack batch based on model type
        
        optimizer.zero_grad()
        
        # Forward pass (placeholder - implement based on model type)
        if model_type == 'tts':
            # Dummy forward pass
            loss = torch.tensor(1.0, requires_grad=True)
        elif model_type == 'stt':
            loss = torch.tensor(1.0, requires_grad=True)
        elif model_type == 'hifigan':
            loss = torch.tensor(1.0, requires_grad=True)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, loss_fn, device, model_type):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # Dummy validation
            loss = torch.tensor(1.0)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, output_dir, model_type):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_type': model_type
    }
    
    checkpoint_path = os.path.join(output_dir, f'{model_type}_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(output_dir, f'{model_type}_latest.pt')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    logger = get_logger()
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    
    return start_epoch


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logger(log_level)
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("TTS-STT Training Pipeline")
    logger.info("="*60)
    
    # Setup device
    device = setup_device(args)
    logger.info(f"Using device: {device}")
    
    # Load configuration
    data_config, model_config, training_config = load_config(args)
    logger.info(f"Configuration loaded")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create model
    model = create_model(args.model, model_config, device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Create loss function
    loss_fn = create_loss_function(args.model, model_config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.model, args.data_dir, data_config, training_config
    )
    
    # Load checkpoint if resuming
    start_epoch = 1
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
    
    # Training loop
    logger.info("="*60)
    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, training_config.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, 
                device, epoch, args.model
            )
            
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if epoch % args.validate_every == 0 and val_loader:
                val_loss = validate(model, val_loader, loss_fn, device, args.model)
                logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, 
                        args.output_dir, f'{args.model}_best'
                    )
                    logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            if epoch % args.save_every == 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, epoch, train_loss,
                    args.output_dir, args.model
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info("-"*60)
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, train_loss,
            args.output_dir, f'{args.model}_interrupted'
        )
        logger.info(f"Saved interrupted checkpoint: {checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
