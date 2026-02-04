"""
Training pipeline implementation for TTS-STT system.
Implements loss functions, optimization strategies, and training loops as specified.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

from ..utils.logger import get_model_logger
from ..utils.config import TrainingConfig, ModelConfig
from ..models.tts_model import TTSModel
from ..models.stt_model import STTModel
from ..models.hifigan_vocoder import HiFiGANVocoder

logger = get_model_logger()

class TTSLoss(nn.Module):
    """Loss functions for TTS training following specification."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        logger.info("TTSLoss initialized with MSE and L1 losses")
    
    def forward(self, model_outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute TTS losses.
        
        Args:
            model_outputs: Dictionary containing model predictions
            targets: Dictionary containing target values
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Mel-spectrogram reconstruction loss
        if 'mel_spectrogram' in model_outputs and 'target_mel' in targets:
            mel_loss = self.l1_loss(model_outputs['mel_spectrogram'], targets['target_mel'])
            losses['mel_loss'] = mel_loss
        
        # Duration prediction loss
        if 'pred_durations' in model_outputs and 'target_durations' in targets:
            duration_loss = self.mse_loss(
                model_outputs['pred_durations'], 
                targets['target_durations']
            )
            losses['duration_loss'] = duration_loss
        
        # Pitch prediction loss
        if 'pred_pitch' in model_outputs and 'target_pitch' in targets:
            pitch_loss = self.l1_loss(
                model_outputs['pred_pitch'], 
                targets['target_pitch']
            )
            losses['pitch_loss'] = pitch_loss
        
        # Energy prediction loss
        if 'pred_energy' in model_outputs and 'target_energy' in targets:
            energy_loss = self.l1_loss(
                model_outputs['pred_energy'], 
                targets['target_energy']
            )
            losses['energy_loss'] = energy_loss
        
        # Total loss (weighted combination)
        total_loss = 0.0
        if 'mel_loss' in losses:
            total_loss += losses['mel_loss']
        if 'duration_loss' in losses:
            total_loss += 0.1 * losses['duration_loss']  # Weighted less
        if 'pitch_loss' in losses:
            total_loss += 0.1 * losses['pitch_loss']
        if 'energy_loss' in losses:
            total_loss += 0.1 * losses['energy_loss']
        
        losses['total_loss'] = total_loss
        
        return losses

class STTLoss(nn.Module):
    """Loss functions for STT training following specification."""
    
    def __init__(self, config: TrainingConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding
        
        logger.info("STTLoss initialized with CTC and CrossEntropy losses")
    
    def forward(self, model_outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute STT losses.
        
        Args:
            model_outputs: Dictionary containing model predictions
            targets: Dictionary containing target values
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # CTC loss
        if 'ctc_logits' in model_outputs and 'target_tokens' in targets:
            ctc_logits = model_outputs['ctc_logits']  # (batch, time, vocab)
            target_tokens = targets['target_tokens']  # (batch, seq_len)
            
            # Prepare for CTC loss
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (time, batch, vocab)
            
            # Calculate input lengths (non-padded frames)
            input_lengths = torch.full(
                size=(ctc_logits.size(0),), 
                fill_value=ctc_logits.size(1), 
                dtype=torch.long
            )
            
            # Calculate target lengths (non-padded tokens)
            target_lengths = (target_tokens != 0).sum(dim=1)  # Assuming 0 is padding
            
            ctc_loss = self.ctc_loss(log_probs, target_tokens, input_lengths, target_lengths)
            losses['ctc_loss'] = ctc_loss
        
        # Attention decoder loss
        if 'decoder_logits' in model_outputs and 'target_tokens' in targets:
            decoder_logits = model_outputs['decoder_logits']  # (batch, seq, vocab)
            target_tokens = targets['target_tokens']  # (batch, seq)
            
            # Shift targets for autoregressive training
            decoder_targets = target_tokens[:, 1:]  # Remove first token (usually <sos>)
            decoder_logits = decoder_logits[:, :-1, :]  # Remove last prediction
            
            # Reshape for cross entropy
            decoder_logits = decoder_logits.reshape(-1, self.vocab_size)
            decoder_targets = decoder_targets.reshape(-1)
            
            attention_loss = self.cross_entropy(decoder_logits, decoder_targets)
            losses['attention_loss'] = attention_loss
        
        # Total loss (CTC + attention with weighting)
        total_loss = 0.0
        if 'ctc_loss' in losses:
            total_loss += 0.3 * losses['ctc_loss']  # Weight CTC less
        if 'attention_loss' in losses:
            total_loss += 0.7 * losses['attention_loss']  # Weight attention more
        
        losses['total_loss'] = total_loss
        
        return losses

class HiFiGANLoss(nn.Module):
    """Loss functions for HiFi-GAN training following specification."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        logger.info("HiFiGANLoss initialized with adversarial and feature matching losses")
    
    def forward(self, discriminator_outputs: Dict[str, Any], 
                generator_outputs: Optional[Dict[str, Any]] = None,
                real_audio: Optional[torch.Tensor] = None,
                generated_audio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute HiFi-GAN losses for both generator and discriminator.
        
        Args:
            discriminator_outputs: Outputs from discriminator forward pass
            generator_outputs: Outputs from generator forward pass (for feature matching)
            real_audio: Real audio samples
            generated_audio: Generated audio samples
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Discriminator losses
        if discriminator_outputs:
            # Multi-period discriminator loss
            mpd_real_losses = []
            mpd_gen_losses = []
            
            for real_out, gen_out in zip(
                discriminator_outputs['real_mpd_outputs'],
                discriminator_outputs['gen_mpd_outputs']
            ):
                # Real loss (should be close to 1)
                real_loss = self.mse_loss(real_out, torch.ones_like(real_out))
                mpd_real_losses.append(real_loss)
                
                # Generated loss (should be close to 0)
                gen_loss = self.mse_loss(gen_out, torch.zeros_like(gen_out))
                mpd_gen_losses.append(gen_loss)
            
            losses['mpd_real_loss'] = torch.mean(torch.stack(mpd_real_losses))
            losses['mpd_gen_loss'] = torch.mean(torch.stack(mpd_gen_losses))
            losses['mpd_loss'] = losses['mpd_real_loss'] + losses['mpd_gen_loss']
            
            # Multi-scale discriminator loss
            msd_real_losses = []
            msd_gen_losses = []
            
            for real_out, gen_out in zip(
                discriminator_outputs['real_msd_outputs'],
                discriminator_outputs['gen_msd_outputs']
            ):
                real_loss = self.mse_loss(real_out, torch.ones_like(real_out))
                msd_real_losses.append(real_loss)
                
                gen_loss = self.mse_loss(gen_out, torch.zeros_like(gen_out))
                msd_gen_losses.append(gen_loss)
            
            losses['msd_real_loss'] = torch.mean(torch.stack(msd_real_losses))
            losses['msd_gen_loss'] = torch.mean(torch.stack(msd_gen_losses))
            losses['msd_loss'] = losses['msd_real_loss'] + losses['msd_gen_loss']
            
            # Total discriminator loss
            losses['discriminator_loss'] = losses['mpd_loss'] + losses['msd_loss']
        
        # Generator losses (if generator outputs provided)
        if generator_outputs and real_audio is not None and generated_audio is not None:
            # Adversarial loss (generator wants discriminator to output 1)
            adv_losses = []
            
            # Multi-period adversarial loss
            for gen_out in generator_outputs['mpd_features']:
                for fmap in gen_out:
                    adv_loss = self.mse_loss(fmap, torch.ones_like(fmap))
                    adv_losses.append(adv_loss)
            
            # Multi-scale adversarial loss
            for gen_out in generator_outputs['msd_features']:
                for fmap in gen_out:
                    adv_loss = self.mse_loss(fmap, torch.ones_like(fmap))
                    adv_losses.append(adv_loss)
            
            losses['adversarial_loss'] = torch.mean(torch.stack(adv_losses))
            
            # Feature matching loss
            feature_losses = []
            
            # Compare real and generated features
            for real_features, gen_features in zip(
                discriminator_outputs['real_mpd_features'],
                discriminator_outputs['gen_mpd_features']
            ):
                for real_fmap, gen_fmap in zip(real_features, gen_features):
                    feature_loss = self.l1_loss(gen_fmap, real_fmap)
                    feature_losses.append(feature_loss)
            
            for real_features, gen_features in zip(
                discriminator_outputs['real_msd_features'],
                discriminator_outputs['gen_msd_features']
            ):
                for real_fmap, gen_fmap in zip(real_features, gen_features):
                    feature_loss = self.l1_loss(gen_fmap, real_fmap)
                    feature_losses.append(feature_loss)
            
            losses['feature_matching_loss'] = torch.mean(torch.stack(feature_losses))
            
            # Mel-spectrogram reconstruction loss (if mel provided)
            if 'target_mel' in generator_outputs and 'generated_mel' in generator_outputs:
                mel_loss = self.l1_loss(
                    generator_outputs['generated_mel'],
                    generator_outputs['target_mel']
                )
                losses['mel_reconstruction_loss'] = mel_loss
            
            # Total generator loss
            total_gen_loss = 0.0
            if 'adversarial_loss' in losses:
                total_gen_loss += 0.5 * losses['adversarial_loss']  # Weight adversarial loss
            if 'feature_matching_loss' in losses:
                total_gen_loss += losses['feature_matching_loss']   # Full weight for feature matching
            if 'mel_reconstruction_loss' in losses:
                total_gen_loss += 0.1 * losses['mel_reconstruction_loss']  # Weighted less
            
            losses['generator_loss'] = total_gen_loss
        
        return losses

class LearningRateScheduler:
    """Learning rate scheduler following specification."""
    
    def __init__(self, optimizer: optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.current_step = 0
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps
        self.initial_lr = config.learning_rate
        
        logger.info(f"LearningRateScheduler initialized with warmup={self.warmup_steps}")
    
    def step(self) -> float:
        """Update learning rate and return current value."""
        self.current_step += 1
        
        # Warmup phase
        if self.current_step <= self.warmup_steps:
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class ModelTrainer:
    """Main training pipeline for TTS-STT models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.system_config.device if hasattr(config, 'system_config') else 'cuda')
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.mixed_precision = config.mixed_precision
        
        # Scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and torch.cuda.is_available() else None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"ModelTrainer initialized for device: {self.device}")
    
    def train_tts_model(self, model: TTSModel, train_loader: DataLoader, 
                       val_loader: DataLoader, save_path: str) -> Dict[str, Any]:
        """Train TTS model."""
        logger.info("Starting TTS model training")
        
        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Setup scheduler
        scheduler = LearningRateScheduler(optimizer, self.config)
        
        # Setup loss function
        criterion = TTSLoss(self.config)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.max_steps // len(train_loader)):
            self.current_epoch = epoch
            
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.current_step += 1
                num_batches += 1
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.mixed_precision and torch.cuda.is_available()):
                    outputs = model(
                        tokens=batch['tokens'],
                        target_durations=batch.get('durations'),
                        target_pitch=batch.get('pitch'),
                        target_energy=batch.get('energy')
                    )
                    
                    targets = {
                        'target_mel': batch['mel_spectrogram'],
                        'target_durations': batch.get('durations'),
                        'target_pitch': batch.get('pitch'),
                        'target_energy': batch.get('energy')
                    }
                    
                    losses = criterion(outputs, targets)
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(losses['total_loss']).backward()
                else:
                    losses['total_loss'].backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                # Update learning rate
                current_lr = scheduler.step()
                
                epoch_train_loss += losses['total_loss'].item()
                
                # Log progress
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {losses['total_loss'].item():.4f}, LR: {current_lr:.6f}")
            
            # Average training loss
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    outputs = model(
                        tokens=batch['tokens'],
                        target_durations=batch.get('durations'),
                        target_pitch=batch.get('pitch'),
                        target_energy=batch.get('energy')
                    )
                    
                    targets = {
                        'target_mel': batch['mel_spectrogram'],
                        'target_durations': batch.get('durations'),
                        'target_pitch': batch.get('pitch'),
                        'target_energy': batch.get('energy')
                    }
                    
                    losses = criterion(outputs, targets)
                    val_loss += losses['total_loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.save_model(model, optimizer, scheduler, save_path, epoch, avg_val_loss)
                logger.info(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': self.best_loss,
            'epochs_trained': self.current_epoch + 1
        }
    
    def save_model(self, model: nn.Module, optimizer: optim.Optimizer, 
                   scheduler: LearningRateScheduler, save_path: str, 
                   epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.__dict__,
            'epoch': epoch,
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model checkpoint saved to {save_path}")

# Utility functions
def create_trainer(config: TrainingConfig) -> ModelTrainer:
    """Factory function to create model trainer."""
    return ModelTrainer(config)

def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint

# Training metrics and monitoring
class TrainingMetrics:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def update(self, train_loss: float, val_loss: float, lr: float, epoch_time: float):
        """Update metrics history."""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['epoch_time'].append(epoch_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        return {
            'best_train_loss': min(self.metrics_history['train_loss']) if self.metrics_history['train_loss'] else None,
            'best_val_loss': min(self.metrics_history['val_loss']) if self.metrics_history['val_loss'] else None,
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else None,
            'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
            'total_epochs': len(self.metrics_history['train_loss']),
            'total_training_time': sum(self.metrics_history['epoch_time'])
        }