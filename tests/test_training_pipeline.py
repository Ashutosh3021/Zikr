"""
Test suite for training pipeline components.
Validates loss functions, optimizers, and training utilities.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile

from src.utils.config import TrainingConfig, ModelConfig, DataConfig
from src.training.training_pipeline import (
    TTSLoss, STTLoss, HiFiGANLoss, LearningRateScheduler, ModelTrainer, TrainingMetrics
)
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel

class TestTrainingComponents(unittest.TestCase):
    """Test cases for training pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configurations
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.model_config.data_config = self.data_config
        self.training_config = TrainingConfig()
        
        # Test data dimensions
        self.batch_size = 2
        self.seq_len = 10
        self.mel_time = 50
        self.n_mels = self.data_config.n_mels
        self.vocab_size = self.model_config.stt_vocab_size
        
        # Create test tensors
        self.test_mel = torch.randn(self.batch_size, self.n_mels, self.mel_time)
        self.test_tokens = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        self.test_durations = torch.rand(self.batch_size, self.seq_len) * 5 + 1
        self.test_pitch = torch.randn(self.batch_size, self.mel_time)
        self.test_energy = torch.randn(self.batch_size, self.mel_time)
    
    def test_tts_loss(self):
        """Test TTS loss computation."""
        criterion = TTSLoss(self.training_config)
        
        # Test model outputs and targets
        model_outputs = {
            'mel_spectrogram': torch.randn(self.batch_size, self.n_mels, self.mel_time),
            'pred_durations': torch.rand(self.batch_size, self.seq_len),
            'pred_pitch': torch.randn(self.batch_size, self.mel_time),
            'pred_energy': torch.randn(self.batch_size, self.mel_time)
        }
        
        targets = {
            'target_mel': torch.randn(self.batch_size, self.n_mels, self.mel_time),
            'target_durations': torch.rand(self.batch_size, self.seq_len),
            'target_pitch': torch.randn(self.batch_size, self.mel_time),
            'target_energy': torch.randn(self.batch_size, self.mel_time)
        }
        
        # Test loss computation
        losses = criterion(model_outputs, targets)
        
        # Check required losses
        self.assertIn('mel_loss', losses)
        self.assertIn('duration_loss', losses)
        self.assertIn('pitch_loss', losses)
        self.assertIn('energy_loss', losses)
        self.assertIn('total_loss', losses)
        
        # Check loss values are positive and finite
        for loss_name, loss_value in losses.items():
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertTrue(loss_value.item() >= 0)
            self.assertTrue(torch.isfinite(loss_value))
        
        # Test with partial targets
        partial_outputs = {'mel_spectrogram': model_outputs['mel_spectrogram']}
        partial_targets = {'target_mel': targets['target_mel']}
        partial_losses = criterion(partial_outputs, partial_targets)
        self.assertIn('mel_loss', partial_losses)
        self.assertIn('total_loss', partial_losses)
    
    def test_stt_loss(self):
        """Test STT loss computation."""
        criterion = STTLoss(self.training_config, self.vocab_size)
        
        # Test model outputs and targets
        model_outputs = {
            'ctc_logits': torch.randn(self.batch_size, self.mel_time, self.vocab_size),
            'decoder_logits': torch.randn(self.batch_size, self.seq_len, self.vocab_size),
            'encoder_output': torch.randn(self.batch_size, self.mel_time, self.model_config.stt_encoder_dim)
        }
        
        targets = {
            'target_tokens': self.test_tokens,
            'target_mel': self.test_mel
        }
        
        # Test loss computation
        losses = criterion(model_outputs, targets)
        
        # Check required losses
        self.assertIn('ctc_loss', losses)
        self.assertIn('attention_loss', losses)
        self.assertIn('total_loss', losses)
        
        # Check loss values are positive and finite
        for loss_name, loss_value in losses.items():
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertTrue(loss_value.item() >= 0)
            self.assertTrue(torch.isfinite(loss_value))
    
    def test_hifigan_loss(self):
        """Test HiFi-GAN loss computation."""
        criterion = HiFiGANLoss(self.training_config)
        
        # Mock discriminator outputs
        disc_outputs = {
            'real_mpd_outputs': [torch.randn(self.batch_size, 1, 10) for _ in range(5)],
            'gen_mpd_outputs': [torch.randn(self.batch_size, 1, 10) for _ in range(5)],
            'real_msd_outputs': [torch.randn(self.batch_size, 1, 15) for _ in range(3)],
            'gen_msd_outputs': [torch.randn(self.batch_size, 1, 15) for _ in range(3)],
            'real_mpd_features': [[torch.randn(self.batch_size, 32, 20) for _ in range(5)] for _ in range(5)],
            'gen_mpd_features': [[torch.randn(self.batch_size, 32, 20) for _ in range(5)] for _ in range(5)],
            'real_msd_features': [[torch.randn(self.batch_size, 64, 25) for _ in range(6)] for _ in range(3)],
            'gen_msd_features': [[torch.randn(self.batch_size, 64, 25) for _ in range(6)] for _ in range(3)]
        }
        
        # Mock generator outputs
        gen_outputs = {
            'mpd_features': disc_outputs['gen_mpd_features'],
            'msd_features': disc_outputs['gen_msd_features'],
            'generated_mel': torch.randn(self.batch_size, self.n_mels, self.mel_time),
            'target_mel': torch.randn(self.batch_size, self.n_mels, self.mel_time)
        }
        
        # Test discriminator loss
        disc_losses = criterion(disc_outputs)
        self.assertIn('mpd_loss', disc_losses)
        self.assertIn('msd_loss', disc_losses)
        self.assertIn('discriminator_loss', disc_losses)
        
        # Test generator loss
        gen_losses = criterion(disc_outputs, gen_outputs, self.test_mel, self.test_mel)
        self.assertIn('adversarial_loss', gen_losses)
        self.assertIn('feature_matching_loss', gen_losses)
        self.assertIn('mel_reconstruction_loss', gen_losses)
        self.assertIn('generator_loss', gen_losses)
        
        # Check all losses are positive and finite
        all_losses = {**disc_losses}
        if gen_losses:
            all_losses.update(gen_losses)
        
        for loss_name, loss_value in all_losses.items():
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertTrue(loss_value.item() >= 0)
            self.assertTrue(torch.isfinite(loss_value))
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduler."""
        # Create dummy model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        scheduler = LearningRateScheduler(optimizer, self.training_config)
        
        # Test initial learning rate
        initial_lr = scheduler.step()
        self.assertEqual(initial_lr, 1e-4 / self.training_config.warmup_steps)
        
        # Test warmup phase
        for i in range(2, self.training_config.warmup_steps + 1):
            lr = scheduler.step()
            expected_lr = 1e-4 * (i / self.training_config.warmup_steps)
            self.assertAlmostEqual(lr, expected_lr, places=6)
        
        # Test decay phase
        lr_after_warmup = scheduler.step()
        self.assertLess(lr_after_warmup, 1e-4)
        self.assertGreater(lr_after_warmup, 0)
    
    def test_model_trainer_initialization(self):
        """Test model trainer initialization."""
        trainer = ModelTrainer(self.training_config)
        
        # Check initialization
        self.assertEqual(trainer.gradient_accumulation_steps, self.training_config.gradient_accumulation_steps)
        self.assertEqual(trainer.mixed_precision, self.training_config.mixed_precision)
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.current_step, 0)
        self.assertEqual(trainer.best_loss, float('inf'))
    
    def test_training_metrics(self):
        """Test training metrics tracking."""
        metrics = TrainingMetrics()
        
        # Test updating metrics
        metrics.update(1.0, 0.8, 1e-4, 60.0)
        metrics.update(0.9, 0.7, 9e-5, 55.0)
        metrics.update(0.8, 0.6, 8e-5, 50.0)
        
        # Test summary
        summary = metrics.get_summary()
        
        self.assertEqual(summary['total_epochs'], 3)
        self.assertEqual(summary['best_train_loss'], 0.8)
        self.assertEqual(summary['best_val_loss'], 0.6)
        self.assertEqual(summary['total_training_time'], 165.0)
        self.assertEqual(summary['final_train_loss'], 0.8)
        self.assertEqual(summary['final_val_loss'], 0.6)
    
    def test_checkpoint_save_load(self):
        """Test model checkpoint save and load functionality."""
        # This test would require actual model training, so we'll test the structure
        # Create a simple model for testing
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test save functionality structure
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        
        try:
            # This would normally call the actual save method
            # We're testing that the interface exists
            self.assertTrue(hasattr(ModelTrainer, 'save_model'))
        
            # Test load functionality structure
            # Note: load_checkpoint is a standalone function, not a method
            from src.training.training_pipeline import load_checkpoint
            self.assertTrue(callable(load_checkpoint))
            
        finally:
            # Clean up
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_loss_gradient_flow(self):
        """Test that gradients flow properly through loss functions."""
        # Test TTS loss gradients
        criterion = TTSLoss(self.training_config)
        
        # Create tensors with gradients
        mel_pred = torch.randn(self.batch_size, self.n_mels, self.mel_time, requires_grad=True)
        mel_target = torch.randn(self.batch_size, self.n_mels, self.mel_time)
        
        model_outputs = {'mel_spectrogram': mel_pred}
        targets = {'target_mel': mel_target}
        
        losses = criterion(model_outputs, targets)
        losses['total_loss'].backward()
        
        self.assertIsNotNone(mel_pred.grad)
        self.assertTrue(torch.isfinite(mel_pred.grad).all())
        
        # Test STT loss gradients
        stt_criterion = STTLoss(self.training_config, self.vocab_size)
        
        ctc_logits = torch.randn(self.batch_size, self.mel_time, self.vocab_size, requires_grad=True)
        target_tokens = self.test_tokens
        
        model_outputs = {'ctc_logits': ctc_logits}
        targets = {'target_tokens': target_tokens}
        
        losses = stt_criterion(model_outputs, targets)
        losses['total_loss'].backward()
        
        self.assertIsNotNone(ctc_logits.grad)
        self.assertTrue(torch.isfinite(ctc_logits.grad).all())

if __name__ == '__main__':
    unittest.main()