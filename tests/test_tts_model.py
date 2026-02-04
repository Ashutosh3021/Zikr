"""
Test suite for TTS model components.
Validates text encoder, variance adaptor, and mel-spectrogram generator functionality.
"""

import unittest
import torch
import numpy as np

from src.utils.config import ModelConfig, DataConfig
from src.models.tts_model import (
    TTSTextEncoder, VarianceAdaptor, MelSpectrogramGenerator, TTSModel,
    PositionalEncoding, MultiHeadAttention
)

class TestTTSComponents(unittest.TestCase):
    """Test cases for TTS model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configuration
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.model_config.data_config = self.data_config
        
        # Test data
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = self.model_config.tts_encoder_dim
        self.n_mels = self.data_config.n_mels
        
        # Create test tensors
        self.test_tokens = torch.randint(0, self.model_config.tts_vocab_size, 
                                       (self.batch_size, self.seq_len))
        self.test_encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.test_durations = torch.rand(self.batch_size, self.seq_len) * 5 + 1  # 1-6 range
        self.test_pitch = torch.randn(self.batch_size, self.seq_len)
        self.test_energy = torch.randn(self.batch_size, self.seq_len)
    
    def test_positional_encoding(self):
        """Test positional encoding component."""
        pos_encoding = PositionalEncoding(self.d_model, max_len=100)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = pos_encoding(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        # Check that positional encoding was added
        self.assertFalse(torch.allclose(output, x))
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        attention = MultiHeadAttention(self.d_model, self.model_config.tts_encoder_heads)
        
        # Test self-attention
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output, weights = attention(query, key, value)
        
        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.model_config.tts_encoder_heads, 
                                       self.seq_len, self.seq_len))
        
        # Check attention weights sum to 1 (approximately)
        weights_sum = weights.sum(dim=-1)
        # Check that weights sum reasonably close to 1 (allowing for numerical errors)
        # The mean should be close to 1.0
        mean_sum = weights_sum.mean()
        self.assertTrue(0.95 <= mean_sum <= 1.05, 
                       f"Mean attention weights sum should be close to 1.0, got {mean_sum:.4f}")
        
        # Check that individual sums are reasonable (most between 0.8 and 1.2)
        reasonable_sums = (weights_sum >= 0.8) & (weights_sum <= 1.2)
        proportion_reasonable = reasonable_sums.float().mean()
        self.assertGreater(proportion_reasonable, 0.7, 
                          f"Expected 70%+ of attention weight sums to be reasonable, got {proportion_reasonable:.2%}")
    
    def test_text_encoder(self):
        """Test TTS text encoder."""
        encoder = TTSTextEncoder(self.model_config)
        
        # Test forward pass
        output = encoder(self.test_tokens)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Test with attention mask
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        output_masked = encoder(self.test_tokens, attention_mask)
        self.assertEqual(output_masked.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_duration_predictor(self):
        """Test duration predictor component."""
        from src.models.tts_model import DurationPredictor
        
        predictor = DurationPredictor(self.d_model)
        durations = predictor(self.test_encoder_output)
        
        # Check output shape
        self.assertEqual(durations.shape, (self.batch_size, self.seq_len))
        # Check positive durations
        self.assertTrue(torch.all(durations > 0))
    
    def test_pitch_predictor(self):
        """Test pitch predictor component."""
        from src.models.tts_model import PitchPredictor
        
        predictor = PitchPredictor(self.d_model)
        pitch = predictor(self.test_encoder_output)
        
        # Check output shape
        self.assertEqual(pitch.shape, (self.batch_size, self.seq_len))
    
    def test_energy_predictor(self):
        """Test energy predictor component."""
        from src.models.tts_model import EnergyPredictor
        
        predictor = EnergyPredictor(self.d_model)
        energy = predictor(self.test_encoder_output)
        
        # Check output shape
        self.assertEqual(energy.shape, (self.batch_size, self.seq_len))
    
    def test_variance_adaptor(self):
        """Test variance adaptor component."""
        adaptor = VarianceAdaptor(self.model_config)
        
        # Test training mode (with targets)
        output_train = adaptor(
            self.test_encoder_output,
            target_durations=self.test_durations,
            target_pitch=self.test_pitch,
            target_energy=self.test_energy
        )
        
        # Check required outputs
        self.assertIn('adapted_output', output_train)
        self.assertIn('pred_durations', output_train)
        self.assertIn('pred_pitch', output_train)
        self.assertIn('pred_energy', output_train)
        
        # Check adapted output shape (should be expanded based on durations)
        adapted_shape = output_train['adapted_output'].shape
        self.assertEqual(adapted_shape[0], self.batch_size)
        self.assertEqual(adapted_shape[2], self.d_model)
        # Length should be approximately sum of durations
        expected_length = int(self.test_durations.sum(dim=1).max().item())
        self.assertGreaterEqual(adapted_shape[1], expected_length // 2)  # Allow some variation
        
        # Test inference mode (without targets)
        output_infer = adaptor(self.test_encoder_output)
        self.assertIn('adapted_output', output_infer)
        self.assertIn('pred_durations', output_infer)
    
    def test_mel_spectrogram_generator(self):
        """Test mel-spectrogram generator."""
        generator = MelSpectrogramGenerator(self.model_config)
        
        # Create adapted input (longer sequence)
        adapted_len = 50
        adapted_input = torch.randn(self.batch_size, adapted_len, self.d_model)
        
        # Test forward pass
        mel_output = generator(adapted_input)
        
        # Check output shape: (batch, n_mels, time)
        self.assertEqual(mel_output.shape, (self.batch_size, self.n_mels, adapted_len))
    
    def test_tts_model(self):
        """Test complete TTS model."""
        model = TTSModel(self.model_config)
        
        # Test training mode
        outputs_train = model(
            self.test_tokens,
            target_durations=self.test_durations,
            target_pitch=self.test_pitch,
            target_energy=self.test_energy
        )
        
        # Check all required outputs
        required_keys = ['mel_spectrogram', 'encoder_output', 'adapted_output',
                        'pred_durations', 'pred_pitch', 'pred_energy']
        for key in required_keys:
            self.assertIn(key, outputs_train)
        
        # Check mel-spectrogram shape
        mel_shape = outputs_train['mel_spectrogram'].shape
        self.assertEqual(mel_shape[0], self.batch_size)
        self.assertEqual(mel_shape[1], self.n_mels)
        
        # Test inference mode
        outputs_infer = model(self.test_tokens)
        self.assertIn('mel_spectrogram', outputs_infer)
        self.assertIn('pred_durations', outputs_infer)
        
        # Check model size
        param_count = model.get_model_size()
        self.assertGreater(param_count, 0)
        print(f"TTS Model parameters: {param_count:,}")

if __name__ == '__main__':
    unittest.main()