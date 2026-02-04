"""
Test suite for STT model components.
Validates audio encoder, attention mechanisms, and decoder functionality.
"""

import unittest
import torch
import numpy as np

from src.utils.config import ModelConfig, DataConfig
from src.models.stt_model import (
    STTAudioEncoder, STTDecoder, STTModel, CTCHead,
    STTPositionalEncoding, STTMultiHeadAttention,
    generate_square_subsequent_mask, generate_padding_mask
)

class TestSTTComponents(unittest.TestCase):
    """Test cases for STT model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configuration
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.model_config.data_config = self.data_config
        
        # Test data dimensions
        self.batch_size = 2
        self.n_mels = self.data_config.n_mels
        self.time_steps = 100
        self.d_model = self.model_config.stt_encoder_dim
        self.vocab_size = self.model_config.stt_vocab_size
        self.tgt_seq_len = 15
        
        # Create test tensors
        self.test_mel = torch.randn(self.batch_size, self.n_mels, self.time_steps)
        self.test_tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.tgt_seq_len))
        self.test_encoder_output = torch.randn(self.batch_size, self.time_steps // 2, self.d_model)  # After subsampling
    
    def test_positional_encoding(self):
        """Test positional encoding component."""
        pos_encoding = STTPositionalEncoding(self.d_model, max_len=1000)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        output = pos_encoding(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        # Check that positional encoding was added
        self.assertFalse(torch.allclose(output, x))
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        attention = STTMultiHeadAttention(self.d_model, self.model_config.stt_encoder_heads)
        
        # Test self-attention
        query = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        
        output, weights = attention(query, key, value)
        
        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.model_config.stt_encoder_heads, 
                                       self.tgt_seq_len, self.tgt_seq_len))
        
        # Check attention weights sum reasonably close to 1
        weights_sum = weights.sum(dim=-1)
        mean_sum = weights_sum.mean()
        self.assertTrue(0.9 <= mean_sum <= 1.1, 
                       f"Mean attention weights sum should be close to 1.0, got {mean_sum:.4f}")
    
    def test_audio_encoder(self):
        """Test STT audio encoder."""
        encoder = STTAudioEncoder(self.model_config)
        
        # Test forward pass
        output = encoder(self.test_mel)
        
        # Check output shape (should be subsampled in time dimension)
        expected_time = self.time_steps // 2  # Due to stride=2 in subsample conv
        self.assertEqual(output.shape, (self.batch_size, expected_time, self.d_model))
        
        # Test with attention mask
        encoder_mask = torch.ones(self.batch_size, expected_time)
        output_masked = encoder(self.test_mel, encoder_mask)
        self.assertEqual(output_masked.shape, (self.batch_size, expected_time, self.d_model))
        
        # Test with different input sizes
        large_mel = torch.randn(self.batch_size, self.n_mels, 200)
        large_output = encoder(large_mel)
        self.assertEqual(large_output.shape[1], 100)  # 200 // 2 = 100
    
    def test_ctc_head(self):
        """Test CTC head component."""
        ctc_head = CTCHead(self.d_model, self.vocab_size)
        
        # Test forward pass
        logits = ctc_head(self.test_encoder_output)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.time_steps // 2, self.vocab_size))
        # Check log softmax output (should be negative)
        self.assertTrue(torch.all(logits <= 0))
    
    def test_decoder_layer(self):
        """Test STT decoder layer."""
        from src.models.stt_model import STTDecoderLayer
        
        decoder_layer = STTDecoderLayer(
            self.d_model, 
            self.model_config.stt_decoder_heads,
            dropout=self.model_config.dropout_rate
        )
        
        # Test forward pass
        target_seq = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        memory = self.test_encoder_output
        
        output = decoder_layer(target_seq, memory)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
    
    def test_decoder(self):
        """Test STT decoder."""
        decoder = STTDecoder(self.model_config)
        
        # Test forward pass
        logits = decoder(self.test_tokens, self.test_encoder_output)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.tgt_seq_len, self.vocab_size))
        
        # Test with masks
        tgt_mask = generate_square_subsequent_mask(self.tgt_seq_len, self.test_tokens.device)
        # Expand to proper dimensions for attention (batch, heads, seq, seq)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_seq_len, tgt_seq_len)
        tgt_mask = tgt_mask.expand(self.batch_size, self.model_config.stt_decoder_heads, -1, -1)
        
        memory_mask = torch.ones(self.batch_size, self.test_encoder_output.size(1))
        # Expand memory mask to (batch, 1, 1, encoder_seq_len)
        memory_mask = memory_mask.unsqueeze(1).unsqueeze(2)
        
        logits_masked = decoder(self.test_tokens, self.test_encoder_output, tgt_mask, memory_mask)
        self.assertEqual(logits_masked.shape, (self.batch_size, self.tgt_seq_len, self.vocab_size))
    
    def test_stt_model(self):
        """Test complete STT model."""
        model = STTModel(self.model_config)
        
        # Test inference mode (no targets)
        outputs_infer = model(self.test_mel)
        
        # Check required outputs
        self.assertIn('encoder_output', outputs_infer)
        self.assertIn('ctc_logits', outputs_infer)
        self.assertNotIn('decoder_logits', outputs_infer)  # No decoder without targets
        
        # Check shapes
        encoder_shape = outputs_infer['encoder_output'].shape
        self.assertEqual(encoder_shape[0], self.batch_size)
        self.assertEqual(encoder_shape[2], self.d_model)
        
        ctc_shape = outputs_infer['ctc_logits'].shape
        self.assertEqual(ctc_shape[0], self.batch_size)
        self.assertEqual(ctc_shape[2], self.vocab_size)
        
        # Test training mode (with targets)
        outputs_train = model(self.test_mel, self.test_tokens)
        
        # Check additional decoder output
        self.assertIn('decoder_logits', outputs_train)
        decoder_shape = outputs_train['decoder_logits'].shape
        self.assertEqual(decoder_shape, (self.batch_size, self.tgt_seq_len, self.vocab_size))
        
        # Check model size
        param_count = model.get_model_size()
        self.assertGreater(param_count, 0)
        print(f"STT Model parameters: {param_count:,}")
    
    def test_mask_generation(self):
        """Test mask generation utilities."""
        # Test causal mask
        causal_mask = generate_square_subsequent_mask(5, torch.device('cpu'))
        self.assertEqual(causal_mask.shape, (5, 5))
        # Check that it's lower triangular
        self.assertTrue(torch.allclose(causal_mask, torch.tril(causal_mask)))
        
        # Test padding mask
        seq_with_pad = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])  # 0 is padding
        pad_mask = generate_padding_mask(seq_with_pad, pad_idx=0)
        self.assertEqual(pad_mask.shape, (2, 1, 5))
        # Check that non-padding positions are True
        self.assertTrue(pad_mask[0, 0, 0].item())  # First token (1) should be True
        self.assertFalse(pad_mask[0, 0, 3].item())  # Padding (0) should be False
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = STTModel(self.model_config)
        model.train()
        
        # Test with requires_grad (only for floating point tensors)
        test_mel = self.test_mel.clone().requires_grad_(True)
        # Token indices can't require gradients, so we'll test with embedding instead
        
        outputs = model(test_mel, self.test_tokens)
        
        # Check that we can compute gradients
        loss = outputs['decoder_logits'].sum() + outputs['ctc_logits'].sum()
        loss.backward()
        
        # Check that gradients were computed for mel input
        self.assertIsNotNone(test_mel.grad)
        self.assertTrue(torch.isfinite(test_mel.grad).all())
    
    def test_model_consistency(self):
        """Test model consistency with same inputs."""
        model = STTModel(self.model_config)
        model.eval()
        
        with torch.no_grad():
            # Run same input twice
            output1 = model(self.test_mel)
            output2 = model(self.test_mel)
            
            # Check that outputs are identical
            self.assertTrue(torch.allclose(output1['ctc_logits'], output2['ctc_logits']))

if __name__ == '__main__':
    unittest.main()