"""
Test suite for HiFi-GAN vocoder components.
Validates generator, discriminators, and vocoder functionality.
"""

import unittest
import torch
import numpy as np

from src.utils.config import ModelConfig, DataConfig
from src.models.hifigan_vocoder import (
    HiFiGANGenerator, HiFiGANVocoder, HiFiGANPeriodDiscriminator,
    HiFiGANMultiPeriodDiscriminator, HiFiGANScaleDiscriminator,
    HiFiGANMultiScaleDiscriminator, AudioProcessor
)

class TestHiFiGANComponents(unittest.TestCase):
    """Test cases for HiFi-GAN vocoder components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configuration
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.model_config.data_config = self.data_config
        
        # Test data dimensions
        self.batch_size = 2
        self.n_mels = self.data_config.n_mels
        self.mel_time = 50
        self.audio_length = self.mel_time * self.data_config.hop_length
        
        # Create test tensors
        self.test_mel = torch.randn(self.batch_size, self.n_mels, self.mel_time)
        self.test_audio = torch.randn(self.batch_size, 1, self.audio_length)
    
    def test_hifigan_generator(self):
        """Test HiFi-GAN generator."""
        generator = HiFiGANGenerator(self.model_config)
        
        # Test forward pass
        generated_audio = generator(self.test_mel)
        
        # Check output shape
        expected_length = self.mel_time * 8 * 8 * 2 * 2  # 256x upsampling
        self.assertEqual(generated_audio.shape[0], self.batch_size)
        self.assertEqual(generated_audio.shape[1], 1)  # Mono audio
        self.assertTrue(abs(generated_audio.shape[2] - expected_length) <= 100)  # Allow some variation
        
        # Check output range (tanh activation should bound to [-1, 1])
        self.assertTrue(torch.all(generated_audio >= -1.0))
        self.assertTrue(torch.all(generated_audio <= 1.0))
        
        # Test with different input sizes
        large_mel = torch.randn(self.batch_size, self.n_mels, 100)
        large_audio = generator(large_mel)
        self.assertEqual(large_audio.shape[1], 1)
        self.assertGreater(large_audio.shape[2], generated_audio.shape[2])
    
    def test_period_discriminator(self):
        """Test period-based discriminator."""
        periods = [2, 3, 5]
        for period in periods:
            with self.subTest(period=period):
                discriminator = HiFiGANPeriodDiscriminator(period)
                
                # Test forward pass
                output, feature_maps = discriminator(self.test_audio)
                
                # Check output shape
                # Output should be smaller due to strided convolutions
                self.assertEqual(output.shape[0], self.batch_size)
                self.assertEqual(output.shape[1], 1)
                self.assertLess(output.shape[2], self.audio_length)
                
                # Check feature maps
                self.assertIsInstance(feature_maps, list)
                self.assertGreater(len(feature_maps), 0)
                for fmap in feature_maps:
                    self.assertEqual(fmap.shape[0], self.batch_size)
    
    def test_multi_period_discriminator(self):
        """Test multi-period discriminator."""
        mpd = HiFiGANMultiPeriodDiscriminator([2, 3, 5])
        
        # Test forward pass
        outputs, feature_maps_list = mpd(self.test_audio)
        
        # Check outputs
        self.assertEqual(len(outputs), 3)  # 3 periods
        for output in outputs:
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[1], 1)
        
        # Check feature maps
        self.assertEqual(len(feature_maps_list), 3)
        for feature_maps in feature_maps_list:
            self.assertIsInstance(feature_maps, list)
            self.assertGreater(len(feature_maps), 0)
    
    def test_scale_discriminator(self):
        """Test scale-based discriminator."""
        # Test without spectral norm
        discriminator = HiFiGANScaleDiscriminator(use_spectral_norm=False)
        output, feature_maps = discriminator(self.test_audio)
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], 1)
        self.assertIsInstance(feature_maps, list)
        self.assertGreater(len(feature_maps), 0)
        
        # Test with spectral norm
        discriminator_sn = HiFiGANScaleDiscriminator(use_spectral_norm=True)
        output_sn, feature_maps_sn = discriminator_sn(self.test_audio)
        
        self.assertEqual(output_sn.shape, output.shape)
        self.assertEqual(len(feature_maps_sn), len(feature_maps))
    
    def test_multi_scale_discriminator(self):
        """Test multi-scale discriminator."""
        msd = HiFiGANMultiScaleDiscriminator(scales=3)
        
        # Test forward pass
        outputs, feature_maps_list = msd(self.test_audio)
        
        # Check outputs (3 scales)
        self.assertEqual(len(outputs), 3)
        for i, output in enumerate(outputs):
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[1], 1)
            # Each scale should have progressively smaller output
            if i > 0:
                self.assertLessEqual(output.shape[2], outputs[i-1].shape[2])
        
        # Check feature maps
        self.assertEqual(len(feature_maps_list), 3)
        for feature_maps in feature_maps_list:
            self.assertIsInstance(feature_maps, list)
            self.assertGreater(len(feature_maps), 0)
    
    def test_hifigan_vocoder(self):
        """Test complete HiFi-GAN vocoder."""
        vocoder = HiFiGANVocoder(self.model_config)
        
        # Test generator forward pass
        generated_audio = vocoder(self.test_mel)
        self.assertEqual(generated_audio.shape[0], self.batch_size)
        self.assertEqual(generated_audio.shape[1], 1)
        
        # Test generator forward with features
        gen_audio, features = vocoder.generator_forward(self.test_mel)
        self.assertEqual(gen_audio.shape, generated_audio.shape)
        self.assertIn('mpd_features', features)
        self.assertIn('msd_features', features)
        self.assertIsInstance(features['mpd_features'], list)
        self.assertIsInstance(features['msd_features'], list)
        
        # Test discriminator forward pass
        disc_outputs = vocoder.discriminator_forward(self.test_audio, generated_audio)
        
        # Check all required outputs
        required_keys = [
            'real_mpd_outputs', 'gen_mpd_outputs',
            'real_msd_outputs', 'gen_msd_outputs',
            'real_mpd_features', 'gen_mpd_features',
            'real_msd_features', 'gen_msd_features'
        ]
        
        for key in required_keys:
            self.assertIn(key, disc_outputs)
        
        # Check MPD outputs
        self.assertEqual(len(disc_outputs['real_mpd_outputs']), 5)  # 5 periods
        self.assertEqual(len(disc_outputs['gen_mpd_outputs']), 5)
        
        # Check MSD outputs
        self.assertEqual(len(disc_outputs['real_msd_outputs']), 3)  # 3 scales
        self.assertEqual(len(disc_outputs['gen_msd_outputs']), 3)
    
    def test_model_sizes(self):
        """Test model size calculations."""
        vocoder = HiFiGANVocoder(self.model_config)
        sizes = vocoder.get_model_size()
        
        # Check all components have parameters
        self.assertGreater(sizes['generator'], 0)
        self.assertGreater(sizes['mpd'], 0)
        self.assertGreater(sizes['msd'], 0)
        self.assertGreater(sizes['total'], 0)
        
        # Check total is sum of parts
        expected_total = sizes['generator'] + sizes['mpd'] + sizes['msd']
        self.assertEqual(sizes['total'], expected_total)
        
        print(f"Generator parameters: {sizes['generator']:,}")
        print(f"MPD parameters: {sizes['mpd']:,}")
        print(f"MSD parameters: {sizes['msd']:,}")
        print(f"Total parameters: {sizes['total']:,}")
    
    def test_audio_processor(self):
        """Test audio processing utilities."""
        processor = AudioProcessor(self.model_config)
        
        # Test audio to waveform conversion
        waveform = processor.audio_to_waveform(self.test_audio)
        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.shape[0], self.batch_size)
        self.assertEqual(waveform.shape[1], self.audio_length)
        
        # Test audio normalization
        loud_audio = torch.ones_like(self.test_audio) * 2.0  # Amplitude > 1.0
        normalized = processor.normalize_audio(loud_audio)
        max_amplitude = torch.max(torch.abs(normalized))
        self.assertLessEqual(max_amplitude, 0.95)
        
        # Test audio length calculation
        calculated_length = processor.calculate_audio_length(self.mel_time)
        expected_length = self.mel_time * self.data_config.hop_length
        self.assertEqual(calculated_length, expected_length)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the vocoder."""
        vocoder = HiFiGANVocoder(self.model_config)
        vocoder.train()
        
        # Test generator gradients
        test_mel = self.test_mel.clone().requires_grad_(True)
        generated_audio = vocoder(test_mel)
        gen_loss = generated_audio.sum()
        gen_loss.backward()
        
        self.assertIsNotNone(test_mel.grad)
        self.assertTrue(torch.isfinite(test_mel.grad).all())
        
        # Test discriminator gradients
        vocoder.eval()
        with torch.no_grad():
            generated_audio = vocoder(self.test_mel)
        
        vocoder.train()
        disc_outputs = vocoder.discriminator_forward(self.test_audio, generated_audio)
        
        # Combine discriminator outputs for loss
        disc_loss = sum(torch.sum(out) for out in disc_outputs['real_mpd_outputs'])
        disc_loss += sum(torch.sum(out) for out in disc_outputs['gen_mpd_outputs'])
        disc_loss += sum(torch.sum(out) for out in disc_outputs['real_msd_outputs'])
        disc_loss += sum(torch.sum(out) for out in disc_outputs['gen_msd_outputs'])
        
        # Clear previous gradients
        vocoder.zero_grad()
        disc_loss.backward()
        
        # Check that discriminator parameters have gradients
        for param in vocoder.mpd.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                break  # Just check first parameter with gradients
    
    def test_model_consistency(self):
        """Test model consistency with same inputs."""
        vocoder = HiFiGANVocoder(self.model_config)
        vocoder.eval()
        
        with torch.no_grad():
            # Run same input twice
            output1 = vocoder(self.test_mel)
            output2 = vocoder(self.test_mel)
            
            # Check that outputs are identical
            self.assertTrue(torch.allclose(output1, output2))

if __name__ == '__main__':
    unittest.main()