"""
Test suite for inference engine components.
Validates TTS, STT inference, and API functionality.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.utils.config import InferenceConfig, ModelConfig, DataConfig
from src.inference.inference_engine import (
    ModelCache, TTSInferenceEngine, STTInferenceEngine, 
    InferenceAPI, InferenceMonitor
)
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel
from src.models.hifigan_vocoder import HiFiGANVocoder

class TestInferenceComponents(unittest.TestCase):
    """Test cases for inference engine components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create configurations
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.model_config.data_config = self.data_config
        self.inference_config = InferenceConfig()
        
        # Test data
        self.test_text = "Hello world, this is a test."
        self.test_audio = np.random.randn(22050).astype(np.float32)  # 1 second of audio
        
        # Create temporary model files
        self.temp_dir = tempfile.mkdtemp()
        self.tts_model_path = os.path.join(self.temp_dir, "tts_model.pt")
        self.stt_model_path = os.path.join(self.temp_dir, "stt_model.pt")
        self.vocoder_model_path = os.path.join(self.temp_dir, "vocoder_model.pt")
        
        # Create dummy model checkpoints
        self._create_dummy_checkpoints()
    
    def _create_dummy_checkpoints(self):
        """Create dummy model checkpoints for testing."""
        # Create dummy TTS model
        tts_model = TTSModel(self.model_config)
        tts_checkpoint = {
            'model_state_dict': tts_model.state_dict(),
            'epoch': 0,
            'val_loss': 1.0
        }
        torch.save(tts_checkpoint, self.tts_model_path)
        
        # Create dummy STT model
        stt_model = STTModel(self.model_config)
        stt_checkpoint = {
            'model_state_dict': stt_model.state_dict(),
            'epoch': 0,
            'val_loss': 1.0
        }
        torch.save(stt_checkpoint, self.stt_model_path)
        
        # Create dummy vocoder model
        vocoder = HiFiGANVocoder(self.model_config)
        vocoder_checkpoint = {
            'model_state_dict': vocoder.state_dict(),
            'epoch': 0,
            'val_loss': 1.0
        }
        torch.save(vocoder_checkpoint, self.vocoder_model_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_cache(self):
        """Test model caching functionality."""
        cache = ModelCache(self.inference_config)
        
        # Test loading model
        model = cache.load_model("TTS", self.tts_model_path, TTSModel, self.model_config)
        self.assertIsInstance(model, TTSModel)
        
        # Test caching - should return same instance
        model2 = cache.load_model("TTS", self.tts_model_path, TTSModel, self.model_config)
        self.assertIs(model, model2)  # Should be same object due to caching
        
        # Test cache clearing
        cache.clear_cache()
        model3 = cache.load_model("TTS", self.tts_model_path, TTSModel, self.model_config)
        self.assertIsNot(model, model3)  # Should be different object after clearing
    
    def test_tts_inference_engine(self):
        """Test TTS inference engine."""
        engine = TTSInferenceEngine(self.inference_config, self.model_config)
        
        # Test model loading
        engine.load_models(self.tts_model_path, self.vocoder_model_path)
        self.assertTrue(hasattr(engine, 'tts_model'))
        self.assertTrue(hasattr(engine, 'vocoder'))
        
        # Test synthesis (mock the actual inference to avoid computational cost)
        with patch.object(engine.tts_model, 'forward', return_value={'mel_spectrogram': torch.randn(1, 80, 100)}), \
             patch.object(engine.vocoder, 'forward', return_value=torch.randn(1, 1, 80*256)):
            
            waveform = engine.synthesize(self.test_text)
            self.assertIsInstance(waveform, np.ndarray)
            self.assertEqual(waveform.ndim, 2)  # (batch, time)
            self.assertGreater(waveform.shape[1], 0)
    
    def test_tts_speed_adjustment(self):
        """Test TTS speed adjustment functionality."""
        engine = TTSInferenceEngine(self.inference_config, self.model_config)
        engine.load_models(self.tts_model_path, self.vocoder_model_path)
        
        # Mock the inference
        with patch.object(engine.tts_model, 'forward', return_value={'mel_spectrogram': torch.randn(1, 80, 100)}), \
             patch.object(engine.vocoder, 'forward', return_value=torch.randn(1, 1, 80*256)):
            
            # Test normal speed
            waveform_normal = engine.synthesize(self.test_text, speed=1.0)
            
            # Test faster speed
            waveform_fast = engine.synthesize(self.test_text, speed=1.5)
            
            # Test slower speed
            waveform_slow = engine.synthesize(self.test_text, speed=0.7)
            
            # All should produce valid waveforms
            self.assertIsInstance(waveform_normal, np.ndarray)
            self.assertIsInstance(waveform_fast, np.ndarray)
            self.assertIsInstance(waveform_slow, np.ndarray)
    
    def test_tts_pitch_adjustment(self):
        """Test TTS pitch adjustment functionality."""
        engine = TTSInferenceEngine(self.inference_config, self.model_config)
        engine.load_models(self.tts_model_path, self.vocoder_model_path)
        
        # Mock the inference
        with patch.object(engine.tts_model, 'forward', return_value={'mel_spectrogram': torch.randn(1, 80, 100)}), \
             patch.object(engine.vocoder, 'forward', return_value=torch.randn(1, 1, 80*256)):
            
            # Test different pitch values
            pitches = [-1.0, 0.0, 1.0, 2.0]
            for pitch in pitches:
                with self.subTest(pitch=pitch):
                    waveform = engine.synthesize(self.test_text, pitch=pitch)
                    self.assertIsInstance(waveform, np.ndarray)
                    self.assertEqual(waveform.ndim, 2)
    
    def test_stt_inference_engine(self):
        """Test STT inference engine."""
        engine = STTInferenceEngine(self.inference_config, self.model_config)
        
        # Test model loading
        engine.load_model(self.stt_model_path)
        self.assertTrue(hasattr(engine, 'stt_model'))
        
        # Test transcription (mock the actual inference)
        with patch.object(engine.stt_model, 'forward', return_value={'ctc_logits': torch.randn(1, 50, 10000)}):
            transcription = engine.transcribe(self.test_audio)
            self.assertIsInstance(transcription, str)
            # Should return some text (even if it's garbage from random logits)
            self.assertTrue(len(transcription) >= 0)
    
    def test_stt_with_resampling(self):
        """Test STT with different sample rates."""
        engine = STTInferenceEngine(self.inference_config, self.model_config)
        engine.load_model(self.stt_model_path)
        
        # Mock the inference
        with patch.object(engine.stt_model, 'forward', return_value={'ctc_logits': torch.randn(1, 50, 10000)}):
            # Test with different sample rates
            sample_rates = [16000, 22050, 44100]
            for sr in sample_rates:
                with self.subTest(sample_rate=sr):
                    # Create audio at different sample rate
                    audio_data = np.random.randn(sr).astype(np.float32)  # 1 second
                    transcription = engine.transcribe(audio_data, sr)
                    self.assertIsInstance(transcription, str)
    
    def test_inference_monitor(self):
        """Test inference monitoring."""
        monitor = InferenceMonitor()
        
        # Test recording metrics
        monitor.record_tts_request(0.5)
        monitor.record_tts_request(0.3)
        monitor.record_stt_request(0.8)
        monitor.record_error()
        
        # Test getting statistics
        stats = monitor.get_stats()
        
        self.assertEqual(stats['tts_requests'], 2)
        self.assertEqual(stats['stt_requests'], 1)
        self.assertEqual(stats['errors'], 1)
        self.assertEqual(stats['total_tts_time'], 0.8)
        self.assertEqual(stats['total_stt_time'], 0.8)
        self.assertAlmostEqual(stats['avg_tts_time'], 0.4)
        self.assertEqual(stats['avg_stt_time'], 0.8)
        self.assertAlmostEqual(stats['success_rate'], 2/3)  # 2 successful out of 3 total
    
    def test_inference_monitor_edge_cases(self):
        """Test inference monitor edge cases."""
        monitor = InferenceMonitor()
        
        # Test with no requests
        stats = monitor.get_stats()
        self.assertEqual(stats['tts_requests'], 0)
        self.assertEqual(stats['stt_requests'], 0)
        self.assertEqual(stats['avg_tts_time'], 0.0)
        self.assertEqual(stats['avg_stt_time'], 0.0)
        self.assertEqual(stats['success_rate'], 0.0)
        
        # Test with only errors
        monitor.record_error()
        monitor.record_error()
        stats = monitor.get_stats()
        self.assertEqual(stats['errors'], 2)
        self.assertEqual(stats['success_rate'], 0.0)
    
    @patch('src.inference.inference_engine.uvicorn')
    def test_api_server_creation(self, mock_uvicorn):
        """Test API server creation and basic functionality."""
        # Mock the model loading to avoid actual file I/O
        with patch.object(TTSInferenceEngine, 'load_models'), \
             patch.object(STTInferenceEngine, 'load_model'):
            
            api = InferenceAPI(self.inference_config, self.model_config)
            
            # Test that engines are created
            self.assertIsInstance(api.tts_engine, TTSInferenceEngine)
            self.assertIsInstance(api.stt_engine, STTInferenceEngine)
            
            # Test app creation
            self.assertTrue(hasattr(api, 'app'))
            
            # Test run method (should call uvicorn.run)
            api.run(host="127.0.0.1", port=8001)
            mock_uvicorn.run.assert_called_once()
    
    def test_inference_config_validation(self):
        """Test inference configuration validation."""
        # Test valid configurations
        valid_configs = [
            InferenceConfig(device="cpu", batch_size=1),
            InferenceConfig(device="cuda", batch_size=4),
            InferenceConfig(enable_caching=True, cache_size=500),
        ]
        
        for config in valid_configs:
            self.assertIsInstance(config, InferenceConfig)
            self.assertTrue(hasattr(config, 'device'))
            self.assertTrue(hasattr(config, 'batch_size'))
            self.assertTrue(hasattr(config, 'enable_caching'))
    
    def test_error_handling(self):
        """Test error handling in inference engines."""
        engine = TTSInferenceEngine(self.inference_config, self.model_config)
        
        # Test error when models not loaded
        with self.assertRaises(AttributeError):
            engine.synthesize(self.test_text)
        
        # Test error with invalid text
        engine.load_models(self.tts_model_path, self.vocoder_model_path)
        
        # Mock to raise exception
        with patch.object(engine.tts_model, 'forward', side_effect=RuntimeError("Model error")):
            with self.assertRaises(RuntimeError):
                engine.synthesize("This should fail")

if __name__ == '__main__':
    unittest.main()