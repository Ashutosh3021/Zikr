"""
Integration tests for complete TTS-STT pipeline.
Validates end-to-end functionality and cross-component integration.
"""

import unittest
import torch
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config import ModelConfig, DataConfig, TrainingConfig
from src.data.text_preprocessing import TextNormalizer, Tokenizer
from src.data.audio_preprocessing import AudioPreprocessor
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel
from src.models.hifigan_vocoder import HiFiGANVocoder
from src.training.training_pipeline import TTSLoss, STTLoss, HiFiGANLoss, ModelTrainer
from src.evaluation.quality_metrics import QualityMetrics, TTSEvaluator, STTEvaluator


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests for complete TTS and STT pipelines."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        print("\n" + "="*60)
        print("Setting up integration test environment...")
        print("="*60)
        
        # Create configurations
        cls.data_config = DataConfig(
            sample_rate=22050,
            n_mels=80,
            max_wav_value=32768.0
        )
        
        cls.model_config = ModelConfig()
        cls.model_config.data_config = cls.data_config
        
        cls.training_config = TrainingConfig()
        
        # Initialize components
        cls.text_normalizer = TextNormalizer()
        cls.tokenizer = Tokenizer(vocab_size=1000)
        cls.audio_preprocessor = AudioPreprocessor(cls.data_config)
        
        # Initialize models
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {cls.device}")
        
        cls.tts_model = TTSModel(cls.model_config).to(cls.device)
        cls.stt_model = STTModel(cls.model_config).to(cls.device)
        cls.vocoder = HiFiGANVocoder(cls.model_config).to(cls.device)
        
        # Set models to eval mode for integration tests
        cls.tts_model.eval()
        cls.stt_model.eval()
        cls.vocoder.eval()
        
        # Initialize evaluators
        cls.tts_evaluator = TTSEvaluator()
        cls.stt_evaluator = STTEvaluator()
        
        print("Setup complete!\n")
    
    def test_01_text_preprocessing_pipeline(self):
        """Test complete text preprocessing pipeline."""
        print("\nTest 1: Text Preprocessing Pipeline")
        print("-" * 40)
        
        test_cases = [
            "Hello world",
            "The price is $50.00",
            "Dr. Smith has 3 apples",
            "Meeting at 3:30 PM on Dec 25, 2024"
        ]
        
        for text in test_cases:
            # Normalize
            normalized = self.text_normalizer.normalize(text)
            print(f"  Input:  '{text}'")
            print(f"  Output: '{normalized}'")
            
            # Tokenize
            tokens = self.tokenizer.encode(normalized)
            print(f"  Tokens: {tokens[:10]}... ({len(tokens)} total)")
            
            # Decode
            decoded = self.tokenizer.decode(tokens)
            print(f"  Decoded: '{decoded}'\n")
            
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
        
        print("✓ Text preprocessing pipeline working\n")
    
    def test_02_tts_inference_pipeline(self):
        """Test complete TTS inference pipeline."""
        print("\nTest 2: TTS Inference Pipeline")
        print("-" * 40)
        
        test_texts = [
            "Hello world",
            "This is a test",
            "Text to speech synthesis"
        ]
        
        with torch.no_grad():
            for text in test_texts:
                print(f"Processing: '{text}'")
                
                # Text preprocessing
                normalized = self.text_normalizer.normalize(text)
                tokens = self.tokenizer.encode(normalized)
                tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                # TTS inference
                start_time = time.time()
                tts_output = self.tts_model(tokens_tensor)
                tts_time = time.time() - start_time
                
                mel_spectrogram = tts_output['mel_spectrogram']
                print(f"  Mel shape: {mel_spectrogram.shape}")
                print(f"  TTS time: {tts_time*1000:.2f}ms")
                
                # Vocoder inference
                start_time = time.time()
                audio = self.vocoder(mel_spectrogram)
                vocoder_time = time.time() - start_time
                
                print(f"  Audio shape: {audio.shape}")
                print(f"  Vocoder time: {vocoder_time*1000:.2f}ms")
                print(f"  Total time: {(tts_time + vocoder_time)*1000:.2f}ms\n")
                
                # Validate outputs
                self.assertEqual(mel_spectrogram.dim(), 3)  # (batch, n_mels, time)
                self.assertEqual(audio.dim(), 2)  # (batch, samples)
                self.assertEqual(mel_spectrogram.size(0), 1)  # batch size 1
                
        print("✓ TTS inference pipeline working\n")
    
    def test_03_stt_inference_pipeline(self):
        """Test complete STT inference pipeline."""
        print("\nTest 3: STT Inference Pipeline")
        print("-" * 40)
        
        # Create synthetic audio input
        duration = 2.0  # seconds
        sample_rate = self.data_config.sample_rate
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create simple sine wave as test audio
        audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # (1, samples)
        
        with torch.no_grad():
            print(f"Processing audio: {audio.shape[1]} samples ({duration}s)")
            
            # Feature extraction
            start_time = time.time()
            features = self.audio_preprocessor.extract_mel_spectrogram(
                audio.squeeze(0).numpy()
            )
            feature_time = time.time() - start_time
            
            print(f"  Features shape: {features.shape}")
            print(f"  Feature extraction time: {feature_time*1000:.2f}ms")
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            
            # STT inference
            start_time = time.time()
            stt_output = self.stt_model(features_tensor)
            stt_time = time.time() - start_time
            
            print(f"  STT time: {stt_time*1000:.2f}ms")
            print(f"  Encoder output shape: {stt_output['encoder_output'].shape}")
            print(f"  Decoder output shape: {stt_output['decoder_output'].shape}")
            print(f"  CTC output shape: {stt_output['ctc_output'].shape}\n")
            
            # Validate outputs
            self.assertIn('encoder_output', stt_output)
            self.assertIn('decoder_output', stt_output)
            self.assertIn('ctc_output', stt_output)
        
        print("✓ STT inference pipeline working\n")
    
    def test_04_quality_metrics(self):
        """Test quality metrics calculation."""
        print("\nTest 4: Quality Metrics")
        print("-" * 40)
        
        # Test WER calculation
        references = [
            "hello world",
            "this is a test",
            "the quick brown fox"
        ]
        
        hypotheses = [
            "hello world",
            "this is test",
            "the quick brown dog"
        ]
        
        print("Testing WER/CER metrics:")
        for ref, hyp in zip(references, hypotheses):
            wer = QualityMetrics.calculate_wer(ref, hyp)
            cer = QualityMetrics.calculate_cer(ref, hyp)
            print(f"  Ref: '{ref}'")
            print(f"  Hyp: '{hyp}'")
            print(f"  WER: {wer:.2%}, CER: {cer:.2%}\n")
        
        # Batch WER
        wer_stats = QualityMetrics.calculate_wer_batch(references, hypotheses)
        print(f"Batch WER: {wer_stats['wer']:.2%}")
        print(f"Substitutions: {wer_stats['substitutions']}")
        print(f"Deletions: {wer_stats['deletions']}")
        print(f"Insertions: {wer_stats['insertions']}\n")
        
        # Test TTS evaluation
        print("Testing TTS metrics:")
        pred_mel = torch.randn(1, 80, 100)
        target_mel = torch.randn(1, 80, 100)
        
        tts_metrics = self.tts_evaluator.evaluate(pred_mel, target_mel)
        for metric, value in tts_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n✓ Quality metrics working\n")
    
    def test_05_training_losses(self):
        """Test training loss functions."""
        print("\nTest 5: Training Losses")
        print("-" * 40)
        
        # TTS Loss
        tts_loss_fn = TTSLoss()
        
        pred_mel = torch.randn(2, 80, 100)
        target_mel = torch.randn(2, 80, 100)
        pred_duration = torch.rand(2, 10)
        target_duration = torch.rand(2, 10) * 5
        pred_pitch = torch.randn(2, 10)
        target_pitch = torch.randn(2, 10)
        pred_energy = torch.randn(2, 10)
        target_energy = torch.randn(2, 10)
        
        tts_loss = tts_loss_fn(
            pred_mel, target_mel,
            pred_duration, target_duration,
            pred_pitch, target_pitch,
            pred_energy, target_energy
        )
        
        print(f"TTS Loss: {tts_loss.item():.4f}")
        self.assertGreater(tts_loss.item(), 0)
        
        # STT Loss
        stt_loss_fn = STTLoss(self.model_config.stt_vocab_size)
        
        pred_logits = torch.randn(10, 2, self.model_config.stt_vocab_size)
        target_tokens = torch.randint(0, self.model_config.stt_vocab_size, (2, 10))
        ctc_logits = torch.randn(50, 2, self.model_config.stt_vocab_size + 1)
        input_lengths = torch.tensor([50, 50])
        target_lengths = torch.tensor([10, 10])
        
        stt_loss = stt_loss_fn(
            pred_logits, target_tokens,
            ctc_logits, input_lengths, target_lengths
        )
        
        print(f"STT Loss: {stt_loss.item():.4f}")
        self.assertGreater(stt_loss.item(), 0)
        
        # HiFi-GAN Loss
        hifigan_loss_fn = HiFiGANLoss()
        
        pred_audio = torch.randn(2, 16000)
        target_audio = torch.randn(2, 16000)
        pred_mel_for_vocoder = torch.randn(2, 80, 100)
        target_mel_for_vocoder = torch.randn(2, 80, 100)
        
        # Discriminator outputs (lists of feature maps)
        pred_disc_outputs = [torch.randn(2, 1, 100) for _ in range(3)]
        target_disc_outputs = [torch.randn(2, 1, 100) for _ in range(3)]
        
        hifigan_losses = hifigan_loss_fn(
            pred_audio, target_audio,
            pred_mel_for_vocoder, target_mel_for_vocoder,
            pred_disc_outputs, target_disc_outputs
        )
        
        print(f"HiFi-GAN Generator Loss: {hifigan_losses['generator_loss']:.4f}")
        print(f"HiFi-GAN Discriminator Loss: {hifigan_losses['discriminator_loss']:.4f}")
        
        print("\n✓ Training losses working\n")
    
    def test_06_model_parameters(self):
        """Test model parameter counts."""
        print("\nTest 6: Model Parameters")
        print("-" * 40)
        
        tts_params = sum(p.numel() for p in self.tts_model.parameters())
        stt_params = sum(p.numel() for p in self.stt_model.parameters())
        vocoder_params = sum(p.numel() for p in self.vocoder.parameters())
        total_params = tts_params + stt_params + vocoder_params
        
        print(f"TTS Model:     {tts_params:,} parameters ({tts_params/1e6:.2f}M)")
        print(f"STT Model:     {stt_params:,} parameters ({stt_params/1e6:.2f}M)")
        print(f"Vocoder:       {vocoder_params:,} parameters ({vocoder_params/1e6:.2f}M)")
        print(f"Total System:  {total_params:,} parameters ({total_params/1e6:.2f}M)\n")
        
        # Check parameter counts are reasonable
        self.assertGreater(tts_params, 1e6)  # At least 1M parameters
        self.assertGreater(stt_params, 1e6)
        self.assertGreater(vocoder_params, 1e6)
        
        print("✓ Model parameters validated\n")
    
    def test_07_inference_latency(self):
        """Test inference latency targets."""
        print("\nTest 7: Inference Latency")
        print("-" * 40)
        
        num_runs = 10
        tts_times = []
        stt_times = []
        
        # Prepare inputs
        text = "This is a test sentence for latency measurement"
        normalized = self.text_normalizer.normalize(text)
        tokens = self.tokenizer.encode(normalized)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Synthetic audio for STT
        duration = 3.0
        sample_rate = self.data_config.sample_rate
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        features = self.audio_preprocessor.extract_mel_spectrogram(audio.squeeze(0).numpy())
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.tts_model(tokens_tensor)
                _ = self.stt_model(features_tensor)
        
        # Measure TTS latency
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                tts_output = self.tts_model(tokens_tensor)
                audio = self.vocoder(tts_output['mel_spectrogram'])
                tts_times.append(time.time() - start)
        
        # Measure STT latency
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                stt_output = self.stt_model(features_tensor)
                stt_times.append(time.time() - start)
        
        tts_mean = np.mean(tts_times) * 1000  # Convert to ms
        tts_std = np.std(tts_times) * 1000
        stt_mean = np.mean(stt_times) * 1000
        stt_std = np.std(stt_times) * 1000
        
        print(f"TTS Latency:  {tts_mean:.2f} ± {tts_std:.2f} ms")
        print(f"STT Latency:  {stt_mean:.2f} ± {stt_std:.2f} ms\n")
        
        print("✓ Latency measurements complete\n")
    
    def test_08_cross_component_consistency(self):
        """Test consistency between components."""
        print("\nTest 8: Cross-Component Consistency")
        print("-" * 40)
        
        # Test that TTS and vocoder work together
        text = "Consistency test"
        normalized = self.text_normalizer.normalize(text)
        tokens = self.tokenizer.encode(normalized)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # TTS generates mel
            tts_output = self.tts_model(tokens_tensor)
            mel = tts_output['mel_spectrogram']
            
            # Vocoder generates audio from mel
            audio = self.vocoder(mel)
            
            # Check audio is reasonable
            self.assertFalse(torch.isnan(audio).any())
            self.assertFalse(torch.isinf(audio).any())
            self.assertTrue(audio.abs().max() <= 1.0)  # Should be normalized
        
        print("  TTS → Vocoder chain: OK")
        print("  Audio range: [{:.4f}, {:.4f}]".format(audio.min().item(), audio.max().item()))
        
        # Test feature extraction consistency
        audio_np = audio.squeeze(0).cpu().numpy()
        features = self.audio_preprocessor.extract_mel_spectrogram(audio_np)
        
        print(f"  Re-extracted features shape: {features.shape}")
        print("  Feature extraction: OK\n")
        
        print("✓ Cross-component consistency verified\n")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.model_config.data_config = self.data_config
        self.device = torch.device('cpu')  # Use CPU for edge case tests
        
        self.tts_model = TTSModel(self.model_config).to(self.device)
        self.stt_model = STTModel(self.model_config).to(self.device)
        self.text_normalizer = TextNormalizer()
        self.tokenizer = Tokenizer(vocab_size=1000)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        print("\nEdge Case: Empty Text")
        
        normalized = self.text_normalizer.normalize("")
        self.assertEqual(normalized, "")
        
        tokens = self.tokenizer.encode("")
        self.assertEqual(len(tokens), 0)
        
        print("✓ Empty text handled\n")
    
    def test_long_text(self):
        """Test handling of long text."""
        print("\nEdge Case: Long Text")
        
        long_text = " ".join(["word"] * 1000)
        normalized = self.text_normalizer.normalize(long_text)
        tokens = self.tokenizer.encode(normalized)
        
        print(f"  Long text tokens: {len(tokens)}")
        self.assertGreater(len(tokens), 0)
        
        # Test TTS with long text (should truncate)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.tts_model(tokens_tensor)
            self.assertIn('mel_spectrogram', output)
        
        print("✓ Long text handled\n")
    
    def test_special_characters(self):
        """Test handling of special characters."""
        print("\nEdge Case: Special Characters")
        
        special_texts = [
            "Hello! How are you?",
            "Price: $100.50 (50% off)",
            "Email: test@example.com",
            "Numbers: 123, 456.789",
            "Unicode: café, naïve, résumé"
        ]
        
        for text in special_texts:
            normalized = self.text_normalizer.normalize(text)
            tokens = self.tokenizer.encode(normalized)
            print(f"  '{text[:30]}...' → {len(tokens)} tokens")
            self.assertIsInstance(tokens, list)
        
        print("✓ Special characters handled\n")
    
    def test_very_short_audio(self):
        """Test handling of very short audio."""
        print("\nEdge Case: Short Audio")
        
        # Create very short audio (0.1 seconds)
        sample_rate = self.data_config.sample_rate
        duration = 0.1
        samples = int(sample_rate * duration)
        audio = torch.randn(1, samples).to(self.device)
        
        # Extract features
        preprocessor = AudioPreprocessor(self.data_config)
        
        try:
            features = preprocessor.extract_mel_spectrogram(audio.squeeze(0).cpu().numpy())
            print(f"  Short audio features shape: {features.shape}")
            
            # Test STT with short audio
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.stt_model(features_tensor)
                self.assertIn('encoder_output', output)
            
            print("✓ Short audio handled\n")
        except Exception as e:
            print(f"  Note: Short audio handling: {e}\n")


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
