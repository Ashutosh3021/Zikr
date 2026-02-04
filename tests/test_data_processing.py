"""
Test suite for data processing components.
Validates audio preprocessing, text normalization, and feature extraction functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.utils.config import DataConfig
from src.data.audio_preprocessing import AudioPreprocessor, AudioAugmentation
from src.data.text_preprocessing import TextNormalizer, Tokenizer, Phonemizer

class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DataConfig()
        self.test_audio = np.random.randn(22050).astype(np.float32)  # 1 second of random audio
        self.test_text = "Hello world! This is a test sentence with numbers 123 and abbreviations Dr. Mr."
    
    def test_audio_preprocessor_initialization(self):
        """Test AudioPreprocessor initialization."""
        preprocessor = AudioPreprocessor(self.config)
        self.assertEqual(preprocessor.sample_rate, self.config.sample_rate)
        self.assertEqual(preprocessor.n_mels, self.config.n_mels)
    
    def test_audio_preprocessing(self):
        """Test audio preprocessing pipeline."""
        preprocessor = AudioPreprocessor(self.config)
        
        # Test normalization
        normalized = preprocessor._normalize_audio(self.test_audio)
        self.assertTrue(np.max(np.abs(normalized)) <= 1.0)
        
        # Test pre-emphasis
        emphasized = preprocessor._pre_emphasis(self.test_audio)
        self.assertEqual(len(emphasized), len(self.test_audio))
        
        # Test feature extraction
        features = preprocessor.extract_features(self.test_audio)
        self.assertIn('mel_spectrogram', features)
        self.assertIn('mfcc', features)
        self.assertEqual(features['mel_spectrogram'].shape[0], self.config.n_mels)
    
    def test_audio_augmentation(self):
        """Test audio augmentation functionality."""
        augmenter = AudioAugmentation(self.config)
        
        # Test noise addition
        noisy = augmenter.add_noise(self.test_audio)
        self.assertNotEqual(np.sum(noisy), np.sum(self.test_audio))
        
        # Test volume perturbation
        louder = augmenter.volume_perturbation(self.test_audio, 1.5)
        self.assertGreater(np.sum(np.abs(louder)), np.sum(np.abs(self.test_audio)))
    
    def test_text_normalizer(self):
        """Test text normalization."""
        normalizer = TextNormalizer(self.config)
        
        # Test number normalization
        normalized = normalizer.normalize_text("I have 123 dollars")
        self.assertIn("one hundred twenty three", normalized)
        
        # Test abbreviation expansion
        normalized = normalizer.normalize_text("Dr. Smith is here")
        self.assertIn("doctor", normalized)
        
        # Test basic normalization
        normalized = normalizer.normalize_text(self.test_text)
        self.assertIsInstance(normalized, str)
        self.assertTrue(len(normalized) > 0)
    
    def test_tokenizer(self):
        """Test text tokenization."""
        tokenizer = Tokenizer(self.config)
        
        # Test tokenization
        tokens = tokenizer.tokenize(self.test_text)
        self.assertEqual(len(tokens), self.config.max_text_length)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        
        # Test detokenization
        text_back = tokenizer.detokenize(tokens)
        self.assertIsInstance(text_back, str)
        
        # Test vocabulary size
        vocab_size = tokenizer.get_vocabulary_size()
        self.assertGreater(vocab_size, 0)
    
    def test_phonemizer(self):
        """Test grapheme-to-phoneme conversion."""
        phonemizer = Phonemizer(self.config)
        
        # Test basic phonemization
        phonemes = phonemizer.phonemize("hello")
        self.assertIsInstance(phonemes, str)
        self.assertTrue(len(phonemes) > 0)
        
        # Test word exceptions
        phonemes = phonemizer.phonemize("the")
        self.assertEqual(phonemes, "dh ah")

if __name__ == '__main__':
    unittest.main()