"""
Audio preprocessing utilities for the TTS-STT system.
Implements the audio feature extraction pipeline as specified in the technical specification.
"""

import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional, Union
from pathlib import Path
import soundfile as sf

# Try to import librosa, but handle gracefully if not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, using fallback implementations")

from ..utils.logger import get_data_logger
from ..utils.config import DataConfig

logger = get_data_logger()

class AudioPreprocessor:
    """Audio preprocessing pipeline for TTS and STT systems."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_mels = config.n_mels
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        logger.info(f"AudioPreprocessor initialized with sample_rate={self.sample_rate}")
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file with error handling.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio using soundfile for better format support
            audio_data, sr = sf.read(str(file_path), dtype='float32')
            
            # Handle multi-channel audio
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                logger.debug(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
            
            logger.debug(f"Loaded audio: {len(audio_data)} samples, duration: {len(audio_data)/self.sample_rate:.2f}s")
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to raw audio data.
        
        Args:
            audio_data: Raw audio array
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Convert to float32 if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio amplitude
            if self.config.normalize_audio:
                audio_data = self._normalize_audio(audio_data)
            
            # Trim silence if configured
            if self.config.trim_silence:
                audio_data = self._trim_silence(audio_data)
            
            # Apply pre-emphasis filter
            audio_data = self._pre_emphasis(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 0:
            audio_data = audio_data / max_amp
        return audio_data
    
    def _trim_silence(self, audio_data: np.ndarray) -> np.ndarray:
        """Remove silence from beginning and end of audio."""
        try:
            # Use librosa's built-in trimming
            trimmed_audio, _ = librosa.effects.trim(
                audio_data, 
                top_db=20,  # threshold in dB below maximum peak
                frame_length=2048,
                hop_length=512
            )
            return trimmed_audio
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio_data  # Return original if trimming fails
    
    def _pre_emphasis(self, audio_data: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to emphasize high frequencies."""
        # Apply pre-emphasis: y[t] = x[t] - coef * x[t-1]
        emphasized = np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])
        return emphasized
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract Mel-frequency spectrogram features.
        
        Args:
            audio_data: Preprocessed audio array
            
        Returns:
            Mel spectrogram of shape (n_mels, time_frames)
        """
        try:
            # Ensure audio data is numpy array
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            
            # Use torchaudio for more reliable spectrogram computation
            if hasattr(torchaudio, 'transforms') and hasattr(torchaudio.transforms, 'MelSpectrogram'):
                # Use torchaudio implementation
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    power=2.0
                )
                
                # Convert to tensor if needed
                if not isinstance(audio_data, torch.Tensor):
                    audio_tensor = torch.from_numpy(audio_data).float()
                else:
                    audio_tensor = audio_data
                
                # Add batch dimension if needed
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Compute mel spectrogram
                mel_spec = mel_transform(audio_tensor)
                
                # Convert back to numpy and remove batch dimension
                mel_spec = mel_spec.squeeze(0).numpy()
                
                # Convert to log scale
                mel_spec = np.log(mel_spec + 1e-9)
                
                # Normalize
                mel_spec = self._normalize_mel_spectrogram(mel_spec)
                
                logger.debug(f"Extracted mel spectrogram using torchaudio: shape {mel_spec.shape}")
                return mel_spec
            
            # Fallback implementation using basic numpy operations
            else:
                # Simple spectrogram using FFT
                window_size = min(self.win_length, len(audio_data))
                hop_size = self.hop_length
                
                # Pad audio to ensure we can process it
                if len(audio_data) < window_size:
                    audio_data = np.pad(audio_data, (0, window_size - len(audio_data)))
                
                # Simple STFT implementation
                n_frames = max(1, (len(audio_data) - window_size) // hop_size + 1)
                spectrogram = np.zeros((window_size // 2 + 1, n_frames))
                
                window = np.hanning(window_size)
                
                for i in range(n_frames):
                    start = i * hop_size
                    end = start + window_size
                    if end <= len(audio_data):
                        frame = audio_data[start:end] * window
                        fft_result = np.fft.rfft(frame)
                        spectrogram[:, i] = np.abs(fft_result) ** 2
                
                # Convert to mel scale (simplified)
                n_mel_filters = self.n_mels
                mel_filters = self._create_mel_filterbank(window_size // 2 + 1, n_mel_filters)
                mel_spec = np.dot(mel_filters, spectrogram)
                
                # Convert to log scale
                mel_spec = np.log(mel_spec + 1e-9)
                
                # Normalize
                mel_spec = self._normalize_mel_spectrogram(mel_spec)
                
                logger.debug(f"Extracted mel spectrogram using fallback: shape {mel_spec.shape}")
                return mel_spec
            
        except Exception as e:
            logger.error(f"Mel spectrogram extraction failed: {e}")
            # Return default spectrogram on failure
            return np.zeros((self.n_mels, 10))
    
    def _create_mel_filterbank(self, n_fft: int, n_mels: int) -> np.ndarray:
        """Create mel filterbank matrix."""
        # Simplified mel filterbank creation
        mel_freqs = np.linspace(0, self.sample_rate // 2, n_mels + 2)
        hz_freqs = 700 * (10**(mel_freqs / 2595.0) - 1)  # Convert mel to Hz
        
        # Map Hz frequencies to FFT bins
        fft_freqs = np.linspace(0, self.sample_rate // 2, n_fft)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft))
        
        for i in range(1, n_mels + 1):
            left = hz_freqs[i-1]
            center = hz_freqs[i]
            right = hz_freqs[i+1]
            
            # Create triangular filter
            for j in range(n_fft):
                freq = fft_freqs[j]
                if left <= freq <= center:
                    filterbank[i-1, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i-1, j] = (right - freq) / (right - center)
        
        return filterbank
    
    def _normalize_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Normalize mel spectrogram using mean and standard deviation."""
        mean = np.mean(mel_spec)
        std = np.std(mel_spec)
        if std > 0:
            mel_spec = (mel_spec - mean) / std
        return mel_spec
    
    def extract_features(self, audio_data: np.ndarray) -> dict:
        """
        Extract all relevant audio features.
        
        Args:
            audio_data: Preprocessed audio array
            
        Returns:
            Dictionary containing various audio features
        """
        try:
            features = {}
            
            # Extract mel spectrogram (core feature)
            features['mel_spectrogram'] = self.extract_mel_spectrogram(audio_data)
            
            # Extract basic statistical features
            features['rms_energy'] = self._extract_rms_energy(audio_data)
            features['zero_crossing_rate'] = self._extract_zero_crossing_rate(audio_data)
            features['spectral_centroid'] = self._extract_spectral_centroid(audio_data)
            
            # Simple MFCC-like features
            features['mfcc'] = self._extract_simple_mfcc(audio_data)
            
            logger.debug(f"Extracted {len(features)} feature types")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return minimal feature set on complete failure
            return {
                'mel_spectrogram': np.zeros((self.n_mels, 10)),
                'mfcc': np.zeros((13, 10)),
                'spectral_centroid': np.zeros((1, 10)),
                'zero_crossing_rate': np.zeros((1, 10)),
                'rms_energy': np.zeros((1, 10))
            }
    
    def _extract_rms_energy(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract RMS energy features."""
        try:
            window_size = min(self.win_length, len(audio_data))
            hop_size = self.hop_length
            
            if len(audio_data) < window_size:
                return np.zeros((1, 1))
            
            n_frames = max(1, (len(audio_data) - window_size) // hop_size + 1)
            rms_energy = np.zeros((1, n_frames))
            
            for i in range(n_frames):
                start = i * hop_size
                end = start + window_size
                if end <= len(audio_data):
                    frame = audio_data[start:end]
                    rms_energy[0, i] = np.sqrt(np.mean(frame ** 2))
            
            return rms_energy
        except Exception:
            return np.zeros((1, 10))
    
    def _extract_zero_crossing_rate(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate features."""
        try:
            window_size = min(self.win_length, len(audio_data))
            hop_size = self.hop_length
            
            if len(audio_data) < window_size:
                return np.zeros((1, 1))
            
            n_frames = max(1, (len(audio_data) - window_size) // hop_size + 1)
            zcr = np.zeros((1, n_frames))
            
            for i in range(n_frames):
                start = i * hop_size
                end = start + window_size
                if end <= len(audio_data):
                    frame = audio_data[start:end]
                    zcr[0, i] = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
            
            return zcr
        except Exception:
            return np.zeros((1, 10))
    
    def _extract_spectral_centroid(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract spectral centroid features."""
        try:
            # Simple implementation using FFT
            window_size = min(self.win_length, len(audio_data))
            hop_size = self.hop_length
            
            if len(audio_data) < window_size:
                return np.zeros((1, 1))
            
            n_frames = max(1, (len(audio_data) - window_size) // hop_size + 1)
            centroid = np.zeros((1, n_frames))
            
            for i in range(n_frames):
                start = i * hop_size
                end = start + window_size
                if end <= len(audio_data):
                    frame = audio_data[start:end]
                    fft_result = np.fft.rfft(frame)
                    magnitude = np.abs(fft_result)
                    
                    if np.sum(magnitude) > 0:
                        freqs = np.linspace(0, self.sample_rate // 2, len(magnitude))
                        centroid[0, i] = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            return centroid
        except Exception:
            return np.zeros((1, 10))
    
    def _extract_simple_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract simple MFCC-like features."""
        try:
            # Use mel spectrogram and apply DCT to get MFCC-like features
            mel_spec = self.extract_mel_spectrogram(audio_data)
            
            # Simple DCT approximation using cosine transform
            n_mfcc = 13
            if mel_spec.shape[1] >= n_mfcc:
                # Apply DCT along time axis
                mfcc = np.zeros((n_mfcc, mel_spec.shape[1]))
                for i in range(n_mfcc):
                    for t in range(mel_spec.shape[1]):
                        mfcc[i, t] = np.sum(mel_spec[:, t] * np.cos(np.pi * i * np.arange(mel_spec.shape[0]) / mel_spec.shape[0]))
                return mfcc
            else:
                return np.zeros((n_mfcc, 10))
        except Exception:
            return np.zeros((13, 10))
    
    def audio_to_tensor(self, audio_data: np.ndarray) -> torch.Tensor:
        """Convert numpy audio array to PyTorch tensor."""
        return torch.from_numpy(audio_data).float()
    
    def tensor_to_audio(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy audio array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

class AudioAugmentation:
    """Audio data augmentation techniques for training robustness."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        logger.info("AudioAugmentation initialized")
    
    def add_noise(self, audio_data: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add random noise to audio."""
        noise = np.random.normal(0, noise_factor, audio_data.shape)
        return audio_data + noise
    
    def time_stretch(self, audio_data: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Apply time stretching."""
        return librosa.effects.time_stretch(audio_data, rate=rate)
    
    def pitch_shift(self, audio_data: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """Apply pitch shifting."""
        return librosa.effects.pitch_shift(
            audio_data, 
            sr=self.config.sample_rate, 
            n_steps=n_steps
        )
    
    def volume_perturbation(self, audio_data: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Apply volume perturbation."""
        return audio_data * factor
    
    def apply_augmentations(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply random augmentations to audio data."""
        # Apply augmentations with certain probabilities
        if np.random.random() < 0.3:  # 30% chance
            audio_data = self.add_noise(audio_data)
        
        if np.random.random() < 0.2:  # 20% chance
            rate = np.random.uniform(0.9, 1.1)
            audio_data = self.time_stretch(audio_data, rate)
        
        if np.random.random() < 0.2:  # 20% chance
            steps = np.random.uniform(-2, 2)
            audio_data = self.pitch_shift(audio_data, steps)
        
        if np.random.random() < 0.3:  # 30% chance
            factor = np.random.uniform(0.8, 1.2)
            audio_data = self.volume_perturbation(audio_data, factor)
        
        return audio_data

# Utility functions
def load_and_preprocess_audio(file_path: Union[str, Path], config: DataConfig) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to load and preprocess audio in one step.
    
    Args:
        file_path: Path to audio file
        config: Data configuration
        
    Returns:
        Tuple of (preprocessed_audio, extracted_features)
    """
    preprocessor = AudioPreprocessor(config)
    audio_data, _ = preprocessor.load_audio(file_path)
    preprocessed_audio = preprocessor.preprocess_audio(audio_data)
    features = preprocessor.extract_features(preprocessed_audio)
    
    return preprocessed_audio, features

def create_audio_preprocessor(config: DataConfig) -> AudioPreprocessor:
    """Factory function to create audio preprocessor."""
    return AudioPreprocessor(config)