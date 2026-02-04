"""
HiFi-GAN vocoder implementation for TTS audio synthesis.
Follows the technical specification for generative adversarial network approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
import numpy as np

from ..utils.logger import get_model_logger
from ..utils.config import ModelConfig

logger = get_model_logger()

class HiFiGANResBlock1(nn.Module):
    """Residual block with different kernel sizes - Type 1."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    1, 
                    dilation=dilation[i],
                    padding=self._get_padding(kernel_size, dilation[i])
                ),
                nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    1, 
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1)
                )
            ) for i in range(len(dilation))
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    1, 
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1)
                )
            ) for i in range(len(dilation))
        ])
    
    def _get_padding(self, kernel_size: int, dilation: int) -> int:
        """Calculate appropriate padding."""
        return (kernel_size * dilation - dilation) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = conv1(x)  # Apply first set of convolutions
            xt = conv2(xt)  # Apply second set of convolutions
            x = xt + x      # Residual connection
        return x

class HiFiGANResBlock2(nn.Module):
    """Residual block with different kernel sizes - Type 2."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    1, 
                    dilation=dilation[i],
                    padding=self._get_padding(kernel_size, dilation[i])
                )
            ) for i in range(len(dilation))
        ])
    
    def _get_padding(self, kernel_size: int, dilation: int) -> int:
        """Calculate appropriate padding."""
        return (kernel_size * dilation - dilation) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        for conv in self.convs:
            xt = conv(x)  # Apply convolution with different dilation
            x = xt + x     # Residual connection
        return x

class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator following the technical specification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_mels = config.data_config.n_mels if hasattr(config, 'data_config') else 80
        self.hop_size = config.data_config.hop_length if hasattr(config, 'data_config') else 256
        
        # Initial convolution from mel-spectrogram to hidden representation
        self.conv_pre = nn.Conv1d(self.n_mels, 512, 7, 1, padding=3)
        
        # Upsampling layers (should match hop_size factor)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, 8, padding=4),    # 8x upsampling
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),    # 8x upsampling
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),      # 2x upsampling
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1),       # 2x upsampling
        ])
        
        # Residual blocks after each upsampling
        self.resblocks = nn.ModuleList([
            HiFiGANResBlock1(256, 3, [1, 3, 5]),
            HiFiGANResBlock1(128, 3, [1, 3, 5]),
            HiFiGANResBlock2(64, 3, [1, 3, 5]),
            HiFiGANResBlock2(32, 3, [1, 3, 5]),
        ])
        
        # Final convolution to waveform
        self.conv_post = nn.Conv1d(32, 1, 7, 1, padding=3)
        
        # Weight normalization
        self._init_weights()
        logger.info(f"HiFiGANGenerator initialized with hop_size={self.hop_size}")
    
    def _init_weights(self):
        """Initialize weights with normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generate audio waveform from mel-spectrogram.
        
        Args:
            mel_spectrogram: Input mel-spectrogram of shape (batch_size, n_mels, time)
            
        Returns:
            Generated audio waveform of shape (batch_size, 1, audio_length)
        """
        # Initial transformation
        x = self.conv_pre(mel_spectrogram)
        
        # Apply upsampling with residual blocks
        for i, (up, resblock) in enumerate(zip(self.ups, self.resblocks)):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = resblock(x)
        
        # Final activation and output
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)  # Normalize to [-1, 1]
        
        return x

class HiFiGANPeriodDiscriminator(nn.Module):
    """Period-based discriminator for HiFi-GAN."""
    
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        
        # Feature extraction layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        
        # Final classification layer
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through period discriminator.
        
        Args:
            x: Audio waveform of shape (batch_size, 1, audio_length)
            
        Returns:
            Tuple of (discriminator_output, feature_maps)
        """
        # Reshape to (batch, 1, audio_length // period, period)
        batch_size, _, audio_length = x.shape
        if audio_length % self.period != 0:
            # Pad to make length divisible by period
            pad_len = self.period - (audio_length % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            audio_length = x.shape[2]
        
        x = x.view(batch_size, 1, audio_length // self.period, self.period)
        
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        return x, feature_maps

class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator with different periods."""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            HiFiGANPeriodDiscriminator(period) for period in periods
        ])
        logger.info(f"HiFiGANMultiPeriodDiscriminator initialized with periods: {periods}")
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through all period discriminators.
        
        Args:
            x: Audio waveform of shape (batch_size, 1, audio_length)
            
        Returns:
            Tuple of (discriminator_outputs, feature_maps_list)
        """
        outputs = []
        feature_maps_list = []
        
        for discriminator in self.discriminators:
            output, feature_maps = discriminator(x)
            outputs.append(output)
            feature_maps_list.append(feature_maps)
        
        return outputs, feature_maps_list

class HiFiGANScaleDiscriminator(nn.Module):
    """Scale-based discriminator for HiFi-GAN."""
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        
        # Feature extraction layers
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        
        # Final classification layer
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self.convs = nn.ModuleList([
                nn.utils.spectral_norm(conv) for conv in self.convs
            ])
            self.conv_post = nn.utils.spectral_norm(self.conv_post)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through scale discriminator.
        
        Args:
            x: Audio waveform of shape (batch_size, 1, audio_length)
            
        Returns:
            Tuple of (discriminator_output, feature_maps)
        """
        feature_maps = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        return x, feature_maps

class HiFiGANMultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator with different resolutions."""
    
    def __init__(self, scales: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.scales = scales
        
        self.discriminators = nn.ModuleList([
            HiFiGANScaleDiscriminator(use_spectral_norm) 
            for _ in range(scales)
        ])
        
        # Downsampling layers for multi-scale
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2) for _ in range(scales - 1)
        ])
        
        logger.info(f"HiFiGANMultiScaleDiscriminator initialized with {scales} scales")
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through all scale discriminators.
        
        Args:
            x: Audio waveform of shape (batch_size, 1, audio_length)
            
        Returns:
            Tuple of (discriminator_outputs, feature_maps_list)
        """
        outputs = []
        feature_maps_list = []
        
        # First scale (original resolution)
        output, feature_maps = self.discriminators[0](x)
        outputs.append(output)
        feature_maps_list.append(feature_maps)
        
        # Subsequent scales (downsampled)
        for i in range(1, self.scales):
            x = self.meanpools[i-1](x)
            output, feature_maps = self.discriminators[i](x)
            outputs.append(output)
            feature_maps_list.append(feature_maps)
        
        return outputs, feature_maps_list

class HiFiGANVocoder(nn.Module):
    """Complete HiFi-GAN vocoder system."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Generator
        self.generator = HiFiGANGenerator(config)
        
        # Discriminators
        self.mpd = HiFiGANMultiPeriodDiscriminator()
        self.msd = HiFiGANMultiScaleDiscriminator()
        
        logger.info("HiFiGANVocoder initialized with generator and discriminators")
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generate audio from mel-spectrogram.
        
        Args:
            mel_spectrogram: Input mel-spectrogram of shape (batch_size, n_mels, time)
            
        Returns:
            Generated audio waveform of shape (batch_size, 1, audio_length)
        """
        return self.generator(mel_spectrogram)
    
    def generator_forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through generator with feature extraction for training.
        
        Args:
            mel_spectrogram: Input mel-spectrogram
            
        Returns:
            Tuple of (generated_audio, features_dict)
        """
        generated_audio = self.generator(mel_spectrogram)
        
        # Get discriminator features for feature matching loss
        with torch.no_grad():
            _, mpd_features = self.mpd(generated_audio)
            _, msd_features = self.msd(generated_audio)
        
        features = {
            'mpd_features': mpd_features,
            'msd_features': msd_features
        }
        
        return generated_audio, features
    
    def discriminator_forward(self, real_audio: torch.Tensor, 
                            generated_audio: torch.Tensor) -> dict:
        """
        Forward pass through discriminators.
        
        Args:
            real_audio: Real audio samples
            generated_audio: Generated audio samples
            
        Returns:
            Dictionary containing discriminator outputs
        """
        # Multi-period discriminator
        real_mpd_outputs, real_mpd_features = self.mpd(real_audio)
        gen_mpd_outputs, gen_mpd_features = self.mpd(generated_audio)
        
        # Multi-scale discriminator
        real_msd_outputs, real_msd_features = self.msd(real_audio)
        gen_msd_outputs, gen_msd_features = self.msd(generated_audio)
        
        return {
            'real_mpd_outputs': real_mpd_outputs,
            'gen_mpd_outputs': gen_mpd_outputs,
            'real_msd_outputs': real_msd_outputs,
            'gen_msd_outputs': gen_msd_outputs,
            'real_mpd_features': real_mpd_features,
            'gen_mpd_features': gen_mpd_features,
            'real_msd_features': real_msd_features,
            'gen_msd_features': gen_msd_features
        }
    
    def get_model_size(self) -> dict:
        """Get parameter counts for generator and discriminators."""
        return {
            'generator': sum(p.numel() for p in self.generator.parameters() if p.requires_grad),
            'mpd': sum(p.numel() for p in self.mpd.parameters() if p.requires_grad),
            'msd': sum(p.numel() for p in self.msd.parameters() if p.requires_grad),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

# Utility functions
def create_hifigan_vocoder(config: ModelConfig) -> HiFiGANVocoder:
    """Factory function to create HiFi-GAN vocoder."""
    return HiFiGANVocoder(config)

def get_vocoder_config() -> dict:
    """Get default HiFi-GAN configuration."""
    return {
        'resblock_type': '1',  # or '2'
        'upsample_rates': [8, 8, 2, 2],
        'upsample_kernel_sizes': [16, 16, 4, 4],
        'upsample_initial_channel': 512,
        'resblock_kernel_sizes': [3, 7, 11],
        'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    }

# Audio processing utilities for vocoder
class AudioProcessor:
    """Audio processing utilities for HiFi-GAN."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.sample_rate = config.data_config.sample_rate
        self.hop_length = config.data_config.hop_length
    
    def audio_to_waveform(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Convert audio tensor to numpy waveform."""
        if isinstance(audio_tensor, torch.Tensor):
            waveform = audio_tensor.squeeze().cpu().numpy()
        else:
            waveform = audio_tensor
        return waveform
    
    def normalize_audio(self, audio: torch.Tensor, max_val: float = 0.95) -> torch.Tensor:
        """Normalize audio to prevent clipping."""
        max_amplitude = torch.max(torch.abs(audio))
        if max_amplitude > 0:
            audio = audio * (max_val / max_amplitude)
        return audio
    
    def calculate_audio_length(self, mel_length: int) -> int:
        """Calculate expected audio length from mel-spectrogram length."""
        return mel_length * self.hop_length