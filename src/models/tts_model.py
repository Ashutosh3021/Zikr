"""
TTS Transformer model components following the technical specification.
Implements the text encoder, variance adaptor, and mel-spectrogram generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np

from ..utils.logger import get_model_logger
from ..utils.config import ModelConfig

logger = get_model_logger()

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :self.d_model]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention computation."""
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask to match attention scores dimensions
            # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear transformation
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.out_linear(context)
        
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layer."""
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class TTSTextEncoder(nn.Module):
    """Text encoder for TTS system following specification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.tts_encoder_dim
        self.n_layers = config.tts_encoder_layers
        self.n_heads = config.tts_encoder_heads
        self.vocab_size = config.tts_vocab_size
        self.max_length = config.tts_max_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_length)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=config.dropout_rate
            ) for _ in range(self.n_layers)
        ])
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"TTSTextEncoder initialized with {self.n_layers} layers, "
                   f"{self.n_heads} heads, d_model={self.d_model}")
    
    def forward(self, tokens: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            tokens: Token indices tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Encoded text representations of shape (batch_size, seq_len, d_model)
        """
        # Token embedding
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x

class DurationPredictor(nn.Module):
    """Duration predictor for TTS system."""
    
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.linear = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict durations for each token.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Duration predictions of shape (batch_size, seq_len)
        """
        # Transpose for conv1d: (batch, channels, length)
        x = x.transpose(1, 2)
        
        # First convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer_norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.layer_norm2(x.transpose(1, 2))
        
        # Linear projection to scalar duration
        durations = self.linear(x).squeeze(-1)
        
        # Ensure positive durations
        durations = F.softplus(durations)
        
        return durations

class PitchPredictor(nn.Module):
    """Pitch predictor for TTS system."""
    
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.linear = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict pitch contours.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Pitch predictions of shape (batch_size, seq_len)
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer_norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.layer_norm2(x.transpose(1, 2))
        
        # Linear projection
        pitch = self.linear(x).squeeze(-1)
        
        return pitch

class EnergyPredictor(nn.Module):
    """Energy predictor for TTS system."""
    
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.linear = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict energy contours.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Energy predictions of shape (batch_size, seq_len)
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer_norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.layer_norm2(x.transpose(1, 2))
        
        # Linear projection
        energy = self.linear(x).squeeze(-1)
        
        return energy

class VarianceAdaptor(nn.Module):
    """Variance adaptor that combines duration, pitch, and energy predictions."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.tts_encoder_dim
        
        # Predictors
        self.duration_predictor = DurationPredictor(self.d_model, dropout=config.dropout_rate)
        self.pitch_predictor = PitchPredictor(self.d_model, dropout=config.dropout_rate)
        self.energy_predictor = EnergyPredictor(self.d_model, dropout=config.dropout_rate)
        
        # Pitch and energy embedding
        self.pitch_embedding = nn.Embedding(256, self.d_model)  # 256 discrete pitch levels
        self.energy_embedding = nn.Embedding(256, self.d_model)  # 256 discrete energy levels
        
        logger.info("VarianceAdaptor initialized")
    
    def forward(self, encoder_output: torch.Tensor, 
                target_durations: Optional[torch.Tensor] = None,
                target_pitch: Optional[torch.Tensor] = None,
                target_energy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply variance adaptation to encoder output.
        
        Args:
            encoder_output: Encoder output of shape (batch_size, seq_len, d_model)
            target_durations: Ground truth durations (for training)
            target_pitch: Ground truth pitch (for training)
            target_energy: Ground truth energy (for training)
            
        Returns:
            Dictionary containing adapted output and predictions
        """
        # Predict variances
        pred_durations = self.duration_predictor(encoder_output)
        pred_pitch = self.pitch_predictor(encoder_output)
        pred_energy = self.energy_predictor(encoder_output)
        
        # Use targets during training, predictions during inference
        if target_durations is not None:
            durations = target_durations
        else:
            durations = pred_durations
        
        if target_pitch is not None:
            pitch = target_pitch
        else:
            pitch = pred_pitch
            
        if target_energy is not None:
            energy = target_energy
        else:
            energy = pred_energy
        
        # Apply length regulation (expand based on durations)
        adapted_output = self._length_regulate(encoder_output, durations)
        
        # Apply pitch and energy embeddings
        # Resize pitch and energy to match adapted_output length
        if pitch.size(1) != adapted_output.size(1):
            # Interpolate or repeat to match length
            if pitch.size(1) < adapted_output.size(1):
                # Repeat or interpolate to longer length
                ratio = adapted_output.size(1) // pitch.size(1)
                pitch = pitch.repeat(1, ratio)[:,:adapted_output.size(1)]
                energy = energy.repeat(1, ratio)[:,:adapted_output.size(1)]
            else:
                # Truncate to match length
                pitch = pitch[:, :adapted_output.size(1)]
                energy = energy[:, :adapted_output.size(1)]
        
        pitch_embed = self._apply_variance_embedding(pitch, self.pitch_embedding)
        energy_embed = self._apply_variance_embedding(energy, self.energy_embedding)
        
        # Ensure embedding dimensions match
        if pitch_embed.size(1) != adapted_output.size(1):
            # Truncate or pad to match
            if pitch_embed.size(1) > adapted_output.size(1):
                pitch_embed = pitch_embed[:, :adapted_output.size(1), :]
                energy_embed = energy_embed[:, :adapted_output.size(1), :]
            else:
                # Pad with zeros
                pad_len = adapted_output.size(1) - pitch_embed.size(1)
                pitch_embed = F.pad(pitch_embed, (0, 0, 0, pad_len))
                energy_embed = F.pad(energy_embed, (0, 0, 0, pad_len))
        
        # Combine embeddings
        adapted_output = adapted_output + pitch_embed + energy_embed
        
        return {
            'adapted_output': adapted_output,
            'pred_durations': pred_durations,
            'pred_pitch': pred_pitch,
            'pred_energy': pred_energy,
            'target_durations': target_durations,
            'target_pitch': target_pitch,
            'target_energy': target_energy
        }
    
    def _length_regulate(self, x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Expand sequences based on duration predictions."""
        batch_size, seq_len, d_model = x.size()
        
        # Convert durations to integers and ensure minimum length of 1
        durations = torch.clamp(torch.round(durations).long(), min=1)
        
        # Calculate output length (use maximum across batch for consistent sizing)
        output_length = int(durations.sum(dim=1).max().item())
        # Ensure minimum reasonable length
        output_length = max(output_length, 10)
        
        # Create output tensor
        output = torch.zeros(batch_size, output_length, d_model, device=x.device)
        
        for i in range(batch_size):
            pos = 0
            for j in range(seq_len):
                duration = durations[i, j].item()
                if pos + duration <= output_length and j < x.size(1):
                    # Expand the j-th token for duration frames
                    end_pos = min(pos + duration, output_length)
                    output[i, pos:end_pos] = x[i, j]
                    pos = end_pos
        
        return output
    
    def _apply_variance_embedding(self, variance: torch.Tensor, embedding_layer: nn.Embedding) -> torch.Tensor:
        """Apply variance embedding to sequence."""
        # Discretize continuous variance values
        # Normalize to [0, 1] and convert to discrete bins
        variance_norm = torch.sigmoid(variance)  # Clamp to [0, 1]
        variance_discrete = (variance_norm * (embedding_layer.num_embeddings - 1)).long()
        
        # Apply embedding
        variance_embed = embedding_layer(variance_discrete)
        
        return variance_embed

class MelSpectrogramGenerator(nn.Module):
    """Mel-spectrogram generator (decoder) for TTS system."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.tts_encoder_dim
        self.n_mels = config.data_config.n_mels if hasattr(config, 'data_config') else 80
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=config.tts_encoder_heads,
                dropout=config.dropout_rate
            ) for _ in range(4)  # 4 decoder layers as per specification
        ])
        
        # Output projection to mel dimensions
        self.mel_projection = nn.Linear(self.d_model, self.n_mels)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"MelSpectrogramGenerator initialized with {self.n_mels} mel channels")
    
    def forward(self, adapted_input: torch.Tensor) -> torch.Tensor:
        """
        Generate mel-spectrogram from adapted input.
        
        Args:
            adapted_input: Adapted encoder output of shape (batch_size, mel_len, d_model)
            
        Returns:
            Mel-spectrogram of shape (batch_size, n_mels, mel_len)
        """
        # Apply transformer decoder layers
        x = adapted_input
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Project to mel dimensions
        mel_output = self.mel_projection(x)
        
        # Transpose to (batch, n_mels, time)
        mel_output = mel_output.transpose(1, 2)
        
        return mel_output

class TTSModel(nn.Module):
    """Complete TTS model combining all components."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.text_encoder = TTSTextEncoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.mel_generator = MelSpectrogramGenerator(config)
        
        logger.info("TTSModel initialized with all components")
    
    def forward(self, tokens: torch.Tensor,
                target_durations: Optional[torch.Tensor] = None,
                target_pitch: Optional[torch.Tensor] = None,
                target_energy: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete TTS pipeline.
        
        Args:
            tokens: Input token indices
            target_durations: Ground truth durations (training)
            target_pitch: Ground truth pitch (training)
            target_energy: Ground truth energy (training)
            attention_mask: Attention mask for encoder
            
        Returns:
            Dictionary containing all outputs and predictions
        """
        # Text encoding
        encoder_output = self.text_encoder(tokens, attention_mask)
        
        # Variance adaptation
        variance_output = self.variance_adaptor(
            encoder_output, target_durations, target_pitch, target_energy
        )
        
        # Mel-spectrogram generation
        mel_output = self.mel_generator(variance_output['adapted_output'])
        
        # Combine all outputs
        outputs = {
            'mel_spectrogram': mel_output,
            'encoder_output': encoder_output,
            'adapted_output': variance_output['adapted_output'],
            'pred_durations': variance_output['pred_durations'],
            'pred_pitch': variance_output['pred_pitch'],
            'pred_energy': variance_output['pred_energy']
        }
        
        # Include targets if provided
        if target_durations is not None:
            outputs['target_durations'] = target_durations
        if target_pitch is not None:
            outputs['target_pitch'] = target_pitch
        if target_energy is not None:
            outputs['target_energy'] = target_energy
            
        return outputs
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Utility functions
def create_tts_model(config: ModelConfig) -> TTSModel:
    """Factory function to create TTS model."""
    return TTSModel(config)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)