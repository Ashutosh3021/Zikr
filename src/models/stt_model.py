"""
STT Transformer model components following the technical specification.
Implements audio encoder, attention mechanisms, and decoder for speech recognition.
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

class STTPositionalEncoding(nn.Module):
    """Positional encoding for STT transformer models."""
    
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

class STTMultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for STT."""
    
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
            if mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # Already in correct format (batch, heads, seq_len, seq_len)
                pass
            else:
                raise ValueError(f"Unsupported mask dimension: {mask.dim()}")
            
            # Ensure mask matches scores dimensions
            if mask.size(-1) != scores.size(-1) or mask.size(-2) != scores.size(-2):
                # Handle mismatch by expanding appropriately
                if mask.size(-1) == 1:  # Causal mask pattern
                    mask = mask.expand(-1, -1, scores.size(-2), scores.size(-1))
                else:
                    # Truncate or pad to match
                    target_len = scores.size(-1)
                    if mask.size(-1) > target_len:
                        mask = mask[..., :target_len]
                    elif mask.size(-1) < target_len:
                        pad_len = target_len - mask.size(-1)
                        mask = F.pad(mask, (0, pad_len), value=True)
            
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

class STTTransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer for STT."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = STTMultiHeadAttention(d_model, n_heads, dropout)
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

class STTAudioEncoder(nn.Module):
    """Audio encoder for STT system following specification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.stt_encoder_dim
        self.n_layers = config.stt_encoder_layers
        self.n_heads = config.stt_encoder_heads
        self.n_mels = config.data_config.n_mels if hasattr(config, 'data_config') else 80
        
        # Front-end CNN for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.n_mels, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Temporal subsampling (4x-8x as specified)
        self.subsample = nn.Conv1d(self.d_model, self.d_model, kernel_size=4, stride=2, padding=1)
        
        # Positional encoding
        self.positional_encoding = STTPositionalEncoding(self.d_model, max_len=2000)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            STTTransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=config.dropout_rate
            ) for _ in range(self.n_layers)
        ])
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"STTAudioEncoder initialized with {self.n_layers} layers, "
                   f"{self.n_heads} heads, d_model={self.d_model}")
    
    def forward(self, mel_spectrogram: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through audio encoder.
        
        Args:
            mel_spectrogram: Mel spectrogram tensor of shape (batch_size, n_mels, time)
            attention_mask: Optional attention mask
            
        Returns:
            Encoded audio representations of shape (batch_size, seq_len, d_model)
        """
        # Convert (batch, n_mels, time) to (batch, time, n_mels) for conv1d
        x = mel_spectrogram.transpose(1, 2)  # (batch, time, n_mels)
        
        # Apply CNN layers
        x = x.transpose(1, 2)  # (batch, n_mels, time) for conv1d
        x = self.conv_layers(x)
        x = self.subsample(x)
        
        # Convert back to (batch, time, features)
        x = x.transpose(1, 2)  # (batch, subsampled_time, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x

class CTCHead(nn.Module):
    """CTC head for alignment-free training."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CTC projection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            CTC logits of shape (batch_size, seq_len, vocab_size)
        """
        logits = self.linear(x)
        return self.log_softmax(logits)

class STTDecoderLayer(nn.Module):
    """Single transformer decoder layer for STT."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = STTMultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = STTMultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through decoder layer."""
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention with encoder output
        cross_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class STTDecoder(nn.Module):
    """STT decoder with attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.stt_encoder_dim
        self.n_layers = config.stt_decoder_layers
        self.n_heads = config.stt_decoder_heads
        self.vocab_size = config.stt_vocab_size
        
        # Embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = STTPositionalEncoding(self.d_model, max_len=1000)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            STTDecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=config.dropout_rate
            ) for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"STTDecoder initialized with {self.n_layers} layers, "
                   f"{self.n_heads} heads, vocab_size={self.vocab_size}")
    
    def forward(self, target_tokens: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            target_tokens: Target token indices (batch_size, tgt_seq_len)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: Target sequence mask
            memory_mask: Memory (encoder) mask
            
        Returns:
            Decoder output logits (batch_size, tgt_seq_len, vocab_size)
        """
        # Token embedding
        x = self.token_embedding(target_tokens) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits

class STTModel(nn.Module):
    """Complete STT model combining all components."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.audio_encoder = STTAudioEncoder(config)
        self.decoder = STTDecoder(config)
        self.ctc_head = CTCHead(config.stt_encoder_dim, config.stt_vocab_size)
        
        logger.info("STTModel initialized with all components")
    
    def forward(self, mel_spectrogram: torch.Tensor,
                target_tokens: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                decoder_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete STT pipeline.
        
        Args:
            mel_spectrogram: Input mel spectrogram (batch_size, n_mels, time)
            target_tokens: Target token indices (batch_size, seq_len) - for training
            encoder_mask: Encoder attention mask
            decoder_mask: Decoder attention mask
            
        Returns:
            Dictionary containing all outputs and predictions
        """
        # Audio encoding
        encoder_output = self.audio_encoder(mel_spectrogram, encoder_mask)
        
        outputs = {
            'encoder_output': encoder_output,
            'ctc_logits': self.ctc_head(encoder_output)
        }
        
        # Decoder forward pass (if targets provided for training)
        if target_tokens is not None:
            decoder_output = self.decoder(
                target_tokens, encoder_output, decoder_mask, encoder_mask
            )
            outputs['decoder_logits'] = decoder_output
        
        return outputs
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Utility functions
def create_stt_model(config: ModelConfig) -> STTModel:
    """Factory function to create STT model."""
    return STTModel(config)

def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """Generate causal mask for decoder self-attention."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask == 0  # True for allowed positions

def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Generate padding mask for sequences."""
    return (seq != pad_idx).unsqueeze(-2)  # (batch, 1, seq_len)

# Attention visualization utilities
def get_attention_weights(model: STTModel, mel_spectrogram: torch.Tensor, 
                         target_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract attention weights for visualization."""
    # This would require modifying the attention modules to return weights
    # Implementation would depend on specific visualization needs
    pass