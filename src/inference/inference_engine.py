"""
Inference engine for TTS-STT system.
Implements real-time inference components and API endpoints as specified.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import time
import asyncio
from pathlib import Path
import json
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from ..utils.logger import get_api_logger
from ..utils.config import ModelConfig, DataConfig
from ..models.tts_model import TTSModel
from ..models.stt_model import STTModel
from ..models.hifigan_vocoder import HiFiGANVocoder, AudioProcessor
from ..data.text_preprocessing import TextNormalizer, Tokenizer, Phonemizer
from ..data.audio_preprocessing import AudioPreprocessor

logger = get_api_logger()

@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_input_length: int = 200
    enable_caching: bool = True
    cache_size: int = 1000
    timeout_seconds: int = 30

class ModelCache:
    """Cache for loaded models to avoid reloading."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.models = {}
        self.device = torch.device(config.device)
        logger.info(f"ModelCache initialized on device: {self.device}")
    
    def load_model(self, model_type: str, model_path: str, 
                   model_class: type, config: ModelConfig) -> nn.Module:
        """Load model with caching."""
        cache_key = f"{model_type}_{model_path}"
        
        if cache_key in self.models and self.config.enable_caching:
            logger.info(f"Loading {model_type} from cache")
            return self.models[cache_key]
        
        logger.info(f"Loading {model_type} from {model_path}")
        try:
            model = model_class(config)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            if self.config.enable_caching:
                self.models[cache_key] = model
            
            logger.info(f"{model_type} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            raise
    
    def clear_cache(self):
        """Clear model cache."""
        self.models.clear()
        logger.info("Model cache cleared")

class TTSInferenceEngine:
    """Text-to-Speech inference engine."""
    
    def __init__(self, config: InferenceConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(config.device)
        
        # Model cache
        self.model_cache = ModelCache(config)
        
        # Preprocessing components
        self.text_normalizer = TextNormalizer(model_config.data_config)
        self.tokenizer = Tokenizer(model_config.data_config)
        self.phonemizer = Phonemizer(model_config.data_config)
        self.audio_processor = AudioProcessor(model_config)
        
        logger.info("TTSInferenceEngine initialized")
    
    def load_models(self, tts_model_path: str, vocoder_model_path: str):
        """Load TTS and vocoder models."""
        from ..models.tts_model import TTSModel
        from ..models.hifigan_vocoder import HiFiGANVocoder
        
        self.tts_model = self.model_cache.load_model(
            "TTS", tts_model_path, TTSModel, self.model_config
        )
        self.vocoder = self.model_cache.load_model(
            "HiFiGAN", vocoder_model_path, HiFiGANVocoder, self.model_config
        )
    
    def synthesize(self, text: str, speed: float = 1.0, pitch: float = 0.0) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            speed: Speaking speed factor (0.5-2.0)
            pitch: Pitch adjustment (-2.0 to 2.0 semitones)
            
        Returns:
            Audio waveform as numpy array
        """
        start_time = time.time()
        
        try:
            # Text preprocessing
            normalized_text = self.text_normalizer.normalize_text(text)
            tokens = self.tokenizer.tokenize(normalized_text)
            phonemes = self.phonemizer.phonemize(text)
            
            # Convert to tensors
            tokens_tensor = torch.tensor([tokens], device=self.device)
            
            # TTS model inference
            with torch.no_grad():
                tts_outputs = self.tts_model(tokens_tensor)
                mel_spectrogram = tts_outputs['mel_spectrogram']
            
            # Apply speed and pitch modifications
            if speed != 1.0:
                mel_spectrogram = self._adjust_speed(mel_spectrogram, speed)
            
            if pitch != 0.0:
                mel_spectrogram = self._adjust_pitch(mel_spectrogram, pitch)
            
            # Vocoder inference
            with torch.no_grad():
                audio_waveform = self.vocoder(mel_spectrogram)
            
            # Post-processing
            audio_waveform = self.audio_processor.normalize_audio(audio_waveform)
            waveform = self.audio_processor.audio_to_waveform(audio_waveform)
            
            inference_time = time.time() - start_time
            logger.info(f"TTS inference completed in {inference_time:.2f}s")
            
            return waveform
            
        except Exception as e:
            logger.error(f"TTS inference failed: {e}")
            raise
    
    def _adjust_speed(self, mel_spectrogram: torch.Tensor, speed: float) -> torch.Tensor:
        """Adjust speaking speed by modifying mel-spectrogram."""
        # Simple time-stretching by interpolation
        batch_size, n_mels, time_steps = mel_spectrogram.shape
        new_time_steps = int(time_steps / speed)
        
        # Create new time indices
        old_indices = torch.linspace(0, time_steps - 1, time_steps, device=mel_spectrogram.device)
        new_indices = torch.linspace(0, time_steps - 1, new_time_steps, device=mel_spectrogram.device)
        
        # Interpolate
        mel_stretched = torch.zeros(batch_size, n_mels, new_time_steps, device=mel_spectrogram.device)
        for i in range(n_mels):
            mel_stretched[:, i, :] = torch.nn.functional.interpolate(
                mel_spectrogram[:, i:i+1, :], 
                size=new_time_steps, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        return mel_stretched
    
    def _adjust_pitch(self, mel_spectrogram: torch.Tensor, pitch: float) -> torch.Tensor:
        """Adjust pitch by shifting mel frequencies."""
        # Simple pitch shifting by adding offset to mel values
        # This is a simplified approach - real pitch shifting is more complex
        pitch_factor = pitch * 0.1  # Scale factor for pitch adjustment
        return mel_spectrogram + pitch_factor

class STTInferenceEngine:
    """Speech-to-Text inference engine."""
    
    def __init__(self, config: InferenceConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(config.device)
        
        # Model cache
        self.model_cache = ModelCache(config)
        
        # Preprocessing components
        self.audio_preprocessor = AudioPreprocessor(model_config.data_config)
        self.tokenizer = Tokenizer(model_config.data_config)
        
        logger.info("STTInferenceEngine initialized")
    
    def load_model(self, model_path: str):
        """Load STT model."""
        self.stt_model = self.model_cache.load_model(
            "STT", model_path, STTModel, self.model_config
        )
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 22050) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio waveform data
            sample_rate: Sample rate of audio data
            
        Returns:
            Transcribed text
        """
        start_time = time.time()
        
        try:
            # Audio preprocessing
            if sample_rate != self.model_config.data_config.sample_rate:
                # Resample if needed
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.model_config.data_config.sample_rate
                )
            
            # Extract features
            features = self.audio_preprocessor.extract_features(audio_data)
            mel_spectrogram = features['mel_spectrogram']
            
            # Convert to tensor
            mel_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).to(self.device)
            
            # STT model inference
            with torch.no_grad():
                outputs = self.stt_model(mel_tensor)
                ctc_logits = outputs['ctc_logits']
            
            # Decode CTC output
            transcription = self._decode_ctc(ctc_logits)
            
            inference_time = time.time() - start_time
            logger.info(f"STT inference completed in {inference_time:.2f}s")
            
            return transcription
            
        except Exception as e:
            logger.error(f"STT inference failed: {e}")
            raise
    
    def _decode_ctc(self, ctc_logits: torch.Tensor) -> str:
        """Decode CTC logits to text."""
        # Simple greedy decoding
        # In practice, you'd want to use beam search or other advanced decoding
        predicted_ids = torch.argmax(ctc_logits, dim=-1)
        
        # Remove consecutive duplicates and blank tokens (assuming 0 is blank)
        batch_size, seq_len = predicted_ids.shape
        decoded_texts = []
        
        for i in range(batch_size):
            # Remove consecutive duplicates
            unique_ids = [predicted_ids[i, 0].item()]
            for j in range(1, seq_len):
                if predicted_ids[i, j] != predicted_ids[i, j-1]:
                    unique_ids.append(predicted_ids[i, j].item())
            
            # Remove blank tokens (0) and convert to text
            token_ids = [id for id in unique_ids if id != 0]
            text = self.tokenizer.detokenize(token_ids)
            decoded_texts.append(text)
        
        return decoded_texts[0] if decoded_texts else ""

class InferenceAPI:
    """FastAPI application for inference endpoints."""
    
    def __init__(self, config: InferenceConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.app = FastAPI(title="TTS-STT Inference API", version="1.0.0")
        
        # Initialize engines
        self.tts_engine = TTSInferenceEngine(config, model_config)
        self.stt_engine = STTInferenceEngine(config, model_config)
        
        # Setup routes
        self._setup_routes()
        
        logger.info("InferenceAPI initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "device": self.config.device}
        
        @self.app.post("/tts")
        async def text_to_speech(
            text: str = Form(...),
            speed: float = Form(1.0),
            pitch: float = Form(0.0)
        ):
            """Text-to-speech endpoint."""
            try:
                # Validate parameters
                if not text.strip():
                    raise HTTPException(status_code=400, detail="Text cannot be empty")
                
                if not 0.5 <= speed <= 2.0:
                    raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")
                
                if not -2.0 <= pitch <= 2.0:
                    raise HTTPException(status_code=400, detail="Pitch must be between -2.0 and 2.0")
                
                # Generate audio
                waveform = self.tts_engine.synthesize(text, speed, pitch)
                
                # Convert to bytes for streaming
                audio_bytes = waveform.tobytes()
                
                return StreamingResponse(
                    iter([audio_bytes]),
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=speech.wav"}
                )
                
            except Exception as e:
                logger.error(f"TTS API error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/stt")
        async def speech_to_text(audio: UploadFile = File(...)):
            """Speech-to-text endpoint."""
            try:
                # Validate file
                if not audio.content_type.startswith("audio/"):
                    raise HTTPException(status_code=400, detail="File must be audio")
                
                # Read audio data
                audio_bytes = await audio.read()
                
                # Convert to numpy array (simplified - in practice use proper audio loading)
                import soundfile as sf
                import io
                
                audio_buffer = io.BytesIO(audio_bytes)
                audio_data, sample_rate = sf.read(audio_buffer)
                
                # Transcribe
                transcription = self.stt_engine.transcribe(audio_data, sample_rate)
                
                return {"text": transcription, "confidence": 0.95}  # Confidence is placeholder
                
            except Exception as e:
                logger.error(f"STT API error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/load-models")
        async def load_models(
            tts_model_path: str = Form(...),
            stt_model_path: str = Form(...),
            vocoder_model_path: str = Form(...)
        ):
            """Load models endpoint."""
            try:
                self.tts_engine.load_models(tts_model_path, vocoder_model_path)
                self.stt_engine.load_model(stt_model_path)
                return {"status": "success", "message": "Models loaded successfully"}
                
            except Exception as e:
                logger.error(f"Model loading error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Utility functions
def create_inference_engine(config: InferenceConfig, model_config: ModelConfig) -> Tuple[TTSInferenceEngine, STTInferenceEngine]:
    """Factory function to create inference engines."""
    tts_engine = TTSInferenceEngine(config, model_config)
    stt_engine = STTInferenceEngine(config, model_config)
    return tts_engine, stt_engine

def create_api_server(config: InferenceConfig, model_config: ModelConfig) -> InferenceAPI:
    """Factory function to create API server."""
    return InferenceAPI(config, model_config)

# Performance monitoring
class InferenceMonitor:
    """Monitor inference performance and metrics."""
    
    def __init__(self):
        self.metrics = {
            'tts_requests': 0,
            'stt_requests': 0,
            'total_tts_time': 0.0,
            'total_stt_time': 0.0,
            'errors': 0
        }
    
    def record_tts_request(self, inference_time: float):
        """Record TTS inference metrics."""
        self.metrics['tts_requests'] += 1
        self.metrics['total_tts_time'] += inference_time
    
    def record_stt_request(self, inference_time: float):
        """Record STT inference metrics."""
        self.metrics['stt_requests'] += 1
        self.metrics['total_stt_time'] += inference_time
    
    def record_error(self):
        """Record inference error."""
        self.metrics['errors'] += 1
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get inference statistics."""
        stats = self.metrics.copy()
        
        if stats['tts_requests'] > 0:
            stats['avg_tts_time'] = stats['total_tts_time'] / stats['tts_requests']
        else:
            stats['avg_tts_time'] = 0.0
            
        if stats['stt_requests'] > 0:
            stats['avg_stt_time'] = stats['total_stt_time'] / stats['stt_requests']
        else:
            stats['avg_stt_time'] = 0.0
        
        total_requests = stats['tts_requests'] + stats['stt_requests']
        stats['success_rate'] = (total_requests - stats['errors']) / total_requests if total_requests > 0 else 0.0
        
        return stats