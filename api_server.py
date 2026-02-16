#!/usr/bin/env python3
"""
API Server for TTS-STT system using FastAPI.
Provides REST endpoints for TTS and STT inference.

Usage:
  python api_server.py --port 8000
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import io
import base64
from typing import Optional, List
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.utils.config import ModelConfig, DataConfig
from src.utils.logger import setup_logger, get_logger
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel
from src.models.hifigan_vocoder import HiFiGANVocoder
from src.data.text_preprocessing import TextNormalizer, Tokenizer
from src.data.audio_preprocessing import AudioPreprocessor


# Setup logging
setup_logger('INFO')
logger = get_logger()

# Create FastAPI app
app = FastAPI(
    title="TTS-STT API",
    description="Text-to-Speech and Speech-to-Text API Server",
    version="1.0.0"
)

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", example="Hello world")
    voice: Optional[str] = Field("default", description="Voice ID to use")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")
    pitch: Optional[float] = Field(0.0, ge=-1.0, le=1.0, description="Pitch adjustment (-1.0 to 1.0)")
    format: Optional[str] = Field("wav", description="Output audio format (wav, mp3)")


class TTSResponse(BaseModel):
    success: bool
    audio: str = Field(..., description="Base64 encoded audio data")
    format: str
    duration: float
    sample_rate: int


class STTResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    cuda_available: bool
    version: str


class BatchTTSRequest(BaseModel):
    texts: List[str]
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    pitch: Optional[float] = 0.0


# Global variables for models
device = None
tts_model = None
stt_model = None
vocoder = None
text_normalizer = None
tokenizer = None
audio_preprocessor = None
model_config = None
data_config = None


def load_models():
    """Load all models on startup."""
    global device, tts_model, stt_model, vocoder
    global text_normalizer, tokenizer, audio_preprocessor
    global model_config, data_config
    
    logger.info("Loading models...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configs
    data_config = DataConfig()
    model_config = ModelConfig()
    model_config.data_config = data_config
    
    # Initialize preprocessors
    text_normalizer = TextNormalizer()
    tokenizer = Tokenizer(vocab_size=model_config.tts_vocab_size)
    audio_preprocessor = AudioPreprocessor(data_config)
    
    # Load TTS model
    logger.info("Loading TTS model...")
    tts_model = TTSModel(model_config).to(device)
    tts_model.eval()
    
    # Load vocoder
    logger.info("Loading HiFi-GAN vocoder...")
    vocoder = HiFiGANVocoder(model_config).to(device)
    vocoder.eval()
    
    # Load STT model
    logger.info("Loading STT model...")
    stt_model = STTModel(model_config).to(device)
    stt.eval()
    
    logger.info("All models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    load_models()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(
        status="running",
        models_loaded=tts_model is not None,
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if tts_model is not None else "loading",
        models_loaded=tts_model is not None,
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        version="1.0.0"
    )


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech.
    
    - **text**: Text to synthesize
    - **voice**: Voice ID (default: "default")
    - **speed**: Speech speed multiplier (0.5-2.0)
    - **pitch**: Pitch adjustment (-1.0 to 1.0)
    - **format**: Output format (wav, mp3)
    """
    try:
        logger.info(f"TTS request: '{request.text[:50]}...' " if len(request.text) > 50 else f"TTS request: '{request.text}'")
        
        # Preprocess text
        normalized = text_normalizer.normalize(request.text)
        tokens = tokenizer.encode(normalized)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            tts_output = tts_model(tokens_tensor)
            mel_spectrogram = tts_output['mel_spectrogram']
            
            # Generate audio
            audio = vocoder(mel_spectrogram)
            
            # Apply speed adjustment
            if request.speed != 1.0:
                import torch.nn.functional as F
                original_length = audio.shape[-1]
                new_length = int(original_length / request.speed)
                audio = F.interpolate(
                    audio.unsqueeze(1),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
        
        # Convert to numpy
        audio_np = audio.squeeze(0).cpu().numpy()
        audio_np = audio_np / np.abs(audio_np).max() * 0.95
        
        # Calculate duration
        duration = len(audio_np) / data_config.sample_rate
        
        # Encode to base64
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, data_config.sample_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        logger.info(f"TTS complete: {duration:.2f}s audio generated")
        
        return TTSResponse(
            success=True,
            audio=audio_base64,
            format=request.format,
            duration=duration,
            sample_rate=data_config.sample_rate
        )
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """
    Stream text-to-speech audio.
    
    Returns audio as streaming response.
    """
    try:
        # Preprocess text
        normalized = text_normalizer.normalize(request.text)
        tokens = tokenizer.encode(normalized)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Generate audio
        with torch.no_grad():
            tts_output = tts_model(tokens_tensor)
            audio = vocoder(tts_output['mel_spectrogram'])
            
            if request.speed != 1.0:
                import torch.nn.functional as F
                original_length = audio.shape[-1]
                new_length = int(original_length / request.speed)
                audio = F.interpolate(
                    audio.unsqueeze(1),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
        
        audio_np = audio.squeeze(0).cpu().numpy()
        audio_np = audio_np / np.abs(audio_np).max() * 0.95
        
        # Stream audio
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, data_config.sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech.wav"
            }
        )
    
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """
    Convert speech to text.
    
    - **file**: Audio file (WAV, MP3, FLAC)
    
    Returns transcribed text with confidence score.
    """
    try:
        logger.info(f"STT request: {file.filename}")
        
        import time
        start_time = time.time()
        
        # Read audio file
        contents = await file.read()
        audio, sr = sf.read(io.BytesIO(contents))
        
        # Resample if needed
        if sr != data_config.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=data_config.sample_rate)
        
        # Extract features
        features = audio_preprocessor.extract_mel_spectrogram(audio)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        
        # Run STT model
        with torch.no_grad():
            output = stt_model(features_tensor)
            ctc_output = output['ctc_output']
            
            # Simple greedy decoding
            predictions = ctc_output.argmax(dim=-1)
            
            # Remove duplicates and blanks
            transcription = []
            prev_token = None
            blank_id = model_config.stt_vocab_size
            
            for token in predictions[0].cpu().numpy():
                if token != prev_token and token != blank_id:
                    transcription.append(token)
                prev_token = token
            
            # Convert to text (placeholder)
            text = " ".join([f"tok_{t}" for t in transcription])
            
            # Calculate confidence (placeholder)
            confidence = 0.95
        
        processing_time = time.time() - start_time
        
        logger.info(f"STT complete: '{text}' (confidence: {confidence:.2f})")
        
        return STTResponse(
            success=True,
            text=text,
            confidence=confidence,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/batch")
async def batch_text_to_speech(request: BatchTTSRequest):
    """
    Batch text-to-speech processing.
    
    Process multiple texts at once.
    """
    results = []
    
    for text in request.texts:
        try:
            # Preprocess
            normalized = text_normalizer.normalize(text)
            tokens = tokenizer.encode(normalized)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Generate
            with torch.no_grad():
                tts_output = tts_model(tokens_tensor)
                audio = vocoder(tts_output['mel_spectrogram'])
            
            audio_np = audio.squeeze(0).cpu().numpy()
            audio_np = audio_np / np.abs(audio_np).max() * 0.95
            
            # Encode
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, data_config.sample_rate, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            results.append({
                "text": text,
                "success": True,
                "audio": audio_base64,
                "duration": len(audio_np) / data_config.sample_rate
            })
        
        except Exception as e:
            results.append({
                "text": text,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    tts_params = sum(p.numel() for p in tts_model.parameters()) if tts_model else 0
    stt_params = sum(p.numel() for p in stt_model.parameters()) if stt_model else 0
    vocoder_params = sum(p.numel() for p in vocoder.parameters()) if vocoder else 0
    
    return {
        "tts_model": {
            "parameters": tts_params,
            "parameters_millions": round(tts_params / 1e6, 2),
            "encoder_layers": model_config.tts_encoder_layers if model_config else None,
            "encoder_dim": model_config.tts_encoder_dim if model_config else None,
        },
        "stt_model": {
            "parameters": stt_params,
            "parameters_millions": round(stt_params / 1e6, 2),
            "encoder_layers": model_config.stt_encoder_layers if model_config else None,
            "vocab_size": model_config.stt_vocab_size if model_config else None,
        },
        "vocoder": {
            "parameters": vocoder_params,
            "parameters_millions": round(vocoder_params / 1e6, 2),
        },
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }


def main():
    """Run API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TTS-STT API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload (development only)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting TTS-STT API Server")
    logger.info("="*60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"API Documentation: http://{args.host}:{args.port}/docs")
    logger.info("="*60)
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == '__main__':
    main()
