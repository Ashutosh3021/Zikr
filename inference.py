#!/usr/bin/env python3
"""
Inference script for TTS-STT system.
Usage: 
  TTS: python inference.py --mode tts --text "Hello world" --output output.wav
  STT: python inference.py --mode stt --input audio.wav
"""

import argparse
import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig, DataConfig
from src.utils.logger import setup_logger, get_logger
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel
from src.models.hifigan_vocoder import HiFiGANVocoder
from src.data.text_preprocessing import TextNormalizer, Tokenizer
from src.data.audio_preprocessing import AudioPreprocessor
from src.inference.inference_engine import InferenceEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run TTS or STT inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Speech
  python inference.py --mode tts --text "Hello world" --output speech.wav
  python inference.py --mode tts --text "Hello world" --output speech.wav --speed 1.2
  python inference.py --mode tts --text-file input.txt --output output.wav
  
  # Speech-to-Text
  python inference.py --mode stt --input audio.wav
  python inference.py --mode stt --input audio.wav --checkpoint ./checkpoints/stt_best.pt
  
  # Batch processing
  python inference.py --mode tts --text-file input.txt --output-dir ./outputs/
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['tts', 'stt'],
                       help='Inference mode: tts or stt')
    
    # TTS arguments
    parser.add_argument('--text', type=str, default=None,
                       help='Text to synthesize (for TTS)')
    
    parser.add_argument('--text-file', type=str, default=None,
                       help='File containing text to synthesize')
    
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path (for TTS)')
    
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for batch processing')
    
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed multiplier (0.5-2.0)')
    
    parser.add_argument('--pitch', type=float, default=0.0,
                       help='Pitch adjustment (-1.0 to 1.0)')
    
    # STT arguments
    parser.add_argument('--input', type=str, default=None,
                       help='Input audio file (for STT)')
    
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory for batch STT processing')
    
    # Common arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Output sample rate (for TTS)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def setup_device(args):
    """Setup inference device."""
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    return device


def load_models(mode, checkpoint_path, device, config):
    """Load models for inference."""
    logger = get_logger()
    
    data_config = DataConfig()
    model_config = ModelConfig()
    model_config.data_config = data_config
    
    models = {}
    
    if mode == 'tts':
        logger.info("Loading TTS model...")
        tts_model = TTSModel(model_config).to(device)
        vocoder = HiFiGANVocoder(model_config).to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            tts_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        tts_model.eval()
        vocoder.eval()
        
        models['tts'] = tts_model
        models['vocoder'] = vocoder
        
    elif mode == 'stt':
        logger.info("Loading STT model...")
        stt_model = STTModel(model_config).to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            stt_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        stt_model.eval()
        models['stt'] = stt_model
    
    return models, data_config, model_config


def run_tts_inference(text, models, data_config, model_config, device, args):
    """Run TTS inference."""
    logger = get_logger()
    
    # Initialize preprocessors
    text_normalizer = TextNormalizer()
    tokenizer = Tokenizer(vocab_size=model_config.tts_vocab_size)
    
    # Preprocess text
    logger.info(f"Processing text: '{text[:50]}...' " if len(text) > 50 else f"Processing text: '{text}'")
    
    normalized = text_normalizer.normalize(text)
    tokens = tokenizer.encode(normalized)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Run TTS model
    logger.info("Generating mel-spectrogram...")
    with torch.no_grad():
        tts_output = models['tts'](tokens_tensor)
        mel_spectrogram = tts_output['mel_spectrogram']
        
        logger.info(f"Mel-spectrogram shape: {mel_spectrogram.shape}")
        
        # Run vocoder
        logger.info("Synthesizing audio...")
        audio = models['vocoder'](mel_spectrogram)
        
        # Apply speed adjustment if needed
        if args.speed != 1.0:
            # Simple speed adjustment using interpolation
            import torch.nn.functional as F
            original_length = audio.shape[-1]
            new_length = int(original_length / args.speed)
            audio = F.interpolate(
                audio.unsqueeze(1), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
    
    # Convert to numpy
    audio_np = audio.squeeze(0).cpu().numpy()
    
    # Normalize audio
    audio_np = audio_np / np.abs(audio_np).max() * 0.95
    
    return audio_np


def run_stt_inference(audio_path, models, data_config, device):
    """Run STT inference."""
    logger = get_logger()
    
    # Load and preprocess audio
    logger.info(f"Loading audio: {audio_path}")
    
    audio, sr = sf.read(audio_path)
    
    # Resample if needed
    if sr != data_config.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=data_config.sample_rate)
    
    # Extract features
    preprocessor = AudioPreprocessor(data_config)
    features = preprocessor.extract_mel_spectrogram(audio)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    logger.info(f"Features shape: {features_tensor.shape}")
    
    # Run STT model
    logger.info("Transcribing...")
    with torch.no_grad():
        output = models['stt'](features_tensor)
        
        # Get CTC output for decoding
        ctc_output = output['ctc_output']
        
        # Simple greedy decoding (placeholder)
        # In production, use proper CTC decoding with language model
        predictions = ctc_output.argmax(dim=-1)
        
        # Remove duplicates and blanks
        transcription = []
        prev_token = None
        blank_id = model_config.stt_vocab_size  # Assuming blank is last token
        
        for token in predictions[0].cpu().numpy():
            if token != prev_token and token != blank_id:
                transcription.append(token)
            prev_token = token
        
        # Convert to text (placeholder - would use actual vocab)
        text = " ".join([f"tok_{t}" for t in transcription])
    
    return text


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logger(log_level)
    logger = get_logger()
    
    logger.info("="*60)
    logger.info(f"TTS-STT Inference - Mode: {args.mode.upper()}")
    logger.info("="*60)
    
    # Validate arguments
    if args.mode == 'tts':
        if not args.text and not args.text_file:
            logger.error("Error: Either --text or --text-file must be provided for TTS")
            sys.exit(1)
    elif args.mode == 'stt':
        if not args.input and not args.input_dir:
            logger.error("Error: Either --input or --input-dir must be provided for STT")
            sys.exit(1)
    
    # Setup device
    device = setup_device(args)
    logger.info(f"Using device: {device}")
    
    # Load models
    models, data_config, model_config = load_models(
        args.mode, args.checkpoint, device, args.config
    )
    
    # Run inference based on mode
    if args.mode == 'tts':
        # Get text to synthesize
        if args.text_file:
            with open(args.text_file, 'r') as f:
                text = f.read().strip()
        else:
            text = args.text
        
        # Run inference
        audio = run_tts_inference(
            text, models, data_config, model_config, device, args
        )
        
        # Save output
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        sf.write(args.output, audio, args.sample_rate)
        logger.info(f"Saved audio to: {args.output}")
        logger.info(f"Duration: {len(audio)/args.sample_rate:.2f}s")
        
    elif args.mode == 'stt':
        if args.input:
            # Single file
            text = run_stt_inference(args.input, models, data_config, device)
            print(f"\nTranscription: {text}\n")
            
        elif args.input_dir:
            # Batch processing
            logger.info(f"Processing files in: {args.input_dir}")
            
            audio_files = list(Path(args.input_dir).glob('*.wav')) + \
                         list(Path(args.input_dir).glob('*.mp3')) + \
                         list(Path(args.input_dir).glob('*.flac'))
            
            results = []
            for audio_file in audio_files:
                text = run_stt_inference(str(audio_file), models, data_config, device)
                results.append((audio_file.name, text))
                print(f"{audio_file.name}: {text}")
            
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, 'transcriptions.txt')
            with open(output_file, 'w') as f:
                for filename, text in results:
                    f.write(f"{filename}: {text}\n")
            
            logger.info(f"Saved transcriptions to: {output_file}")
    
    logger.info("="*60)
    logger.info("Inference complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
