#!/usr/bin/env python3
"""
Main entry point for TTS-STT system.
Provides a unified CLI for all operations.

Usage:
  python main.py test              # Run all tests
  python main.py train --model tts # Train a model
  python main.py inference --mode tts --text "Hello"  # Run inference
  python main.py server            # Start API server
  python main.py benchmark         # Run performance benchmarks
  python main.py validate          # Validate installation
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.absolute()


def run_tests(args):
    """Run test suite."""
    print("\n" + "="*60)
    print("RUNNING TEST SUITE")
    print("="*60 + "\n")
    
    test_files = [
        "tests/test_data_processing.py",
        "tests/test_tts_model.py",
        "tests/test_stt_model.py",
        "tests/test_hifigan_vocoder.py",
        "tests/test_training_pipeline.py",
        "tests/test_inference_engine.py",
    ]
    
    if args.integration:
        test_files.append("tests/test_integration.py")
    
    failed = []
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nRunning {test_file}...")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                failed.append(test_file)
        else:
            print(f"[SKIP] {test_file} not found, skipping")
    
    print("\n" + "="*60)
    if not failed:
        print("[SUCCESS] ALL TESTS PASSED")
    else:
        print(f"[FAIL] {len(failed)} TEST(S) FAILED:")
        for f in failed:
            print(f"  - {f}")
    print("="*60 + "\n")
    
    return len(failed) == 0


def run_training(args):
    """Run training."""
    cmd = [
        sys.executable, "train.py",
        "--model", args.model,
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    
    if args.verbose:
        cmd.append("--verbose")
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_inference(args):
    """Run inference."""
    cmd = [
        sys.executable, "inference.py",
        "--mode", args.mode,
    ]
    
    if args.text:
        cmd.extend(["--text", args.text])
    if args.text_file:
        cmd.extend(["--text-file", args.text_file])
    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_server(args):
    """Start API server."""
    cmd = [
        sys.executable, "api_server.py",
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_benchmark(args):
    """Run performance benchmarks."""
    cmd = [
        sys.executable, "-m", "src.evaluation.benchmark",
        "--runs", str(args.runs),
    ]
    
    if args.export:
        cmd.extend(["--export", args.export])
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_validation(args):
    """Validate installation and setup."""
    print("\n" + "="*60)
    print("TTS-STT SYSTEM VALIDATION")
    print("="*60 + "\n")
    
    # Check Python version
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   [OK] Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"   [FAIL] Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8)")
        return False
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    required_packages = [
        'torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 
        'soundfile', 'fastapi', 'uvicorn', 'pytest'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   [OK] {package}")
        except ImportError:
            print(f"   [MISSING] {package}")
            missing.append(package)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # Check CUDA availability
    print("\n3. Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   [OK] CUDA version: {torch.version.cuda}")
        else:
            print("   [WARN] CUDA not available (will use CPU)")
    except Exception as e:
        print(f"   [ERROR] Error checking CUDA: {e}")
    
    # Check project structure
    print("\n4. Checking project structure...")
    required_dirs = [
        'src/models', 'src/data', 'src/training', 
        'src/inference', 'src/evaluation', 'src/utils', 'tests'
    ]
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"   [OK] {dir_path}/")
        else:
            print(f"   [MISSING] {dir_path}/")
            return False
    
    # Check key files
    print("\n5. Checking key files...")
    required_files = [
        'train.py', 'inference.py', 'api_server.py',
        'requirements.txt', 'README.md'
    ]
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"   [OK] {file_path}")
        else:
            print(f"   [MISSING] {file_path}")
    
    # Try importing project modules
    print("\n6. Checking project modules...")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.config import ModelConfig, DataConfig
        print("   [OK] config module")
        
        from src.models.tts_model import TTSModel
        print("   [OK] tts_model module")
        
        from src.models.stt_model import STTModel
        print("   [OK] stt_model module")
        
        from src.models.hifigan_vocoder import HiFiGANVocoder
        print("   [OK] hifigan_vocoder module")
        
        from src.training.training_pipeline import ModelTrainer
        print("   [OK] training_pipeline module")
        
        from src.evaluation.quality_metrics import QualityMetrics
        print("   [OK] quality_metrics module")
        
    except Exception as e:
        print(f"   [ERROR] Error importing modules: {e}")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("[SUCCESS] VALIDATION COMPLETE - SYSTEM READY")
    print("="*60)
    print("\nYou can now:")
    print("  - Run tests:        python main.py test")
    print("  - Train models:     python main.py train --model tts")
    print("  - Run inference:    python main.py inference --mode tts --text 'Hello'")
    print("  - Start API server: python main.py server")
    print("  - Run benchmarks:   python main.py benchmark")
    print("="*60 + "\n")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TTS-STT System - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate installation
  python main.py validate

  # Run all tests
  python main.py test
  python main.py test --integration

  # Train models
  python main.py train --model tts --data-dir ./data/tts
  python main.py train --model stt --data-dir ./data/stt
  python main.py train --model hifigan --data-dir ./data/audio

  # Run inference
  python main.py inference --mode tts --text "Hello world"
  python main.py inference --mode stt --input audio.wav

  # Start API server
  python main.py server
  python main.py server --port 8080

  # Run benchmarks
  python main.py benchmark
  python main.py benchmark --runs 100

For more help on a specific command:
  python main.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--integration', action='store_true',
                            help='Include integration tests')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, required=True,
                             choices=['tts', 'stt', 'hifigan'],
                             help='Model to train')
    train_parser.add_argument('--data-dir', type=str, default='./data',
                             help='Training data directory')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Resume from checkpoint')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Verbose logging')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--mode', type=str, required=True,
                                 choices=['tts', 'stt'],
                                 help='Inference mode')
    inference_parser.add_argument('--text', type=str, default=None,
                                 help='Text for TTS')
    inference_parser.add_argument('--text-file', type=str, default=None,
                                 help='Text file for TTS')
    inference_parser.add_argument('--input', type=str, default=None,
                                 help='Input audio file for STT')
    inference_parser.add_argument('--output', type=str, default='output.wav',
                                 help='Output file')
    inference_parser.add_argument('--checkpoint', type=str, default=None,
                                 help='Model checkpoint')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', type=str, default='0.0.0.0',
                              help='Host to bind')
    server_parser.add_argument('--port', type=int, default=8000,
                              help='Port to bind')
    server_parser.add_argument('--reload', action='store_true',
                              help='Enable auto-reload')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--runs', type=int, default=50,
                                 help='Number of benchmark runs')
    benchmark_parser.add_argument('--export', type=str, default='benchmark_results.txt',
                                 help='Export file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate installation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == 'test':
        success = run_tests(args)
        sys.exit(0 if success else 1)
    
    elif args.command == 'train':
        run_training(args)
    
    elif args.command == 'inference':
        run_inference(args)
    
    elif args.command == 'server':
        run_server(args)
    
    elif args.command == 'benchmark':
        run_benchmark(args)
    
    elif args.command == 'validate':
        success = run_validation(args)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
