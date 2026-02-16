"""
Performance benchmarking suite for TTS-STT system.
Provides comprehensive benchmarking for latency, throughput, and resource usage.
"""

import torch
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

from src.utils.config import ModelConfig, DataConfig
from src.models.tts_model import TTSModel
from src.models.stt_model import STTModel
from src.models.hifigan_vocoder import HiFiGANVocoder
from src.data.text_preprocessing import TextNormalizer, Tokenizer
from src.data.audio_preprocessing import AudioPreprocessor


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float  # items per second
    memory_mb: float
    num_runs: int


class PerformanceBenchmark:
    """Performance benchmarking for TTS-STT system."""
    
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, 
                 device: Optional[torch.device] = None):
        """
        Initialize benchmark suite.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
            device: Device to run benchmarks on
        """
        self.model_config = model_config
        self.data_config = data_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.tts_model = TTSModel(model_config).to(self.device)
        self.stt_model = STTModel(model_config).to(self.device)
        self.vocoder = HiFiGANVocoder(model_config).to(self.device)
        
        # Set to eval mode
        self.tts_model.eval()
        self.stt_model.eval()
        self.vocoder.eval()
        
        # Initialize preprocessors
        self.text_normalizer = TextNormalizer()
        self.tokenizer = Tokenizer(vocab_size=model_config.tts_vocab_size)
        self.audio_preprocessor = AudioPreprocessor(data_config)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def _measure_memory(self):
        """Context manager to measure memory usage."""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        yield
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        self._last_memory = mem_after - mem_before
        
        if torch.cuda.is_available():
            self._last_cuda_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    def _run_benchmark(self, func: Callable, inputs: List, 
                      warmup_runs: int = 5, num_runs: int = 50) -> BenchmarkResult:
        """
        Run a benchmark on a function.
        
        Args:
            func: Function to benchmark
            inputs: List of inputs to cycle through
            warmup_runs: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult with statistics
        """
        times = []
        
        # Warmup
        with torch.no_grad():
            for i in range(warmup_runs):
                inp = inputs[i % len(inputs)]
                func(inp)
        
        # Synchronize GPU if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark runs
        with torch.no_grad():
            for i in range(num_runs):
                inp = inputs[i % len(inputs)]
                
                # Start timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Run function
                func(inp)
                
                # End timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
        
        times_array = np.array(times)
        
        return BenchmarkResult(
            name=func.__name__,
            mean_ms=float(np.mean(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            p50_ms=float(np.percentile(times_array, 50)),
            p95_ms=float(np.percentile(times_array, 95)),
            p99_ms=float(np.percentile(times_array, 99)),
            throughput=1000.0 / float(np.mean(times_array)),
            memory_mb=getattr(self, '_last_memory', 0.0),
            num_runs=num_runs
        )
    
    def benchmark_tts_single(self, num_runs: int = 50) -> BenchmarkResult:
        """
        Benchmark TTS inference for single sentences.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult
        """
        print("Benchmarking TTS (single sentence)...")
        
        test_texts = [
            "Hello world",
            "This is a test sentence",
            "The quick brown fox jumps",
            "Machine learning is fascinating",
            "Text to speech synthesis works"
        ]
        
        # Prepare inputs
        inputs = []
        for text in test_texts:
            normalized = self.text_normalizer.normalize(text)
            tokens = self.tokenizer.encode(normalized)
            inputs.append(torch.tensor([tokens], dtype=torch.long).to(self.device))
        
        def tts_forward(tokens):
            with torch.no_grad():
                output = self.tts_model(tokens)
                return output['mel_spectrogram']
        
        result = self._run_benchmark(tts_forward, inputs, num_runs=num_runs)
        result.name = "TTS_Single"
        self.results.append(result)
        
        return result
    
    def benchmark_tts_full(self, num_runs: int = 50) -> BenchmarkResult:
        """
        Benchmark full TTS pipeline including vocoder.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult
        """
        print("Benchmarking TTS (full pipeline)...")
        
        test_texts = [
            "Hello world",
            "This is a test sentence",
            "The quick brown fox jumps",
            "Machine learning is fascinating",
            "Text to speech synthesis works"
        ]
        
        # Prepare inputs
        inputs = []
        for text in test_texts:
            normalized = self.text_normalizer.normalize(text)
            tokens = self.tokenizer.encode(normalized)
            inputs.append(torch.tensor([tokens], dtype=torch.long).to(self.device))
        
        def tts_full(tokens):
            with torch.no_grad():
                output = self.tts_model(tokens)
                mel = output['mel_spectrogram']
                audio = self.vocoder(mel)
                return audio
        
        result = self._run_benchmark(tts_full, inputs, num_runs=num_runs)
        result.name = "TTS_Full"
        self.results.append(result)
        
        return result
    
    def benchmark_stt(self, num_runs: int = 50) -> BenchmarkResult:
        """
        Benchmark STT inference.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult
        """
        print("Benchmarking STT...")
        
        # Create synthetic audio inputs of different lengths
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_rate = self.data_config.sample_rate
        
        inputs = []
        for duration in durations:
            samples = int(sample_rate * duration)
            t = torch.linspace(0, duration, samples)
            audio = torch.sin(2 * np.pi * 440 * t).numpy()
            features = self.audio_preprocessor.extract_mel_spectrogram(audio)
            inputs.append(torch.from_numpy(features).unsqueeze(0).to(self.device))
        
        def stt_forward(features):
            with torch.no_grad():
                output = self.stt_model(features)
                return output['encoder_output']
        
        result = self._run_benchmark(stt_forward, inputs, num_runs=num_runs)
        result.name = "STT"
        self.results.append(result)
        
        return result
    
    def benchmark_vocoder(self, num_runs: int = 50) -> BenchmarkResult:
        """
        Benchmark vocoder inference.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult
        """
        print("Benchmarking Vocoder...")
        
        # Create synthetic mel-spectrograms
        lengths = [50, 100, 150, 200, 250]
        n_mels = self.data_config.n_mels
        
        inputs = []
        for length in lengths:
            mel = torch.randn(1, n_mels, length).to(self.device)
            inputs.append(mel)
        
        def vocoder_forward(mel):
            with torch.no_grad():
                return self.vocoder(mel)
        
        result = self._run_benchmark(vocoder_forward, inputs, num_runs=num_runs)
        result.name = "Vocoder"
        self.results.append(result)
        
        return result
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = [1, 2, 4, 8, 16]) -> Dict[int, BenchmarkResult]:
        """
        Benchmark batch processing for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping batch size to BenchmarkResult
        """
        print("Benchmarking batch processing...")
        
        results = {}
        text = "This is a test sentence for batch processing"
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Prepare batch input
            normalized = self.text_normalizer.normalize(text)
            tokens = self.tokenizer.encode(normalized)
            batch_tokens = torch.tensor([tokens] * batch_size, dtype=torch.long).to(self.device)
            
            inputs = [batch_tokens]
            
            def batch_forward(tokens):
                with torch.no_grad():
                    output = self.tts_model(tokens)
                    return output['mel_spectrogram']
            
            result = self._run_benchmark(batch_forward, inputs, num_runs=20)
            result.name = f"Batch_{batch_size}"
            results[batch_size] = result
            self.results.append(result)
        
        return results
    
    def run_full_benchmark(self, num_runs: int = 50) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite.
        
        Args:
            num_runs: Number of runs per benchmark
            
        Returns:
            List of all benchmark results
        """
        print("\n" + "="*60)
        print("STARTING FULL PERFORMANCE BENCHMARK")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Runs per benchmark: {num_runs}\n")
        
        self.results = []
        
        # Run individual component benchmarks
        self.benchmark_tts_single(num_runs)
        self.benchmark_tts_full(num_runs)
        self.benchmark_stt(num_runs)
        self.benchmark_vocoder(num_runs)
        
        # Run batch benchmarks
        self.benchmark_batch_processing()
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        
        return self.results
    
    def print_results(self):
        """Print all benchmark results in a formatted table."""
        print("\n" + "="*100)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*100)
        print(f"{'Name':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<15}")
        print("-"*100)
        
        for result in self.results:
            print(f"{result.name:<20} "
                  f"{result.mean_ms:<12.2f} "
                  f"{result.std_ms:<12.2f} "
                  f"{result.p95_ms:<12.2f} "
                  f"{result.p99_ms:<12.2f} "
                  f"{result.throughput:<15.2f}")
        
        print("="*100)
        
        # Print recommendations
        print("\nPERFORMANCE RECOMMENDATIONS:")
        print("-"*100)
        
        for result in self.results:
            if result.mean_ms < 100:
                status = "✓ EXCELLENT"
            elif result.mean_ms < 300:
                status = "✓ GOOD"
            elif result.mean_ms < 500:
                status = "⚠ ACCEPTABLE"
            else:
                status = "✗ NEEDS OPTIMIZATION"
            
            print(f"{result.name:<20} {status:<20} (Mean: {result.mean_ms:.2f}ms)")
        
        print("="*100)
    
    def export_results(self, filename: str = "benchmark_results.txt"):
        """Export benchmark results to a file."""
        with open(filename, 'w') as f:
            f.write("="*100 + "\n")
            f.write("TTS-STT PERFORMANCE BENCHMARK RESULTS\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Device: {self.device}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA Version: {torch.version.cuda}\n")
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write("\n")
            
            f.write("="*100 + "\n")
            f.write(f"{'Name':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} "
                   f"{'Max (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}\n")
            f.write("="*100 + "\n")
            
            for result in self.results:
                f.write(f"{result.name:<20} "
                       f"{result.mean_ms:<12.2f} "
                       f"{result.std_ms:<12.2f} "
                       f"{result.min_ms:<12.2f} "
                       f"{result.max_ms:<12.2f} "
                       f"{result.p50_ms:<12.2f} "
                       f"{result.p95_ms:<12.2f} "
                       f"{result.p99_ms:<12.2f}\n")
            
            f.write("="*100 + "\n")
        
        print(f"\nResults exported to: {filename}")


def run_benchmarks(num_runs: int = 50, export_file: Optional[str] = None):
    """
    Convenience function to run benchmarks.
    
    Args:
        num_runs: Number of runs per benchmark
        export_file: Optional file to export results
    """
    data_config = DataConfig()
    model_config = ModelConfig()
    model_config.data_config = data_config
    
    benchmark = PerformanceBenchmark(model_config, data_config)
    benchmark.run_full_benchmark(num_runs=num_runs)
    benchmark.print_results()
    
    if export_file:
        benchmark.export_results(export_file)
    
    return benchmark.results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TTS-STT Performance Benchmarks')
    parser.add_argument('--runs', type=int, default=50, help='Number of runs per benchmark')
    parser.add_argument('--export', type=str, default='benchmark_results.txt', 
                       help='Export results to file')
    args = parser.parse_args()
    
    run_benchmarks(num_runs=args.runs, export_file=args.export)
