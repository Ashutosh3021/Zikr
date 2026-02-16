"""
Evaluation module for TTS-STT system.
Provides quality metrics and benchmarking capabilities.
"""

from .quality_metrics import (
    QualityMetrics,
    BenchmarkMetrics,
    TTSEvaluator,
    STTEvaluator,
    calculate_wer,
    calculate_cer,
    calculate_wer_batch
)

from .benchmark import (
    PerformanceBenchmark,
    BenchmarkResult,
    run_benchmarks
)

__all__ = [
    'QualityMetrics',
    'BenchmarkMetrics',
    'TTSEvaluator',
    'STTEvaluator',
    'calculate_wer',
    'calculate_cer',
    'calculate_wer_batch',
    'PerformanceBenchmark',
    'BenchmarkResult',
    'run_benchmarks'
]
