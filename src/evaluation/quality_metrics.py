"""
Quality metrics module for TTS-STT system.
Implements MOS, WER, CER, and other evaluation metrics as per specification.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict


class QualityMetrics:
    """Quality evaluation metrics for TTS and STT systems."""
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER).
        WER = (S + D + I) / (S + D + C)
        where S=substitutions, D=deletions, I=insertions, C=correct
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            WER as a float between 0 and 1
        """
        # Normalize text
        ref_words = reference.lower().strip().split()
        hyp_words = hypothesis.lower().strip().split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        
        # Dynamic programming for edit distance
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Deletions
        for j in range(n + 1):
            dp[0][j] = j  # Insertions
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],     # Deletion
                        dp[i][j-1],     # Insertion
                        dp[i-1][j-1]    # Substitution
                    )
        
        # Calculate WER
        errors = dp[m][n]
        wer = errors / m
        
        return min(wer, 1.0)  # Cap at 100%
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER).
        Similar to WER but at character level.
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            CER as a float between 0 and 1
        """
        # Normalize text
        ref_chars = list(reference.lower().strip())
        hyp_chars = list(hypothesis.lower().strip())
        
        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0
        
        # Dynamic programming for edit distance
        m, n = len(ref_chars), len(hyp_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    )
        
        cer = dp[m][n] / m
        return min(cer, 1.0)
    
    @staticmethod
    def calculate_ser(references: List[str], hypotheses: List[str]) -> float:
        """
        Calculate Sentence Error Rate (SER).
        Percentage of sentences with any errors.
        
        Args:
            references: List of ground truth texts
            hypotheses: List of predicted texts
            
        Returns:
            SER as a float between 0 and 1
        """
        if len(references) == 0:
            return 0.0
        
        errors = sum(1 for ref, hyp in zip(references, hypotheses) 
                    if ref.lower().strip() != hyp.lower().strip())
        
        return errors / len(references)
    
    @staticmethod
    def calculate_wer_batch(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Calculate WER for a batch of texts with detailed breakdown.
        
        Args:
            references: List of ground truth texts
            hypotheses: List of predicted texts
            
        Returns:
            Dictionary with WER statistics
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        wers = []
        substitutions = 0
        deletions = 0
        insertions = 0
        total_words = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().strip().split()
            hyp_words = hyp.lower().strip().split()
            
            m, n = len(ref_words), len(hyp_words)
            
            if m == 0:
                continue
            
            # Compute edit distance with backtracking
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            # Backtrack to count error types
            i, j = m, n
            sub, dels, ins = 0, 0, 0
            
            while i > 0 or j > 0:
                if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
                    i -= 1
                    j -= 1
                elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                    sub += 1
                    i -= 1
                    j -= 1
                elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                    dels += 1
                    i -= 1
                else:
                    ins += 1
                    j -= 1
            
            wer = dp[m][n] / m
            wers.append(wer)
            substitutions += sub
            deletions += dels
            insertions += ins
            total_words += m
        
        avg_wer = np.mean(wers) if wers else 0.0
        
        return {
            'wer': avg_wer,
            'wer_std': np.std(wers) if wers else 0.0,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_words': total_words,
            'num_sentences': len(references)
        }
    
    @staticmethod
    def calculate_spectrogram_distance(predicted: torch.Tensor, 
                                      target: torch.Tensor,
                                      metric: str = 'l1') -> float:
        """
        Calculate distance between predicted and target spectrograms.
        
        Args:
            predicted: Predicted spectrogram (batch, n_mels, time)
            target: Target spectrogram (batch, n_mels, time)
            metric: 'l1', 'l2', or 'mse'
            
        Returns:
            Average distance
        """
        if predicted.shape != target.shape:
            # Resize to match
            min_len = min(predicted.size(-1), target.size(-1))
            predicted = predicted[..., :min_len]
            target = target[..., :min_len]
        
        if metric == 'l1':
            distance = torch.mean(torch.abs(predicted - target))
        elif metric == 'l2':
            distance = torch.mean((predicted - target) ** 2)
        elif metric == 'mse':
            distance = torch.mean((predicted - target) ** 2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distance.item()
    
    @staticmethod
    def calculate_spectral_convergence(predicted: torch.Tensor, 
                                      target: torch.Tensor) -> float:
        """
        Calculate spectral convergence metric.
        SC = ||log(|S|) - log(|Ŝ|)||₂ / ||log(|S|)||₂
        
        Args:
            predicted: Predicted spectrogram
            target: Target spectrogram
            
        Returns:
            Spectral convergence value
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        
        # Take absolute values and add epsilon
        pred_mag = torch.abs(predicted) + eps
        target_mag = torch.abs(target) + eps
        
        # Log magnitudes
        log_pred = torch.log(pred_mag)
        log_target = torch.log(target_mag)
        
        # Compute L2 distance
        numerator = torch.sqrt(torch.sum((log_target - log_pred) ** 2))
        denominator = torch.sqrt(torch.sum(log_target ** 2))
        
        sc = numerator / (denominator + eps)
        
        return sc.item()
    
    @staticmethod
    def calculate_f0_rmse(predicted_f0: torch.Tensor, 
                         target_f0: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate Root Mean Square Error for fundamental frequency (F0).
        
        Args:
            predicted_f0: Predicted F0 contours
            target_f0: Target F0 contours
            mask: Optional mask for voiced regions
            
        Returns:
            RMSE in Hz
        """
        if mask is not None:
            predicted_f0 = predicted_f0[mask]
            target_f0 = target_f0[mask]
        
        mse = torch.mean((predicted_f0 - target_f0) ** 2)
        rmse = torch.sqrt(mse)
        
        return rmse.item()
    
    @staticmethod
    def calculate_mos_proxy(spectrogram_distance: float,
                           f0_rmse: float,
                           weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate proxy MOS (Mean Opinion Score) based on objective metrics.
        This is a simplified proxy - real MOS requires human evaluation.
        
        Args:
            spectrogram_distance: L1 spectrogram distance
            f0_rmse: F0 RMSE
            weights: Optional weights for combining metrics
            
        Returns:
            Proxy MOS score (1-5 scale)
        """
        if weights is None:
            weights = {'spec': 0.6, 'f0': 0.4}
        
        # Normalize metrics (lower is better)
        # These thresholds are approximate
        spec_score = max(0, 1 - spectrogram_distance / 2.0)
        f0_score = max(0, 1 - f0_rmse / 100.0)
        
        # Weighted combination
        combined = weights['spec'] * spec_score + weights['f0'] * f0_score
        
        # Scale to 1-5
        mos = 1 + 4 * combined
        
        return min(5.0, max(1.0, mos))
    
    @staticmethod
    def calculate_real_time_factor(processing_time: float, 
                                   audio_duration: float) -> float:
        """
        Calculate Real-Time Factor (RTF).
        RTF = processing_time / audio_duration
        
        Args:
            processing_time: Time taken to process (seconds)
            audio_duration: Duration of audio (seconds)
            
        Returns:
            RTF value (<1 means faster than real-time)
        """
        if audio_duration == 0:
            return float('inf')
        
        return processing_time / audio_duration


class BenchmarkMetrics:
    """Benchmarking metrics for performance evaluation."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start a timer for a metric."""
        import time
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """End a timer and record the duration."""
        import time
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[name].append(duration)
            del self.start_times[name]
            return duration
        return None
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        self.metrics[name].append(value)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics[name]
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p50': np.percentile(values, 50),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'count': len(values)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_statistics(name) for name in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()
    
    def print_summary(self):
        """Print a summary of all metrics."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for name in sorted(self.metrics.keys()):
            stats = self.get_statistics(name)
            print(f"\n{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean:  {stats['mean']:.4f}")
            print(f"  Std:   {stats['std']:.4f}")
            print(f"  Min:   {stats['min']:.4f}")
            print(f"  Max:   {stats['max']:.4f}")
            print(f"  P95:   {stats['p95']:.4f}")


class TTSEvaluator:
    """Evaluator for TTS quality."""
    
    def __init__(self):
        self.metrics = QualityMetrics()
    
    def evaluate(self, 
                 predicted_mel: torch.Tensor,
                 target_mel: torch.Tensor,
                 predicted_f0: Optional[torch.Tensor] = None,
                 target_f0: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate TTS output quality.
        
        Args:
            predicted_mel: Predicted mel-spectrogram
            target_mel: Target mel-spectrogram
            predicted_f0: Optional predicted F0
            target_f0: Optional target F0
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Spectrogram metrics
        results['mel_l1'] = self.metrics.calculate_spectrogram_distance(
            predicted_mel, target_mel, metric='l1'
        )
        results['mel_l2'] = self.metrics.calculate_spectrogram_distance(
            predicted_mel, target_mel, metric='l2'
        )
        results['spectral_convergence'] = self.metrics.calculate_spectral_convergence(
            predicted_mel, target_mel
        )
        
        # F0 metrics
        if predicted_f0 is not None and target_f0 is not None:
            results['f0_rmse'] = self.metrics.calculate_f0_rmse(predicted_f0, target_f0)
        
        # Proxy MOS
        results['proxy_mos'] = self.metrics.calculate_mos_proxy(
            results['mel_l1'],
            results.get('f0_rmse', 10.0)
        )
        
        return results


class STTEvaluator:
    """Evaluator for STT quality."""
    
    def __init__(self):
        self.metrics = QualityMetrics()
    
    def evaluate(self, 
                 predictions: List[str],
                 references: List[str]) -> Dict[str, float]:
        """
        Evaluate STT output quality.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary of metrics
        """
        # WER with breakdown
        wer_stats = self.metrics.calculate_wer_batch(references, predictions)
        
        # CER for each pair
        cers = [self.metrics.calculate_cer(ref, hyp) 
                for ref, hyp in zip(references, predictions)]
        
        # SER
        ser = self.metrics.calculate_ser(references, predictions)
        
        results = {
            'wer': wer_stats['wer'],
            'wer_std': wer_stats['wer_std'],
            'cer': np.mean(cers),
            'cer_std': np.std(cers),
            'ser': ser,
            'substitutions': wer_stats['substitutions'],
            'deletions': wer_stats['deletions'],
            'insertions': wer_stats['insertions'],
            'total_words': wer_stats['total_words'],
            'num_sentences': wer_stats['num_sentences']
        }
        
        return results


# Convenience functions
def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate WER for a single pair."""
    return QualityMetrics.calculate_wer(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate CER for a single pair."""
    return QualityMetrics.calculate_cer(reference, hypothesis)


def calculate_wer_batch(references: List[str], 
                       hypotheses: List[str]) -> Dict[str, float]:
    """Calculate WER for a batch."""
    return QualityMetrics.calculate_wer_batch(references, hypotheses)
