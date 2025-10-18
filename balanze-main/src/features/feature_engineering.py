"""
Feature Engineering for Gait and Balance Analysis

This module provides functionality to extract meaningful features from raw
gait and balance data for machine learning models.
"""

import numpy as np
from scipy import signal, stats, fft
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from tqdm import tqdm
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureDomain(Enum):
    """Feature domains for gait and balance analysis."""
    TIME = "time"
    FREQUENCY = "frequency"
    NONLINEAR = "nonlinear"
    GEOMETRIC = "geometric"
    STATISTICAL = "statistical"

@dataclass
class FeatureExtractorConfig:
    """Configuration for feature extraction."""
    sample_rate: float = 100.0  # Hz
    window_size: float = 2.0  # seconds
    overlap: float = 0.5  # fraction of window to overlap
    frequency_bands: Dict[str, Tuple[float, float]] = None
    include_domains: List[FeatureDomain] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.frequency_bands is None:
            self.frequency_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
        
        if self.include_domains is None:
            self.include_domains = [
                FeatureDomain.TIME,
                FeatureDomain.FREQUENCY,
                FeatureDomain.STATISTICAL,
                FeatureDomain.GEOMETRIC,
                FeatureDomain.NONLINEAR
            ]

class GaitFeatureExtractor:
    """Extract features from gait and balance data."""
    
    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        """Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config if config is not None else FeatureExtractorConfig()
        self.window_samples = int(self.config.window_size * self.config.sample_rate)
        self.step_size = int(self.window_samples * (1 - self.config.overlap))
    
    def extract_features(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract features from input data.
        
        Args:
            data: Dictionary containing time series data (e.g., joint angles, positions)
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        for signal_name, signal_data in data.items():
            # Skip non-numeric or empty data
            if not isinstance(signal_data, np.ndarray) or signal_data.size == 0:
                continue
                
            # Ensure 2D array (samples, channels)
            if signal_data.ndim == 1:
                signal_data = signal_data.reshape(-1, 1)
            
            # Process each channel separately
            for channel in range(signal_data.shape[1]):
                channel_data = signal_data[:, channel]
                channel_name = f"{signal_name}_{channel}" if signal_data.shape[1] > 1 else signal_name
                
                # Extract features for this channel
                channel_features = self._extract_channel_features(channel_data)
                
                # Add channel features to results
                for feature_name, feature_value in channel_features.items():
                    features[f"{channel_name}_{feature_name}"] = feature_value
        
        return features
    
    def _extract_channel_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract features from a single channel signal."""
        features = {}
        
        # Remove any NaN or inf values
        signal = signal[~np.isnan(signal) & ~np.isinf(signal)]
        
        if len(signal) < self.window_samples:
            # If signal is shorter than window, pad with zeros
            padded_signal = np.zeros(self.window_samples)
            padded_signal[:len(signal)] = signal
            signal = padded_signal
        
        # Extract features from each domain
        if FeatureDomain.TIME in self.config.include_domains:
            features.update(self._extract_time_domain_features(signal))
        
        if FeatureDomain.FREQUENCY in self.config.include_domains:
            features.update(self._extract_frequency_domain_features(signal))
        
        if FeatureDomain.STATISTICAL in self.config.include_domains:
            features.update(self._extract_statistical_features(signal))
        
        if FeatureDomain.GEOMETRIC in self.config.include_domains:
            features.update(self._extract_geometric_features(signal))
        
        if FeatureDomain.NONLINEAR in self.config.include_domains:
            features.update(self._extract_nonlinear_features(signal))
        
        return features
    
    def _extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract time domain features."""
        features = {}
        
        # Basic statistics
        features['mean'] = float(np.mean(signal))
        features['std'] = float(np.std(signal))
        features['rms'] = float(np.sqrt(np.mean(signal**2)))
        features['max'] = float(np.max(signal))
        features['min'] = float(np.min(signal))
        features['range'] = features['max'] - features['min']
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = float(len(zero_crossings)) / len(signal)
        
        # Signal energy
        features['energy'] = float(np.sum(signal**2))
        
        # Signal power
        features['power'] = features['energy'] / len(signal)
        
        # Signal magnitude area
        features['sma'] = float(np.sum(np.abs(signal)))
        
        # Waveform length
        features['waveform_length'] = float(np.sum(np.abs(np.diff(signal))))
        
        return features
    
    def _extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features."""
        features = {}
        
        # Compute FFT
        n = len(signal)
        fft_vals = np.abs(fft.fft(signal - np.mean(signal)))
        fft_freq = fft.fftfreq(n, 1.0 / self.config.sample_rate)
        
        # Get only positive frequencies
        pos_freq_mask = fft_freq > 0
        fft_vals = fft_vals[:len(fft_vals)//2][pos_freq_mask[:n//2]]
        fft_freq = fft_freq[:n//2][pos_freq_mask[:n//2]]
        
        if len(fft_vals) == 0:
            return features
        
        # Total power
        total_power = np.sum(fft_vals**2)
        features['total_power'] = float(total_power)
        
        # Spectral centroid and spread
        if total_power > 0:
            spectral_centroid = np.sum(fft_freq * fft_vals**2) / total_power
            features['spectral_centroid'] = float(spectral_centroid)
            
            if len(fft_vals) > 1:
                spectral_spread = np.sqrt(
                    np.sum((fft_freq - spectral_centroid)**2 * fft_vals**2) / total_power
                )
                features['spectral_spread'] = float(spectral_spread)
        
        # Band power in specified frequency bands
        for band_name, (low_freq, high_freq) in self.config.frequency_bands.items():
            band_mask = (fft_freq >= low_freq) & (fft_freq <= high_freq)
            if np.any(band_mask):
                band_power = np.sum(fft_vals[band_mask]**2)
                features[f'power_{band_name}'] = float(band_power)
                features[f'power_{band_name}_norm'] = float(band_power / total_power)
        
        # Spectral entropy
        if len(fft_vals) > 1:
            power_spectrum = fft_vals**2
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
            features['spectral_entropy'] = float(spectral_entropy)
        
        return features
    
    def _extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}
        
        # Basic statistics
        features['kurtosis'] = float(stats.kurtosis(signal))
        features['skewness'] = float(stats.skew(signal))
        features['median'] = float(np.median(signal))
        features['iqr'] = float(stats.iqr(signal))
        
        # Percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            features[f'percentile_{p}'] = float(np.percentile(signal, p))
        
        # Autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / np.max(autocorr) if np.max(autocorr) > 0 else autocorr
        
        # Find first zero crossing of autocorrelation
        zero_crossings = np.where(np.diff(np.signbit(autocorr - 0.5)))[0]
        features['autocorr_zero_crossing'] = float(zero_crossings[0] / self.config.sample_rate) if len(zero_crossings) > 0 else 0.0
        
        return features
    
    def _extract_geometric_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract geometric features."""
        features = {}
        
        # Signal magnitude area (SMA)
        features['sma'] = float(np.sum(np.abs(signal)) / len(signal))
        
        # Signal vector magnitude (SVM)
        if isinstance(signal, np.ndarray) and signal.ndim > 1 and signal.shape[1] > 1:
            svm = np.sqrt(np.sum(signal**2, axis=1))
            features['svm_mean'] = float(np.mean(svm))
            features['svm_std'] = float(np.std(svm))
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = float(len(zero_crossings)) / len(signal)
        
        # Waveform length
        features['waveform_length'] = float(np.sum(np.abs(np.diff(signal))))
        
        return features
    
    def _extract_nonlinear_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract nonlinear features."""
        features = {}
        
        # Sample entropy
        if len(signal) >= 10:  # Need sufficient samples
            try:
                features['sample_entropy'] = self._calculate_sample_entropy(signal)
            except:
                features['sample_entropy'] = 0.0
        
        # Detrended fluctuation analysis (DFA)
        try:
            features['dfa_alpha'] = self._calculate_dfa(signal)
        except:
            features['dfa_alpha'] = 0.0
        
        # Recurrence quantification analysis (simplified)
        try:
            rqa_features = self._calculate_rqa(signal)
            features.update(rqa_features)
        except:
            pass
        
        return features
    
    def _calculate_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy of a time series."""
        n = len(signal)
        
        def _maxdist(x, y):
            return np.max(np.abs(x - y))
        
        def _phi(m):
            x = np.array([signal[i:i+m] for i in range(n - m + 1)])
            C = np.zeros(len(x))
            
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j:
                        if _maxdist(x[i], x[j]) <= r:
                            C[i] += 1
            
            return np.sum(C) / (n - m + 1.0)
        
        if len(signal) < m + 1:
            return 0.0
        
        r = r * np.std(signal)
        return -np.log(_phi(m + 1) / _phi(m))
    
    def _calculate_dfa(self, signal: np.ndarray, min_scale: int = 4, max_scale: int = None) -> float:
        """Calculate detrended fluctuation analysis (DFA) exponent."""
        n = len(signal)
        if max_scale is None:
            max_scale = n // 4
        
        # Integrate the signal
        y = np.cumsum(signal - np.mean(signal))
        
        # Define scales
        scales = 2**np.arange(int(np.log2(min_scale)), int(np.log2(max_scale)) + 1)
        scales = scales[scales < n // 4]  # Ensure we have enough points
        
        # Calculate RMS for each scale
        rms = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Split the signal into non-overlapping windows
            n_windows = n // scale
            if n_windows < 2:
                continue
                
            # Reshape and detrend
            y_reshaped = y[:n_windows * scale].reshape(n_windows, scale)
            x = np.arange(scale)
            
            # Detrend each window
            for j in range(n_windows):
                if scale > 1:  # Need at least 2 points for linear regression
                    coeffs = np.polyfit(x, y_reshaped[j], 1)
                    y_reshaped[j] = y_reshaped[j] - (coeffs[0] * x + coeffs[1])
            
            # Calculate RMS
            rms[i] = np.sqrt(np.mean(y_reshaped**2))
        
        # Fit line to log-log scale
        if len(scales) >= 2 and np.any(rms > 0):
            coeffs = np.polyfit(np.log2(scales[rms > 0]), np.log2(rms[rms > 0]), 1)
            return float(coeffs[0])
        
        return 0.0
    
    def _calculate_rqa(self, signal: np.ndarray, delay: int = 1, dim: int = 2, threshold: float = 0.2) -> Dict[str, float]:
        """Calculate recurrence quantification analysis features (simplified)."""
        n = len(signal)
        if n < dim * delay:
            return {}
        
        # Create phase space vectors
        vectors = np.zeros((n - (dim - 1) * delay, dim))
        for i in range(dim):
            vectors[:, i] = signal[i*delay : n - (dim - i - 1)*delay]
        
        # Calculate distance matrix (upper triangular)
        dist_matrix = np.zeros((len(vectors), len(vectors)))
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dist_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])
        
        # Threshold to get recurrence plot
        threshold_value = threshold * np.max(dist_matrix)
        recurrence_plot = (dist_matrix <= threshold_value).astype(int)
        
        # Calculate RQA metrics
        n_points = len(vectors)
        total_points = n_points * (n_points - 1) / 2
        
        # Recurrence rate
        rr = np.sum(recurrence_plot) / total_points if total_points > 0 else 0
        
        # Diagonal lines (simplified)
        diag_lines = []
        for i in range(1 - n_points, n_points):
            diag = np.diag(recurrence_plot, i)
            if len(diag) > 1:
                diag_lines.extend(np.split(diag, np.where(np.diff(diag) != 0)[0] + 1))
        
        diag_lengths = [len(line) for line in diag_lines if np.all(line == 1)]
        
        # RQA features
        features = {
            'rqa_rr': float(rr),
            'rqa_det': float(np.sum(np.array(diag_lengths)) / (np.sum(recurrence_plot) + 1e-10)),
            'rqa_lmax': float(np.max(diag_lengths) if diag_lengths else 0),
            'rqa_lmean': float(np.mean(diag_lengths) if diag_lengths else 0),
            'rqa_entropy': float(stats.entropy(diag_lengths) if diag_lengths else 0)
        }
        
        return features
