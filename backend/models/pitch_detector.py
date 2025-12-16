"""
CREPE-based pitch detection for high-accuracy f0 estimation
"""
import numpy as np
import librosa
import crepe
from typing import Tuple, Optional, Dict, Any
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class CreepePitchDetector:
    """
    High-accuracy pitch detection using CREPE (Convolutional Representation for Pitch Estimation)
    """

    def __init__(self, model_capacity: str = 'full', step_size: int = 10):
        """
        Initialize CREPE pitch detector

        Args:
            model_capacity: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
            step_size: Step size in milliseconds between pitch estimates
        """
        self.model_capacity = model_capacity
        self.step_size = step_size

        # Valid model capacities
        self.valid_capacities = ['tiny', 'small', 'medium', 'large', 'full']
        if model_capacity not in self.valid_capacities:
            raise ValueError(f"Invalid model capacity: {model_capacity}. "
                           f"Must be one of {self.valid_capacities}")

        logger.info(f"Initialized CREPE pitch detector: {model_capacity} model, "
                   f"{step_size}ms step size")

    def extract_pitch(self,
                     audio_data: np.ndarray,
                     sample_rate: int = 44100,
                     confidence_threshold: float = 0.85) -> Dict[str, np.ndarray]:
        """
        Extract pitch using CREPE

        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Audio sample rate
            confidence_threshold: Minimum confidence for pitch estimates

        Returns:
            Dictionary containing:
            - time: Time stamps in seconds
            - frequency: Pitch frequencies in Hz
            - confidence: Confidence scores (0-1)
            - voiced_mask: Boolean mask for voiced regions
        """
        try:
            logger.info(f"Starting CREPE pitch extraction on {len(audio_data)} samples "
                       f"at {sample_rate}Hz")

            # Ensure audio is mono and float32
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)

            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Run CREPE
            time, frequency, confidence, _ = crepe.predict(
                audio=audio_data,
                sr=sample_rate,
                model_capacity=self.model_capacity,
                step_size=self.step_size,
                center=True,
                viterbi=True  # Use Viterbi decoding for smoother results
            )

            # Apply confidence threshold
            voiced_mask = confidence >= confidence_threshold

            # Set low-confidence frequencies to 0 (unvoiced)
            frequency_filtered = frequency.copy()
            frequency_filtered[~voiced_mask] = 0.0

            logger.info(f"CREPE extraction complete: "
                       f"{len(time)} frames, "
                       f"{np.sum(voiced_mask)}/{len(voiced_mask)} voiced frames "
                       f"({np.sum(voiced_mask)/len(voiced_mask)*100:.1f}%)")

            return {
                'time': time,
                'frequency': frequency_filtered,
                'frequency_raw': frequency,  # Raw frequencies before confidence filtering
                'confidence': confidence,
                'voiced_mask': voiced_mask
            }

        except Exception as e:
            logger.error(f"CREPE pitch extraction failed: {str(e)}")
            raise

    def extract_pitch_with_vibrato_analysis(self,
                                          audio_data: np.ndarray,
                                          sample_rate: int = 44100) -> Dict[str, np.ndarray]:
        """
        Extract pitch with additional vibrato analysis for preservation

        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Audio sample rate

        Returns:
            Dictionary with pitch data plus vibrato characteristics
        """
        # Get basic pitch extraction
        pitch_data = self.extract_pitch(audio_data, sample_rate)

        # Analyze vibrato characteristics
        vibrato_analysis = self._analyze_vibrato(
            pitch_data['time'],
            pitch_data['frequency'],
            pitch_data['voiced_mask']
        )

        # Combine results
        result = {**pitch_data, **vibrato_analysis}

        return result

    def _analyze_vibrato(self,
                        time: np.ndarray,
                        frequency: np.ndarray,
                        voiced_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze vibrato characteristics for preservation during pitch correction

        Args:
            time: Time stamps
            frequency: Pitch frequencies
            voiced_mask: Voiced/unvoiced mask

        Returns:
            Dictionary with vibrato analysis results
        """
        try:
            # Initialize arrays
            vibrato_rate = np.zeros_like(frequency)
            vibrato_extent = np.zeros_like(frequency)
            vibrato_presence = np.zeros_like(frequency, dtype=bool)

            # Parameters for vibrato detection
            window_size = int(1.0 / (self.step_size / 1000))  # 1 second window
            min_vibrato_rate = 4.0  # Hz
            max_vibrato_rate = 8.0  # Hz
            min_extent = 0.02  # 2% frequency variation

            # Analyze each window
            for i in range(window_size//2, len(frequency) - window_size//2):
                if not voiced_mask[i]:
                    continue

                start_idx = i - window_size//2
                end_idx = i + window_size//2

                window_freq = frequency[start_idx:end_idx]
                window_voiced = voiced_mask[start_idx:end_idx]

                # Only analyze if enough voiced frames
                if np.sum(window_voiced) < window_size * 0.7:
                    continue

                # Extract voiced frequencies only
                voiced_freq = window_freq[window_voiced]
                if len(voiced_freq) < 10:
                    continue

                # Analyze frequency modulation
                freq_mean = np.mean(voiced_freq)
                freq_detrended = voiced_freq - freq_mean

                # Calculate vibrato extent (normalized by mean frequency)
                extent = np.std(freq_detrended) * 2 / freq_mean  # 2 std devs

                if extent > min_extent:
                    # Estimate vibrato rate using autocorrelation
                    rate = self._estimate_vibrato_rate(
                        freq_detrended,
                        self.step_size / 1000
                    )

                    if min_vibrato_rate <= rate <= max_vibrato_rate:
                        vibrato_presence[i] = True
                        vibrato_rate[i] = rate
                        vibrato_extent[i] = extent

            logger.debug(f"Vibrato analysis: "
                        f"{np.sum(vibrato_presence)} frames with vibrato "
                        f"({np.sum(vibrato_presence)/len(vibrato_presence)*100:.1f}%)")

            return {
                'vibrato_rate': vibrato_rate,
                'vibrato_extent': vibrato_extent,
                'vibrato_presence': vibrato_presence
            }

        except Exception as e:
            logger.warning(f"Vibrato analysis failed: {str(e)}")
            return {
                'vibrato_rate': np.zeros_like(frequency),
                'vibrato_extent': np.zeros_like(frequency),
                'vibrato_presence': np.zeros_like(frequency, dtype=bool)
            }

    def _estimate_vibrato_rate(self,
                              freq_modulation: np.ndarray,
                              time_step: float) -> float:
        """
        Estimate vibrato rate using autocorrelation

        Args:
            freq_modulation: Detrended frequency modulation
            time_step: Time step between samples in seconds

        Returns:
            Estimated vibrato rate in Hz
        """
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(freq_modulation, freq_modulation, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in autocorrelation (excluding lag 0)
            if len(autocorr) < 10:
                return 0.0

            # Look for first significant peak
            for lag in range(2, len(autocorr)//3):
                if (autocorr[lag] > autocorr[lag-1] and
                    autocorr[lag] > autocorr[lag+1] and
                    autocorr[lag] > 0.3 * autocorr[0]):  # At least 30% of peak

                    period_seconds = lag * time_step
                    rate = 1.0 / period_seconds
                    return rate

            return 0.0

        except Exception:
            return 0.0


def create_pitch_detector(config: Optional[Dict[str, Any]] = None) -> CreepePitchDetector:
    """
    Factory function to create pitch detector with configuration

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured CreepePitchDetector instance
    """
    if config is None:
        config = {
            'model_capacity': settings.crepe_model,
            'step_size': settings.crepe_step_size
        }

    return CreepePitchDetector(
        model_capacity=config.get('model_capacity', 'full'),
        step_size=config.get('step_size', 10)
    )