"""
Advanced pitch quantization with vibrato preservation for natural sounding pitch correction
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


class MusicalScale:
    """
    Musical scale definitions and utilities
    """

    # Scale definitions (in semitones from root)
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'chromatic': list(range(12)),  # All semitones
        'pentatonic': [0, 2, 4, 7, 9],
    }

    # Note names
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, key: str = 'C', scale_type: str = 'major', reference_freq: float = 440.0):
        """
        Initialize musical scale

        Args:
            key: Root key (C, D, E, F, G, A, B with optional # or b)
            scale_type: Scale type from SCALES
            reference_freq: Reference frequency for A4
        """
        self.key = key
        self.scale_type = scale_type
        self.reference_freq = reference_freq

        # Convert key to semitones from C
        self.root_semitone = self._key_to_semitone(key)

        # Get scale intervals
        if scale_type not in self.SCALES:
            raise ValueError(f"Unknown scale type: {scale_type}")

        self.scale_intervals = self.SCALES[scale_type]

        # Generate all valid pitches (multiple octaves)
        self.valid_pitches = self._generate_valid_pitches()

        logger.info(f"Initialized musical scale: {key} {scale_type}")

    def _key_to_semitone(self, key: str) -> int:
        """Convert key name to semitone offset from C"""
        key = key.upper().replace('B', 'b')  # Normalize flats

        if key in self.NOTE_NAMES:
            return self.NOTE_NAMES.index(key)

        # Handle flats (convert to sharps)
        flat_to_sharp = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
        }

        if key in flat_to_sharp:
            return self.NOTE_NAMES.index(flat_to_sharp[key])

        raise ValueError(f"Invalid key: {key}")

    def _generate_valid_pitches(self, octave_range: Tuple[int, int] = (-2, 6)) -> np.ndarray:
        """
        Generate valid pitch frequencies for multiple octaves

        Args:
            octave_range: Range of octaves relative to A4

        Returns:
            Array of valid frequencies
        """
        # A4 = 440Hz corresponds to semitone 9 in octave 4
        # C4 is 3 semitones below A4
        a4_semitone = 9

        valid_freqs = []

        for octave in range(octave_range[0], octave_range[1] + 1):
            for interval in self.scale_intervals:
                # Calculate semitone index
                semitone = (octave * 12) + self.root_semitone + interval

                # Convert to frequency relative to A4
                semitones_from_a4 = semitone - (4 * 12 + a4_semitone)
                frequency = self.reference_freq * (2 ** (semitones_from_a4 / 12))

                # Keep reasonable frequency range
                if 50 <= frequency <= 2000:
                    valid_freqs.append(frequency)

        return np.array(sorted(valid_freqs))

    def quantize_frequency(self, frequency: float) -> float:
        """
        Quantize a frequency to the nearest valid pitch

        Args:
            frequency: Input frequency

        Returns:
            Quantized frequency
        """
        if frequency <= 0 or len(self.valid_pitches) == 0:
            return 0.0

        # Find nearest valid pitch
        distances = np.abs(self.valid_pitches - frequency)
        nearest_idx = np.argmin(distances)

        return self.valid_pitches[nearest_idx]

    def get_pitch_deviation(self, frequency: float) -> float:
        """
        Get pitch deviation from nearest valid pitch in cents

        Args:
            frequency: Input frequency

        Returns:
            Deviation in cents (100 cents = 1 semitone)
        """
        if frequency <= 0:
            return 0.0

        quantized = self.quantize_frequency(frequency)
        if quantized <= 0:
            return 0.0

        # Calculate deviation in cents
        cents = 1200 * np.log2(frequency / quantized)
        return cents


class VibratoAnalyzer:
    """
    Analyze and preserve vibrato characteristics
    """

    def __init__(self,
                 min_vibrato_rate: float = 3.0,
                 max_vibrato_rate: float = 8.0,
                 min_vibrato_extent: float = 0.01,
                 window_size: float = 1.0):
        """
        Initialize vibrato analyzer

        Args:
            min_vibrato_rate: Minimum vibrato rate in Hz
            max_vibrato_rate: Maximum vibrato rate in Hz
            min_vibrato_extent: Minimum vibrato extent (fraction)
            window_size: Analysis window size in seconds
        """
        self.min_vibrato_rate = min_vibrato_rate
        self.max_vibrato_rate = max_vibrato_rate
        self.min_vibrato_extent = min_vibrato_extent
        self.window_size = window_size

    def analyze_vibrato(self,
                       time: np.ndarray,
                       frequency: np.ndarray,
                       confidence: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Analyze vibrato characteristics in pitch trajectory

        Args:
            time: Time array
            frequency: Frequency array
            confidence: Optional confidence array

        Returns:
            Dictionary with vibrato analysis results
        """
        n_frames = len(frequency)
        vibrato_rate = np.zeros(n_frames)
        vibrato_extent = np.zeros(n_frames)
        vibrato_presence = np.zeros(n_frames, dtype=bool)

        if len(frequency) < 10:
            return {
                'vibrato_rate': vibrato_rate,
                'vibrato_extent': vibrato_extent,
                'vibrato_presence': vibrato_presence
            }

        # Calculate time step
        time_step = np.mean(np.diff(time)) if len(time) > 1 else 0.01

        # Analysis window size in frames
        window_frames = int(self.window_size / time_step)
        window_frames = max(10, min(window_frames, n_frames // 2))

        # Analyze each position
        for i in range(window_frames // 2, n_frames - window_frames // 2):
            start_idx = i - window_frames // 2
            end_idx = i + window_frames // 2

            # Extract window
            window_freq = frequency[start_idx:end_idx]
            window_time = time[start_idx:end_idx]

            # Filter out unvoiced frames
            if confidence is not None:
                window_conf = confidence[start_idx:end_idx]
                voiced_mask = (window_freq > 0) & (window_conf > 0.5)
            else:
                voiced_mask = window_freq > 0

            if np.sum(voiced_mask) < window_frames * 0.6:
                continue

            # Analyze voiced portion
            voiced_freq = window_freq[voiced_mask]
            voiced_time = window_time[voiced_mask]

            if len(voiced_freq) < 5:
                continue

            # Detect vibrato
            vibrato_info = self._detect_vibrato_in_window(voiced_time, voiced_freq)

            if vibrato_info['detected']:
                vibrato_presence[i] = True
                vibrato_rate[i] = vibrato_info['rate']
                vibrato_extent[i] = vibrato_info['extent']

        logger.debug(f"Vibrato analysis: {np.sum(vibrato_presence)} frames with vibrato")

        return {
            'vibrato_rate': vibrato_rate,
            'vibrato_extent': vibrato_extent,
            'vibrato_presence': vibrato_presence
        }

    def _detect_vibrato_in_window(self, time: np.ndarray, frequency: np.ndarray) -> Dict[str, Any]:
        """
        Detect vibrato in a single window

        Args:
            time: Time array for window
            frequency: Frequency array for window

        Returns:
            Dictionary with detection results
        """
        if len(frequency) < 5:
            return {'detected': False, 'rate': 0.0, 'extent': 0.0}

        # Detrend frequency (remove overall pitch trend)
        mean_freq = np.mean(frequency)
        detrended = frequency - mean_freq

        # Calculate vibrato extent
        extent = np.std(detrended) / mean_freq

        if extent < self.min_vibrato_extent:
            return {'detected': False, 'rate': 0.0, 'extent': extent}

        # Estimate vibrato rate using zero-crossings
        rate = self._estimate_vibrato_rate_zero_crossings(time, detrended)

        if self.min_vibrato_rate <= rate <= self.max_vibrato_rate:
            return {'detected': True, 'rate': rate, 'extent': extent}

        return {'detected': False, 'rate': rate, 'extent': extent}

    def _estimate_vibrato_rate_zero_crossings(self, time: np.ndarray, signal: np.ndarray) -> float:
        """
        Estimate vibrato rate using zero-crossing analysis

        Args:
            time: Time array
            signal: Detrended frequency signal

        Returns:
            Estimated rate in Hz
        """
        if len(signal) < 3:
            return 0.0

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]

        if len(zero_crossings) < 2:
            return 0.0

        # Calculate time between zero crossings
        crossing_times = time[zero_crossings]
        intervals = np.diff(crossing_times)

        if len(intervals) == 0:
            return 0.0

        # Vibrato period is twice the average interval (up-down cycle)
        avg_interval = np.median(intervals)
        vibrato_period = 2 * avg_interval

        if vibrato_period <= 0:
            return 0.0

        return 1.0 / vibrato_period


class PitchQuantizer:
    """
    Advanced pitch quantizer with vibrato preservation
    """

    def __init__(self,
                 musical_scale: MusicalScale,
                 vibrato_analyzer: Optional[VibratoAnalyzer] = None):
        """
        Initialize pitch quantizer

        Args:
            musical_scale: Musical scale for quantization
            vibrato_analyzer: Optional vibrato analyzer
        """
        self.scale = musical_scale
        self.vibrato_analyzer = vibrato_analyzer or VibratoAnalyzer()

        logger.info("Initialized pitch quantizer")

    def quantize_pitch_trajectory(self,
                                 time: np.ndarray,
                                 frequency: np.ndarray,
                                 confidence: Optional[np.ndarray] = None,
                                 correction_strength: float = 1.0,
                                 preserve_vibrato: bool = True,
                                 smoothing_factor: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Quantize pitch trajectory with vibrato preservation

        Args:
            time: Time array
            frequency: Original frequency trajectory
            confidence: Optional confidence scores
            correction_strength: Correction strength (0-1)
            preserve_vibrato: Whether to preserve vibrato
            smoothing_factor: Temporal smoothing factor

        Returns:
            Dictionary with quantized pitch data
        """
        try:
            # Initialize output
            quantized_freq = np.zeros_like(frequency)
            vibrato_info = None

            # Find voiced regions
            if confidence is not None:
                voiced_mask = (frequency > 0) & (confidence > 0.5)
            else:
                voiced_mask = frequency > 0

            if np.sum(voiced_mask) == 0:
                logger.warning("No voiced frames found")
                return {
                    'quantized_frequency': quantized_freq,
                    'correction_applied': np.zeros_like(frequency, dtype=bool),
                    'vibrato_preserved': np.zeros_like(frequency, dtype=bool),
                    'pitch_deviation': np.zeros_like(frequency)
                }

            # Analyze vibrato if preservation is enabled
            if preserve_vibrato:
                vibrato_info = self.vibrato_analyzer.analyze_vibrato(
                    time, frequency, confidence
                )

            # Process each voiced segment
            voiced_segments = self._find_voiced_segments(voiced_mask)

            correction_applied = np.zeros_like(frequency, dtype=bool)
            vibrato_preserved = np.zeros_like(frequency, dtype=bool)
            pitch_deviation = np.zeros_like(frequency)

            for start_idx, end_idx in voiced_segments:
                segment_result = self._quantize_segment(
                    time[start_idx:end_idx],
                    frequency[start_idx:end_idx],
                    confidence[start_idx:end_idx] if confidence is not None else None,
                    vibrato_info,
                    start_idx,
                    correction_strength,
                    preserve_vibrato
                )

                quantized_freq[start_idx:end_idx] = segment_result['frequency']
                correction_applied[start_idx:end_idx] = segment_result['corrected']
                vibrato_preserved[start_idx:end_idx] = segment_result['vibrato_preserved']
                pitch_deviation[start_idx:end_idx] = segment_result['deviation']

            # Apply smoothing
            if smoothing_factor > 0:
                quantized_freq = self._apply_temporal_smoothing(
                    quantized_freq, voiced_mask, smoothing_factor
                )

            logger.info(f"Pitch quantization complete: "
                       f"{np.sum(correction_applied)} frames corrected, "
                       f"{np.sum(vibrato_preserved)} frames with preserved vibrato")

            return {
                'quantized_frequency': quantized_freq,
                'correction_applied': correction_applied,
                'vibrato_preserved': vibrato_preserved,
                'pitch_deviation': pitch_deviation,
                'vibrato_analysis': vibrato_info
            }

        except Exception as e:
            logger.error(f"Pitch quantization failed: {str(e)}")
            raise

    def _find_voiced_segments(self, voiced_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find continuous voiced segments

        Args:
            voiced_mask: Boolean mask of voiced frames

        Returns:
            List of (start, end) indices for voiced segments
        """
        segments = []
        in_segment = False
        start_idx = 0

        for i, voiced in enumerate(voiced_mask):
            if voiced and not in_segment:
                start_idx = i
                in_segment = True
            elif not voiced and in_segment:
                segments.append((start_idx, i))
                in_segment = False

        # Handle segment ending at the last frame
        if in_segment:
            segments.append((start_idx, len(voiced_mask)))

        return segments

    def _quantize_segment(self,
                         time: np.ndarray,
                         frequency: np.ndarray,
                         confidence: Optional[np.ndarray],
                         vibrato_info: Optional[Dict[str, np.ndarray]],
                         global_start_idx: int,
                         correction_strength: float,
                         preserve_vibrato: bool) -> Dict[str, np.ndarray]:
        """
        Quantize a single voiced segment

        Args:
            time: Time array for segment
            frequency: Frequency array for segment
            confidence: Optional confidence array for segment
            vibrato_info: Vibrato analysis results
            global_start_idx: Starting index in global arrays
            correction_strength: Correction strength
            preserve_vibrato: Whether to preserve vibrato

        Returns:
            Dictionary with segment quantization results
        """
        n_frames = len(frequency)
        quantized_freq = np.zeros_like(frequency)
        corrected_mask = np.zeros(n_frames, dtype=bool)
        vibrato_preserved_mask = np.zeros(n_frames, dtype=bool)
        deviation = np.zeros_like(frequency)

        for i, freq in enumerate(frequency):
            global_idx = global_start_idx + i

            if freq <= 0:
                continue

            # Get target frequency
            target_freq = self.scale.quantize_frequency(freq)
            deviation[i] = self.scale.get_pitch_deviation(freq)

            # Check if correction is needed
            if abs(deviation[i]) < 5.0:  # Less than 5 cents deviation
                quantized_freq[i] = freq  # Keep original
                continue

            # Apply vibrato preservation
            if (preserve_vibrato and vibrato_info is not None and
                global_idx < len(vibrato_info['vibrato_presence']) and
                vibrato_info['vibrato_presence'][global_idx]):

                # Preserve vibrato by maintaining oscillation around target
                vibrato_preserved_mask[i] = True
                quantized_freq[i] = self._preserve_vibrato(
                    freq, target_freq, correction_strength
                )
            else:
                # Standard quantization
                quantized_freq[i] = (
                    (1 - correction_strength) * freq +
                    correction_strength * target_freq
                )

            corrected_mask[i] = True

        return {
            'frequency': quantized_freq,
            'corrected': corrected_mask,
            'vibrato_preserved': vibrato_preserved_mask,
            'deviation': deviation
        }

    def _preserve_vibrato(self, original_freq: float, target_freq: float, strength: float) -> float:
        """
        Preserve vibrato while correcting pitch

        Args:
            original_freq: Original frequency
            target_freq: Target quantized frequency
            strength: Correction strength

        Returns:
            Corrected frequency with preserved vibrato
        """
        # Calculate the vibrato deviation from a "straight" pitch
        # This is a simplified approach - in practice, you'd want to
        # track the vibrato oscillation more precisely

        freq_ratio = original_freq / target_freq if target_freq > 0 else 1.0

        # If the ratio is close to 1, preserve the exact relationship
        if 0.98 <= freq_ratio <= 1.02:
            return original_freq

        # Otherwise, apply correction but maintain the vibrato character
        # by preserving small deviations
        deviation = original_freq - target_freq
        small_deviation = deviation * 0.3  # Preserve 30% of the deviation as "vibrato"

        corrected = (1 - strength) * original_freq + strength * (target_freq + small_deviation)
        return corrected

    def _apply_temporal_smoothing(self,
                                 frequency: np.ndarray,
                                 voiced_mask: np.ndarray,
                                 smoothing_factor: float) -> np.ndarray:
        """
        Apply temporal smoothing to reduce pitch jumps

        Args:
            frequency: Frequency array
            voiced_mask: Voiced regions mask
            smoothing_factor: Smoothing strength

        Returns:
            Smoothed frequency array
        """
        if smoothing_factor <= 0:
            return frequency

        smoothed = frequency.copy()

        # Apply smoothing only to voiced regions
        voiced_segments = self._find_voiced_segments(voiced_mask)

        for start_idx, end_idx in voiced_segments:
            segment_freq = frequency[start_idx:end_idx]

            if len(segment_freq) < 3:
                continue

            # Apply Savitzky-Golay filter for smooth transitions
            window_length = min(7, len(segment_freq) if len(segment_freq) % 2 == 1 else len(segment_freq) - 1)
            if window_length >= 3:
                try:
                    smoothed_segment = signal.savgol_filter(
                        segment_freq,
                        window_length,
                        polyorder=2,
                        mode='interp'
                    )

                    # Blend with original based on smoothing factor
                    smoothed[start_idx:end_idx] = (
                        (1 - smoothing_factor) * segment_freq +
                        smoothing_factor * smoothed_segment
                    )
                except:
                    # Fallback to simple moving average
                    kernel_size = min(5, len(segment_freq))
                    kernel = np.ones(kernel_size) / kernel_size
                    smoothed_segment = np.convolve(segment_freq, kernel, mode='same')
                    smoothed[start_idx:end_idx] = (
                        (1 - smoothing_factor) * segment_freq +
                        smoothing_factor * smoothed_segment
                    )

        return smoothed