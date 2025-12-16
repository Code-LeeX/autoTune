"""
Audio synthesis for pitch correction using PyWorld and advanced signal processing
"""
import numpy as np
import pyworld as pw
import librosa
from typing import Dict, Tuple, Optional, Any
import logging
from scipy import signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class WorldAudioSynthesizer:
    """
    High-quality audio synthesis using WORLD vocoder for pitch correction
    Preserves formants and natural voice characteristics
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 frame_period: float = 5.0,
                 f0_floor: float = 71.0,
                 f0_ceil: float = 800.0):
        """
        Initialize World-based audio synthesizer

        Args:
            sample_rate: Audio sample rate
            frame_period: Frame period in milliseconds
            f0_floor: Minimum fundamental frequency
            f0_ceil: Maximum fundamental frequency
        """
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil

        logger.info(f"Initialized World synthesizer: {sample_rate}Hz, "
                   f"frame_period={frame_period}ms")

    def analyze_audio(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze audio using WORLD vocoder

        Args:
            audio: Input audio signal

        Returns:
            Dictionary containing f0, sp (spectrum), ap (aperiodicity), and time_axis
        """
        try:
            # Ensure audio is float64 for WORLD
            audio = audio.astype(np.float64)

            # Extract fundamental frequency
            f0, time_axis = pw.harvest(
                audio,
                self.sample_rate,
                frame_period=self.frame_period,
                f0_floor=self.f0_floor,
                f0_ceil=self.f0_ceil
            )

            # Refine F0 estimation
            f0 = pw.stonemask(audio, f0, time_axis, self.sample_rate)

            # Extract spectral envelope
            sp = pw.cheaptrick(audio, f0, time_axis, self.sample_rate)

            # Extract aperiodicity
            ap = pw.d4c(audio, f0, time_axis, self.sample_rate)

            logger.info(f"Audio analysis complete: {len(f0)} frames, "
                       f"f0 range: {np.min(f0[f0>0]):.1f}-{np.max(f0):.1f}Hz")

            return {
                'f0': f0,
                'sp': sp,
                'ap': ap,
                'time_axis': time_axis
            }

        except Exception as e:
            logger.error(f"World analysis failed: {str(e)}")
            raise

    def synthesize_audio(self,
                        f0: np.ndarray,
                        sp: np.ndarray,
                        ap: np.ndarray,
                        time_axis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Synthesize audio from WORLD parameters

        Args:
            f0: Fundamental frequency array
            sp: Spectral envelope
            ap: Aperiodicity
            time_axis: Time axis (optional)

        Returns:
            Synthesized audio signal
        """
        try:
            if time_axis is None:
                time_axis = np.arange(len(f0)) * self.frame_period / 1000.0

            # Synthesize using WORLD
            synthesized = pw.synthesize(
                f0.astype(np.float64),
                sp.astype(np.float64),
                ap.astype(np.float64),
                self.sample_rate,
                self.frame_period
            )

            logger.info(f"Audio synthesis complete: {len(synthesized)} samples")
            return synthesized.astype(np.float32)

        except Exception as e:
            logger.error(f"World synthesis failed: {str(e)}")
            raise

    def pitch_shift_with_formant_preservation(self,
                                            audio: np.ndarray,
                                            original_f0: np.ndarray,
                                            target_f0: np.ndarray,
                                            formant_shift_ratio: float = 1.0) -> np.ndarray:
        """
        Pitch shift with formant preservation

        Args:
            audio: Original audio signal
            original_f0: Original f0 trajectory
            target_f0: Target f0 trajectory
            formant_shift_ratio: Ratio for formant shifting (1.0 = no shift)

        Returns:
            Pitch-corrected audio
        """
        try:
            # Analyze original audio
            analysis = self.analyze_audio(audio)

            # Interpolate target f0 to match analysis frames
            if len(target_f0) != len(analysis['f0']):
                target_f0 = self._resample_f0_trajectory(
                    target_f0,
                    len(analysis['f0'])
                )

            # Preserve formants by adjusting spectral envelope
            if formant_shift_ratio != 1.0:
                modified_sp = self._shift_formants(
                    analysis['sp'],
                    formant_shift_ratio
                )
            else:
                modified_sp = analysis['sp']

            # Apply spectral envelope smoothing to reduce artifacts
            modified_sp = self._smooth_spectral_envelope(modified_sp)

            # Synthesize with new f0
            corrected_audio = self.synthesize_audio(
                target_f0,
                modified_sp,
                analysis['ap'],
                analysis['time_axis']
            )

            return corrected_audio

        except Exception as e:
            logger.error(f"Pitch shift with formant preservation failed: {str(e)}")
            raise

    def _resample_f0_trajectory(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample f0 trajectory to target length

        Args:
            f0: Original f0 trajectory
            target_length: Target number of frames

        Returns:
            Resampled f0 trajectory
        """
        if len(f0) == target_length:
            return f0

        # Create interpolation function
        x_old = np.linspace(0, 1, len(f0))
        x_new = np.linspace(0, 1, target_length)

        # Handle voiced/unvoiced regions carefully
        voiced_mask = f0 > 0

        if np.sum(voiced_mask) == 0:
            return np.zeros(target_length)

        # Interpolate only voiced regions
        f_interp = interp1d(
            x_old[voiced_mask],
            f0[voiced_mask],
            kind='linear',
            bounds_error=False,
            fill_value=0
        )

        resampled_f0 = f_interp(x_new)
        resampled_f0[np.isnan(resampled_f0)] = 0

        return resampled_f0

    def _shift_formants(self, sp: np.ndarray, shift_ratio: float) -> np.ndarray:
        """
        Shift formants in spectral envelope

        Args:
            sp: Original spectral envelope
            shift_ratio: Formant shift ratio

        Returns:
            Modified spectral envelope
        """
        if shift_ratio == 1.0:
            return sp

        # Convert to log scale
        log_sp = np.log(sp + 1e-7)

        # Frequency axis
        n_bins = sp.shape[1]
        freq_axis = np.linspace(0, self.sample_rate/2, n_bins)

        shifted_sp = np.zeros_like(sp)

        for i in range(sp.shape[0]):
            # Shift frequencies
            shifted_freq = freq_axis / shift_ratio

            # Interpolate spectrum at shifted frequencies
            interp_func = interp1d(
                freq_axis,
                log_sp[i],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            shifted_log_sp = interp_func(shifted_freq)
            shifted_sp[i] = np.exp(shifted_log_sp)

        return shifted_sp

    def _smooth_spectral_envelope(self, sp: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
        """
        Apply temporal smoothing to spectral envelope

        Args:
            sp: Spectral envelope
            smoothing_factor: Smoothing strength (0-1)

        Returns:
            Smoothed spectral envelope
        """
        if smoothing_factor <= 0:
            return sp

        # Apply median filter for smoothing
        kernel_size = max(3, int(smoothing_factor * 20))
        if kernel_size % 2 == 0:
            kernel_size += 1

        smoothed_sp = np.zeros_like(sp)

        for freq_bin in range(sp.shape[1]):
            smoothed_sp[:, freq_bin] = signal.medfilt(
                sp[:, freq_bin],
                kernel_size=kernel_size
            )

        return smoothed_sp


class PitchCorrectionEngine:
    """
    Complete pitch correction engine combining CREPE and World
    """

    def __init__(self,
                 sample_rate: int = 44100,
                 frame_period: float = 5.0):
        """
        Initialize pitch correction engine

        Args:
            sample_rate: Audio sample rate
            frame_period: Analysis frame period in ms
        """
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.synthesizer = WorldAudioSynthesizer(
            sample_rate=sample_rate,
            frame_period=frame_period
        )

        logger.info("Initialized pitch correction engine")

    def correct_pitch(self,
                     audio: np.ndarray,
                     target_f0: np.ndarray,
                     correction_strength: float = 1.0,
                     preserve_formants: bool = True,
                     smoothing_factor: float = 0.1) -> Dict[str, Any]:
        """
        Apply pitch correction to audio

        Args:
            audio: Input audio signal
            target_f0: Target f0 trajectory
            correction_strength: Correction strength (0-1)
            preserve_formants: Whether to preserve formants
            smoothing_factor: Spectral smoothing factor

        Returns:
            Dictionary with corrected audio and analysis data
        """
        try:
            # Analyze original audio
            analysis = self.synthesizer.analyze_audio(audio)
            original_f0 = analysis['f0']

            # Blend original and target f0 based on correction strength
            if correction_strength < 1.0:
                blended_f0 = self._blend_f0_trajectories(
                    original_f0,
                    target_f0,
                    correction_strength
                )
            else:
                blended_f0 = target_f0

            # Apply formant preservation
            formant_shift_ratio = 1.0
            if preserve_formants:
                # Calculate average pitch shift to compensate formants
                voiced_orig = original_f0[original_f0 > 0]
                voiced_target = blended_f0[blended_f0 > 0]

                if len(voiced_orig) > 0 and len(voiced_target) > 0:
                    avg_shift = np.mean(voiced_target) / np.mean(voiced_orig)
                    formant_shift_ratio = 1.0 / avg_shift

            # Synthesize corrected audio
            corrected_audio = self.synthesizer.pitch_shift_with_formant_preservation(
                audio,
                original_f0,
                blended_f0,
                formant_shift_ratio
            )

            # Apply post-processing
            corrected_audio = self._post_process_audio(
                corrected_audio,
                smoothing_factor
            )

            return {
                'corrected_audio': corrected_audio,
                'original_f0': original_f0,
                'target_f0': target_f0,
                'corrected_f0': blended_f0,
                'analysis': analysis
            }

        except Exception as e:
            logger.error(f"Pitch correction failed: {str(e)}")
            raise

    def _blend_f0_trajectories(self,
                              original_f0: np.ndarray,
                              target_f0: np.ndarray,
                              strength: float) -> np.ndarray:
        """
        Blend original and target f0 trajectories

        Args:
            original_f0: Original f0
            target_f0: Target f0
            strength: Blending strength

        Returns:
            Blended f0 trajectory
        """
        # Ensure same length
        if len(target_f0) != len(original_f0):
            target_f0 = self.synthesizer._resample_f0_trajectory(
                target_f0, len(original_f0)
            )

        # Blend only voiced regions
        voiced_mask = original_f0 > 0
        blended_f0 = original_f0.copy()

        blended_f0[voiced_mask] = (
            (1 - strength) * original_f0[voiced_mask] +
            strength * target_f0[voiced_mask]
        )

        return blended_f0

    def _post_process_audio(self,
                           audio: np.ndarray,
                           smoothing_factor: float) -> np.ndarray:
        """
        Post-process corrected audio

        Args:
            audio: Input audio
            smoothing_factor: Smoothing strength

        Returns:
            Post-processed audio
        """
        # Apply light smoothing to reduce artifacts
        if smoothing_factor > 0:
            # Simple moving average filter
            window_size = max(3, int(smoothing_factor * 100))
            if window_size % 2 == 0:
                window_size += 1

            audio = signal.savgol_filter(
                audio,
                window_size,
                polyorder=2,
                mode='interp'
            )

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        return audio