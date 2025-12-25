"""
Professional Audio Mixing Engine using Pedalboard
Implements 5 core mixing effects: NoiseGate, HighPass, Compressor, EQ, Reverb
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

try:
    from pedalboard import (
        Pedalboard,
        NoiseGate,
        HighpassFilter,
        Compressor,
        PeakFilter,
        Reverb,
        Convolution
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    # Fallback for environments where pedalboard is not available
    PEDALBOARD_AVAILABLE = False
    logging.warning("Pedalboard not available. Mixing features will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class MixingParams:
    """Audio mixing parameters for all effects"""

    # Noise Gate / De-breath
    noise_gate_enabled: bool = True
    noise_gate_threshold_db: float = -35.0    # dB - sounds below this are gated
    noise_gate_ratio: float = 8.0             # compression ratio for gating
    noise_gate_attack_ms: float = 1.0         # attack time in ms
    noise_gate_release_ms: float = 150.0      # release time in ms

    # High-pass Filter (Subtractive EQ)
    highpass_enabled: bool = True
    highpass_frequency_hz: float = 80.0       # Hz - cut everything below this

    # Compressor (Dynamic Control)
    compressor_enabled: bool = True
    compressor_threshold_db: float = -18.0    # dB threshold
    compressor_ratio: float = 3.0             # compression ratio
    compressor_attack_ms: float = 10.0        # attack time in ms
    compressor_release_ms: float = 100.0      # release time in ms

    # Additive EQ - Multi-band for fuller sound
    eq_enabled: bool = True

    # Low-frequency EQ (Warmth/Body) - 100-200Hz
    eq_low_enabled: bool = True
    eq_low_frequency_hz: float = 150.0        # Hz - low frequency boost for body
    eq_low_gain_db: float = 1.0               # dB - low freq gain
    eq_low_q: float = 1.0                     # Q factor

    # Low-mid EQ (Fullness) - 300-500Hz
    eq_low_mid_enabled: bool = True
    eq_low_mid_frequency_hz: float = 400.0    # Hz - low-mid frequency for fullness
    eq_low_mid_gain_db: float = 1.5           # dB - low-mid gain
    eq_low_mid_q: float = 1.2                 # Q factor

    # Presence EQ (Clarity) - 2-4kHz
    eq_presence_frequency_hz: float = 2800.0  # Hz - presence frequency
    eq_presence_gain_db: float = 1.5          # dB - presence gain
    eq_presence_q: float = 1.2                # Q factor

    # High-frequency shelf EQ (Air) - 8kHz+
    eq_high_enabled: bool = True
    eq_high_frequency_hz: float = 8000.0      # Hz - high shelf frequency
    eq_high_gain_db: float = 0.8              # dB - air/sparkle gain
    eq_high_q: float = 0.7                    # Q factor for shelf

    # Reverb (Space/Depth)
    reverb_enabled: bool = True
    reverb_type: str = "algorithm"            # "algorithm" or "convolution"
    reverb_room_size: float = 0.2             # 0.0 to 1.0
    reverb_damping: float = 0.6               # 0.0 to 1.0
    reverb_wet_level: float = 0.12            # 0.0 to 1.0 (dry/wet mix)
    reverb_width: float = 1.0                 # stereo width


@dataclass
class ProcessingMode:
    """Audio processing mode configuration"""
    pitch_correction: bool = True
    mixing: bool = True
    processing_order: str = "pitch_first"     # "pitch_first", "mix_first", "parallel"


class AudioMixingEngine:
    """
    Professional audio mixing engine using Pedalboard

    Implements the complete mixing chain:
    1. Noise Gate - Remove breath sounds and background noise
    2. High-pass Filter - Remove rumble and low-frequency noise
    3. Compressor - Even out dynamic range for consistent volume
    4. Additive EQ - Add presence and air to vocals
    5. Reverb - Add spatial depth and warmth
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.pedalboard: Optional[Pedalboard] = None

        if not PEDALBOARD_AVAILABLE:
            raise ImportError(
                "Pedalboard library is not available. "
                "Please install with: pip install pedalboard>=0.9.0"
            )

        # Default professional vocal mixing chain
        self.default_params = MixingParams()
        self._create_default_chain()

        logger.info(f"AudioMixingEngine initialized with sample rate: {sample_rate}Hz")

    def _create_default_chain(self) -> None:
        """Create the default professional vocal mixing chain"""
        self.pedalboard = Pedalboard([
            # 1. Noise Gate - Clean up breath sounds and background noise
            NoiseGate(
                threshold_db=self.default_params.noise_gate_threshold_db,
                ratio=self.default_params.noise_gate_ratio,
                attack_ms=self.default_params.noise_gate_attack_ms,
                release_ms=self.default_params.noise_gate_release_ms
            ),

            # 2. High-pass Filter - Remove low-frequency rumble
            HighpassFilter(
                cutoff_frequency_hz=self.default_params.highpass_frequency_hz
            ),

            # 3. Compressor - Even out dynamics for professional sound
            Compressor(
                threshold_db=self.default_params.compressor_threshold_db,
                ratio=self.default_params.compressor_ratio,
                attack_ms=self.default_params.compressor_attack_ms,
                release_ms=self.default_params.compressor_release_ms
            ),

            # 4. Additive EQ - Add presence and clarity (using presence freq as default)
            PeakFilter(
                cutoff_frequency_hz=self.default_params.eq_presence_frequency_hz,
                gain_db=self.default_params.eq_presence_gain_db,
                q=self.default_params.eq_presence_q
            ),

            # 5. Reverb - Add spatial depth and warmth
            Reverb(
                room_size=self.default_params.reverb_room_size,
                damping=self.default_params.reverb_damping,
                wet_level=self.default_params.reverb_wet_level,
                width=self.default_params.reverb_width
            )
        ])

        logger.info("Default professional vocal mixing chain created")

    def create_custom_chain(self, params: MixingParams) -> Pedalboard:
        """
        Create a custom mixing chain based on provided parameters

        Args:
            params: MixingParams object with custom settings

        Returns:
            Configured Pedalboard with enabled effects
        """
        effects = []

        # Add Noise Gate if enabled
        if params.noise_gate_enabled:
            effects.append(
                NoiseGate(
                    threshold_db=params.noise_gate_threshold_db,
                    ratio=params.noise_gate_ratio,
                    attack_ms=params.noise_gate_attack_ms,
                    release_ms=params.noise_gate_release_ms
                )
            )
            logger.debug(f"Added NoiseGate: {params.noise_gate_threshold_db}dB threshold")

        # Add High-pass Filter if enabled
        if params.highpass_enabled:
            effects.append(
                HighpassFilter(cutoff_frequency_hz=params.highpass_frequency_hz)
            )
            logger.debug(f"Added HighpassFilter: {params.highpass_frequency_hz}Hz cutoff")

        # Add Compressor if enabled
        if params.compressor_enabled:
            effects.append(
                Compressor(
                    threshold_db=params.compressor_threshold_db,
                    ratio=params.compressor_ratio,
                    attack_ms=params.compressor_attack_ms,
                    release_ms=params.compressor_release_ms
                )
            )
            logger.debug(f"Added Compressor: {params.compressor_threshold_db}dB, {params.compressor_ratio}:1")

        # Add Multi-band EQ if enabled
        if params.eq_enabled:
            # Low-frequency EQ for body/warmth (100-200Hz)
            if params.eq_low_enabled and params.eq_low_gain_db != 0:
                effects.append(
                    PeakFilter(
                        cutoff_frequency_hz=params.eq_low_frequency_hz,
                        gain_db=params.eq_low_gain_db,
                        q=params.eq_low_q
                    )
                )
                logger.debug(f"Added Low EQ: {params.eq_low_frequency_hz}Hz +{params.eq_low_gain_db}dB")

            # Low-mid EQ for fullness (300-500Hz)
            if params.eq_low_mid_enabled and params.eq_low_mid_gain_db != 0:
                effects.append(
                    PeakFilter(
                        cutoff_frequency_hz=params.eq_low_mid_frequency_hz,
                        gain_db=params.eq_low_mid_gain_db,
                        q=params.eq_low_mid_q
                    )
                )
                logger.debug(f"Added Low-Mid EQ: {params.eq_low_mid_frequency_hz}Hz +{params.eq_low_mid_gain_db}dB")

            # Presence EQ for clarity (2-4kHz)
            if params.eq_presence_gain_db != 0:
                effects.append(
                    PeakFilter(
                        cutoff_frequency_hz=params.eq_presence_frequency_hz,
                        gain_db=params.eq_presence_gain_db,
                        q=params.eq_presence_q
                    )
                )
                logger.debug(f"Added Presence EQ: {params.eq_presence_frequency_hz}Hz +{params.eq_presence_gain_db}dB")

            # High-frequency shelf EQ for air (8kHz+)
            if params.eq_high_enabled and params.eq_high_gain_db != 0:
                try:
                    # Try to use HighShelfFilter for more natural high-freq boost
                    from pedalboard import HighShelfFilter
                    effects.append(
                        HighShelfFilter(
                            cutoff_frequency_hz=params.eq_high_frequency_hz,
                            gain_db=params.eq_high_gain_db,
                            q=params.eq_high_q
                        )
                    )
                    logger.debug(f"Added High Shelf EQ: {params.eq_high_frequency_hz}Hz +{params.eq_high_gain_db}dB")
                except ImportError:
                    # Fallback to PeakFilter if HighShelfFilter is not available
                    effects.append(
                        PeakFilter(
                            cutoff_frequency_hz=params.eq_high_frequency_hz,
                            gain_db=params.eq_high_gain_db,
                            q=params.eq_high_q
                        )
                    )
                    logger.debug(f"Added High EQ (Peak): {params.eq_high_frequency_hz}Hz +{params.eq_high_gain_db}dB")

        # Add Reverb if enabled
        if params.reverb_enabled:
            if params.reverb_type == "convolution":
                # TODO: Implement convolution reverb with IR files
                logger.warning("Convolution reverb not yet implemented, using algorithm reverb")

            effects.append(
                Reverb(
                    room_size=params.reverb_room_size,
                    damping=params.reverb_damping,
                    wet_level=params.reverb_wet_level,
                    width=params.reverb_width
                )
            )
            logger.debug(f"Added Reverb: room={params.reverb_room_size}, wet={params.reverb_wet_level}")

        return Pedalboard(effects)

    async def process_audio(
        self,
        audio_data: np.ndarray,
        params: Optional[MixingParams] = None
    ) -> np.ndarray:
        """
        Process audio with mixing effects

        Args:
            audio_data: Input audio as numpy array (samples, channels) or (samples,)
            params: Custom mixing parameters, uses defaults if None

        Returns:
            Processed audio as numpy array
        """
        if not PEDALBOARD_AVAILABLE:
            logger.warning("Pedalboard not available, returning original audio")
            return audio_data

        try:
            # Ensure audio is in the right format
            if audio_data.ndim == 1:
                # Convert mono to stereo for processing
                audio_stereo = np.stack([audio_data, audio_data], axis=-1)
            elif audio_data.ndim == 2:
                audio_stereo = audio_data
            else:
                raise ValueError(f"Unsupported audio shape: {audio_data.shape}")

            # Use custom params or default chain
            if params is not None:
                pedalboard = self.create_custom_chain(params)
            else:
                pedalboard = self.pedalboard
                params = self.default_params

            # Process audio through the mixing chain
            logger.info(f"Processing audio with {len(pedalboard)} effects")
            processed_audio = pedalboard(
                audio_stereo.astype(np.float32),
                sample_rate=self.sample_rate
            )

            # Convert back to original format
            if audio_data.ndim == 1:
                # Convert back to mono by taking left channel
                processed_audio = processed_audio[:, 0]

            logger.info("Audio mixing processing completed successfully")
            return processed_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Error processing audio with mixing effects: {str(e)}")
            # Return original audio if processing fails
            return audio_data

    def get_chain_info(self, params: Optional[MixingParams] = None) -> Dict[str, Any]:
        """
        Get information about the current or custom mixing chain

        Args:
            params: Optional custom parameters to analyze

        Returns:
            Dictionary with chain information
        """
        if params is None:
            params = self.default_params
            chain = self.pedalboard
        else:
            chain = self.create_custom_chain(params)

        enabled_effects = []

        if params.noise_gate_enabled:
            enabled_effects.append({
                "name": "Noise Gate",
                "type": "dynamics",
                "settings": {
                    "threshold_db": params.noise_gate_threshold_db,
                    "ratio": params.noise_gate_ratio,
                    "release_ms": params.noise_gate_release_ms
                }
            })

        if params.highpass_enabled:
            enabled_effects.append({
                "name": "High-pass Filter",
                "type": "filter",
                "settings": {
                    "cutoff_hz": params.highpass_frequency_hz
                }
            })

        if params.compressor_enabled:
            enabled_effects.append({
                "name": "Compressor",
                "type": "dynamics",
                "settings": {
                    "threshold_db": params.compressor_threshold_db,
                    "ratio": params.compressor_ratio,
                    "attack_ms": params.compressor_attack_ms,
                    "release_ms": params.compressor_release_ms
                }
            })

        if params.eq_enabled:
            # Multi-band EQ information
            eq_bands = []

            if params.eq_low_enabled and params.eq_low_gain_db != 0:
                eq_bands.append({
                    "band": "Low",
                    "frequency_hz": params.eq_low_frequency_hz,
                    "gain_db": params.eq_low_gain_db,
                    "q": params.eq_low_q
                })

            if params.eq_low_mid_enabled and params.eq_low_mid_gain_db != 0:
                eq_bands.append({
                    "band": "Low-Mid",
                    "frequency_hz": params.eq_low_mid_frequency_hz,
                    "gain_db": params.eq_low_mid_gain_db,
                    "q": params.eq_low_mid_q
                })

            if params.eq_presence_gain_db != 0:
                eq_bands.append({
                    "band": "Presence",
                    "frequency_hz": params.eq_presence_frequency_hz,
                    "gain_db": params.eq_presence_gain_db,
                    "q": params.eq_presence_q
                })

            if params.eq_high_enabled and params.eq_high_gain_db != 0:
                eq_bands.append({
                    "band": "High",
                    "frequency_hz": params.eq_high_frequency_hz,
                    "gain_db": params.eq_high_gain_db,
                    "q": params.eq_high_q
                })

            enabled_effects.append({
                "name": "Multi-band EQ",
                "type": "eq",
                "settings": {
                    "bands": eq_bands,
                    "total_bands": len(eq_bands)
                }
            })

        if params.reverb_enabled:
            enabled_effects.append({
                "name": "Reverb",
                "type": "spatial",
                "settings": {
                    "room_size": params.reverb_room_size,
                    "damping": params.reverb_damping,
                    "wet_level": params.reverb_wet_level
                }
            })

        return {
            "total_effects": len(enabled_effects),
            "enabled_effects": enabled_effects,
            "sample_rate": self.sample_rate,
            "processing_available": PEDALBOARD_AVAILABLE
        }


# Note: Presets are now loaded from JSON configuration files
# See /config/mixing_presets.json for detailed preset definitions

# Temporary empty dictionary to prevent import errors
# Presets should be loaded via the preset manager instead
MIXING_PRESETS = {}


def get_mixing_preset(preset_name: str) -> Optional[MixingParams]:
    """
    Get a predefined mixing preset by name

    Args:
        preset_name: Name of the preset to retrieve

    Returns:
        MixingParams object or None if preset not found
    """
    return MIXING_PRESETS.get(preset_name)


def list_available_presets() -> List[str]:
    """
    Get list of available mixing presets

    Returns:
        List of preset names
    """
    return list(MIXING_PRESETS.keys())