"""
Mixing Presets Management System
Handles loading, validation, and management of audio mixing presets
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from models.audio_mixing_engine import MixingParams

logger = logging.getLogger(__name__)


@dataclass
class PresetMetadata:
    """Metadata for a mixing preset"""
    id: str
    name: str
    description: str
    category: str
    difficulty: str
    use_cases: List[str]


class MixingPresetManager:
    """
    Manager for audio mixing presets

    Handles loading from JSON configuration files, validation,
    and runtime management of mixing presets.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preset manager

        Args:
            config_path: Path to mixing presets JSON file
        """
        if config_path is None:
            # Default to config/mixing_presets.json relative to backend directory
            backend_dir = Path(__file__).parent.parent
            config_path = backend_dir / "config" / "mixing_presets.json"

        self.config_path = Path(config_path)
        self.presets: Dict[str, MixingParams] = {}
        self.metadata: Dict[str, PresetMetadata] = {}
        self.categories: Dict[str, Dict[str, str]] = {}
        self.difficulty_levels: Dict[str, Dict[str, str]] = {}

        self._load_presets()

    def _load_presets(self) -> None:
        """Load presets from JSON configuration file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Preset config file not found: {self.config_path}")
                self._load_default_presets()
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Load categories and difficulty levels
            self.categories = config_data.get('categories', {})
            self.difficulty_levels = config_data.get('difficulty_levels', {})

            # Load individual presets
            presets_data = config_data.get('presets', {})

            for preset_id, preset_data in presets_data.items():
                try:
                    # Extract metadata
                    metadata = PresetMetadata(
                        id=preset_id,
                        name=preset_data.get('name', preset_id),
                        description=preset_data.get('description', ''),
                        category=preset_data.get('category', 'general'),
                        difficulty=preset_data.get('difficulty', 'intermediate'),
                        use_cases=preset_data.get('use_cases', [])
                    )
                    self.metadata[preset_id] = metadata

                    # Convert parameters to MixingParams
                    params_data = preset_data.get('parameters', {})
                    mixing_params = self._dict_to_mixing_params(params_data)
                    self.presets[preset_id] = mixing_params

                    logger.debug(f"Loaded preset: {preset_id}")

                except Exception as e:
                    logger.error(f"Failed to load preset '{preset_id}': {e}")
                    continue

            logger.info(f"Loaded {len(self.presets)} mixing presets from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load presets from {self.config_path}: {e}")
            self._load_default_presets()

    def _dict_to_mixing_params(self, params_dict: Dict[str, Any]) -> MixingParams:
        """
        Convert dictionary parameters to MixingParams object

        Args:
            params_dict: Dictionary with parameter values

        Returns:
            MixingParams object
        """
        return MixingParams(
            noise_gate_enabled=params_dict.get('noise_gate_enabled', True),
            noise_gate_threshold_db=params_dict.get('noise_gate_threshold_db', -35.0),
            noise_gate_ratio=params_dict.get('noise_gate_ratio', 8.0),
            noise_gate_attack_ms=params_dict.get('noise_gate_attack_ms', 1.0),
            noise_gate_release_ms=params_dict.get('noise_gate_release_ms', 150.0),
            highpass_enabled=params_dict.get('highpass_enabled', True),
            highpass_frequency_hz=params_dict.get('highpass_frequency_hz', 80.0),
            compressor_enabled=params_dict.get('compressor_enabled', True),
            compressor_threshold_db=params_dict.get('compressor_threshold_db', -18.0),
            compressor_ratio=params_dict.get('compressor_ratio', 3.0),
            compressor_attack_ms=params_dict.get('compressor_attack_ms', 10.0),
            compressor_release_ms=params_dict.get('compressor_release_ms', 100.0),
            eq_enabled=params_dict.get('eq_enabled', True),
            eq_low_enabled=params_dict.get('eq_low_enabled', True),
            eq_low_frequency_hz=params_dict.get('eq_low_frequency_hz', 150.0),
            eq_low_gain_db=params_dict.get('eq_low_gain_db', 1.0),
            eq_low_q=params_dict.get('eq_low_q', 1.0),
            eq_low_mid_enabled=params_dict.get('eq_low_mid_enabled', True),
            eq_low_mid_frequency_hz=params_dict.get('eq_low_mid_frequency_hz', 400.0),
            eq_low_mid_gain_db=params_dict.get('eq_low_mid_gain_db', 1.5),
            eq_low_mid_q=params_dict.get('eq_low_mid_q', 1.2),
            eq_presence_frequency_hz=params_dict.get('eq_presence_frequency_hz', 2800.0),
            eq_presence_gain_db=params_dict.get('eq_presence_gain_db', 1.5),
            eq_presence_q=params_dict.get('eq_presence_q', 1.2),
            eq_high_enabled=params_dict.get('eq_high_enabled', True),
            eq_high_frequency_hz=params_dict.get('eq_high_frequency_hz', 8000.0),
            eq_high_gain_db=params_dict.get('eq_high_gain_db', 0.8),
            eq_high_q=params_dict.get('eq_high_q', 0.7),
            reverb_enabled=params_dict.get('reverb_enabled', True),
            reverb_type=params_dict.get('reverb_type', 'algorithm'),
            reverb_room_size=params_dict.get('reverb_room_size', 0.2),
            reverb_damping=params_dict.get('reverb_damping', 0.6),
            reverb_wet_level=params_dict.get('reverb_wet_level', 0.12),
            reverb_width=params_dict.get('reverb_width', 1.0)
        )

    def _load_default_presets(self) -> None:
        """Load hardcoded default presets as fallback"""
        logger.info("Loading default mixing presets")

        # Import default presets from mixing engine
        from models.audio_mixing_engine import MIXING_PRESETS

        for preset_id, mixing_params in MIXING_PRESETS.items():
            self.presets[preset_id] = mixing_params

            # Create basic metadata for default presets
            descriptions = {
                "vocal_clean": "Clean vocal preset - light processing for natural sound",
                "vocal_warm": "Warm vocal preset - fuller, more present sound",
                "podcast": "Podcast preset - maximum clarity and consistency",
                "vocal_performance": "Performance preset - polished, professional sound"
            }

            self.metadata[preset_id] = PresetMetadata(
                id=preset_id,
                name=preset_id.replace('_', ' ').title(),
                description=descriptions.get(preset_id, f"{preset_id} preset"),
                category="vocal" if "vocal" in preset_id else "speech",
                difficulty="intermediate",
                use_cases=[]
            )

        logger.info(f"Loaded {len(self.presets)} default presets")

    def get_preset(self, preset_id: str) -> Optional[MixingParams]:
        """
        Get mixing parameters for a specific preset

        Args:
            preset_id: ID of the preset to retrieve

        Returns:
            MixingParams object or None if not found
        """
        return self.presets.get(preset_id)

    def get_preset_metadata(self, preset_id: str) -> Optional[PresetMetadata]:
        """
        Get metadata for a specific preset

        Args:
            preset_id: ID of the preset

        Returns:
            PresetMetadata object or None if not found
        """
        return self.metadata.get(preset_id)

    def list_presets(self) -> List[str]:
        """
        Get list of available preset IDs

        Returns:
            List of preset IDs
        """
        return list(self.presets.keys())

    def list_presets_by_category(self, category: str) -> List[str]:
        """
        Get list of presets filtered by category

        Args:
            category: Category to filter by

        Returns:
            List of preset IDs in the specified category
        """
        return [
            preset_id for preset_id, metadata in self.metadata.items()
            if metadata.category == category
        ]

    def list_presets_by_difficulty(self, difficulty: str) -> List[str]:
        """
        Get list of presets filtered by difficulty level

        Args:
            difficulty: Difficulty level to filter by

        Returns:
            List of preset IDs with the specified difficulty
        """
        return [
            preset_id for preset_id, metadata in self.metadata.items()
            if metadata.difficulty == difficulty
        ]

    def search_presets(self, use_case: str) -> List[str]:
        """
        Search presets by use case

        Args:
            use_case: Use case to search for

        Returns:
            List of preset IDs matching the use case
        """
        matching_presets = []

        for preset_id, metadata in self.metadata.items():
            if use_case.lower() in [uc.lower() for uc in metadata.use_cases]:
                matching_presets.append(preset_id)
            elif use_case.lower() in metadata.name.lower():
                matching_presets.append(preset_id)
            elif use_case.lower() in metadata.description.lower():
                matching_presets.append(preset_id)

        return matching_presets

    def get_categories(self) -> Dict[str, Dict[str, str]]:
        """
        Get available categories

        Returns:
            Dictionary of categories with metadata
        """
        return self.categories

    def get_difficulty_levels(self) -> Dict[str, Dict[str, str]]:
        """
        Get available difficulty levels

        Returns:
            Dictionary of difficulty levels with metadata
        """
        return self.difficulty_levels

    def validate_preset(self, preset_id: str) -> bool:
        """
        Validate that a preset is properly configured

        Args:
            preset_id: ID of the preset to validate

        Returns:
            True if preset is valid, False otherwise
        """
        try:
            if preset_id not in self.presets:
                return False

            mixing_params = self.presets[preset_id]

            # Basic validation - check that all required parameters are within reasonable ranges
            if not (0.0 <= mixing_params.noise_gate_ratio <= 20.0):
                return False
            if not (-60.0 <= mixing_params.noise_gate_threshold_db <= 0.0):
                return False
            if not (20.0 <= mixing_params.highpass_frequency_hz <= 500.0):
                return False
            if not (-40.0 <= mixing_params.compressor_threshold_db <= 0.0):
                return False
            if not (1.0 <= mixing_params.compressor_ratio <= 10.0):
                return False
            if not (-6.0 <= mixing_params.eq_presence_gain_db <= 6.0):
                return False
            if not (1000.0 <= mixing_params.eq_presence_frequency_hz <= 8000.0):
                return False
            if not (0.0 <= mixing_params.reverb_room_size <= 1.0):
                return False
            if not (0.0 <= mixing_params.reverb_wet_level <= 0.5):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating preset '{preset_id}': {e}")
            return False

    def add_custom_preset(
        self,
        preset_id: str,
        mixing_params: MixingParams,
        metadata: PresetMetadata
    ) -> bool:
        """
        Add a custom preset at runtime

        Args:
            preset_id: ID for the new preset
            mixing_params: Mixing parameters
            metadata: Preset metadata

        Returns:
            True if preset was added successfully, False otherwise
        """
        try:
            if preset_id in self.presets:
                logger.warning(f"Preset '{preset_id}' already exists, overwriting")

            self.presets[preset_id] = mixing_params
            self.metadata[preset_id] = metadata

            logger.info(f"Added custom preset: {preset_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add custom preset '{preset_id}': {e}")
            return False

    def get_preset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about all presets

        Returns:
            Dictionary with preset statistics and information
        """
        category_counts = {}
        difficulty_counts = {}

        for metadata in self.metadata.values():
            category_counts[metadata.category] = category_counts.get(metadata.category, 0) + 1
            difficulty_counts[metadata.difficulty] = difficulty_counts.get(metadata.difficulty, 0) + 1

        return {
            "total_presets": len(self.presets),
            "categories": category_counts,
            "difficulty_levels": difficulty_counts,
            "config_path": str(self.config_path),
            "config_exists": self.config_path.exists()
        }


# Global preset manager instance
_preset_manager = None


def get_preset_manager() -> MixingPresetManager:
    """
    Get the global preset manager instance

    Returns:
        MixingPresetManager instance
    """
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = MixingPresetManager()
    return _preset_manager


# Convenience functions for backward compatibility
def get_mixing_preset(preset_name: str) -> Optional[MixingParams]:
    """Get a mixing preset by name"""
    return get_preset_manager().get_preset(preset_name)


def list_available_presets() -> List[str]:
    """Get list of available preset names"""
    return get_preset_manager().list_presets()


def get_preset_categories() -> Dict[str, Dict[str, str]]:
    """Get available preset categories"""
    return get_preset_manager().get_categories()


def search_presets_by_use_case(use_case: str) -> List[str]:
    """Search presets by use case"""
    return get_preset_manager().search_presets(use_case)