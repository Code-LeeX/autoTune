"""
Configuration settings for the AI Pitch Correction Backend
"""
import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Application Settings
    app_name: str = "AI Pitch Correction API"
    version: str = "0.1.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # File Upload Settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".wav", ".mp3", ".flac", ".m4a"]
    upload_dir: str = "uploads"
    temp_dir: str = "temp_audio"
    output_dir: str = "processed_audio"

    # AI Model Settings
    crepe_model: str = "full"  # tiny, small, medium, large, full
    crepe_step_size: int = 10  # milliseconds
    crepe_confidence_threshold: float = 0.85

    # DDSP Settings
    ddsp_model_path: Optional[str] = None
    use_ddsp: bool = True
    fallback_to_world: bool = True

    # Audio Processing Settings
    sample_rate: int = 44100
    hop_length: int = 512
    frame_length: int = 2048

    # Pitch Correction Settings
    default_key: str = "C"
    default_scale: str = "major"
    vibrato_preservation: bool = True
    smoothing_factor: float = 0.5
    correction_strength: float = 0.8

    # Database (if needed later)
    database_url: Optional[str] = None

    # Security
    secret_key: str = "your-secret-key-change-in-production"

    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()