"""
FastAPI routes for audio processing and pitch correction
"""
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import json
import traceback

import numpy as np
import librosa
import soundfile as sf
from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from models.pitch_detector import CreepePitchDetector
from models.audio_synthesizer import PitchCorrectionEngine
from audio_processing.pitch_quantizer import MusicalScale, VibratoAnalyzer, PitchQuantizer
from config.settings import Settings
import logging

logger = logging.getLogger(__name__)

# Initialize components
settings = Settings()
pitch_detector = CreepePitchDetector(model_capacity='small', step_size=10)
correction_engine = PitchCorrectionEngine(sample_rate=44100, frame_period=5.0)

# Global storage for processing sessions
processing_sessions: Dict[str, Dict[str, Any]] = {}

router = APIRouter(prefix="/api/audio", tags=["audio"])


class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis"""
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    analyze_vibrato: bool = Field(default=True)


class PitchCorrectionRequest(BaseModel):
    """Request model for pitch correction"""
    session_id: str
    key: str = Field(default="C", description="Musical key")
    scale_type: str = Field(default="major", description="Scale type")
    correction_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    preserve_vibrato: bool = Field(default=True)
    preserve_formants: bool = Field(default=True)
    smoothing_factor: float = Field(default=0.1, ge=0.0, le=1.0)


class ProcessingStatus(BaseModel):
    """Processing status response"""
    session_id: str
    status: str  # 'processing', 'completed', 'error'
    progress: float = Field(ge=0.0, le=1.0)
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class ConnectionManager:
    """WebSocket connection manager for real-time updates"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_update(self, session_id: str, status: ProcessingStatus):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(status.json())
            except Exception as e:
                logger.error(f"Failed to send WebSocket update: {e}")
                self.disconnect(session_id)


manager = ConnectionManager()


@router.post("/upload", response_model=Dict[str, Any])
async def upload_audio(
    file: UploadFile = File(...),
    analysis_request: str = Form(...)
):
    """
    Upload audio file and perform initial analysis

    Args:
        file: Audio file (WAV, MP3, FLAC, etc.)
        analysis_request: JSON string with analysis parameters

    Returns:
        Session ID and initial analysis results
    """
    try:
        # Parse analysis parameters
        params = AudioAnalysisRequest.model_validate_json(analysis_request)

        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Create temporary directory for this session
        temp_dir = Path(tempfile.gettempdir()) / "melodyne_sessions" / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        original_file = temp_dir / f"original{file_ext}"
        with open(original_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load and validate audio
        try:
            audio_data, sample_rate = librosa.load(original_file, sr=None, mono=True)
            duration = len(audio_data) / sample_rate

            # Check duration limits
            if duration > 300:  # 5 minutes max
                raise HTTPException(status_code=400, detail="Audio too long (max 5 minutes)")
            if duration < 0.1:  # 100ms min
                raise HTTPException(status_code=400, detail="Audio too short (min 100ms)")

        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

        # Store session data
        processing_sessions[session_id] = {
            'original_file': str(original_file),
            'temp_dir': str(temp_dir),
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'duration': duration,
            'status': 'uploaded',
            'analysis_params': params.model_dump()
        }

        # Basic audio info
        audio_info = {
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': 1,
            'samples': len(audio_data),
            'bit_depth': 32,  # We use float32
            'file_size': len(content)
        }

        logger.info(f"Audio uploaded successfully: {session_id}, duration: {duration:.2f}s")

        return {
            'session_id': session_id,
            'status': 'uploaded',
            'audio_info': audio_info,
            'message': 'Audio uploaded successfully'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/analyze/{session_id}", response_model=ProcessingStatus)
async def analyze_pitch(session_id: str):
    """
    Perform pitch analysis on uploaded audio

    Args:
        session_id: Session ID from upload

    Returns:
        Analysis results with pitch data
    """
    try:
        if session_id not in processing_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = processing_sessions[session_id]
        if session['status'] != 'uploaded':
            raise HTTPException(status_code=400, detail="Invalid session status")

        # Update status
        session['status'] = 'analyzing'
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='analyzing',
            progress=0.0,
            message="Starting pitch analysis..."
        ))

        # Get audio data
        audio_data = session['audio_data']
        sample_rate = session['sample_rate']
        params = session['analysis_params']

        # Perform pitch detection
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='analyzing',
            progress=0.3,
            message="Extracting pitch with CREPE..."
        ))

        if params['analyze_vibrato']:
            pitch_data = pitch_detector.extract_pitch_with_vibrato_analysis(
                audio_data,
                sample_rate
            )
            # Apply confidence threshold manually for vibrato analysis
            confidence_mask = pitch_data['confidence'] >= params['confidence_threshold']
            pitch_data['voiced_mask'] = confidence_mask
            pitch_data['frequency'][~confidence_mask] = 0.0
        else:
            pitch_data = pitch_detector.extract_pitch(
                audio_data,
                sample_rate,
                confidence_threshold=params['confidence_threshold']
            )

        # Calculate statistics
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='analyzing',
            progress=0.8,
            message="Computing analysis statistics..."
        ))

        voiced_mask = pitch_data['voiced_mask']
        voiced_f0 = pitch_data['frequency'][voiced_mask]

        stats = {
            'total_frames': len(pitch_data['time']),
            'voiced_frames': int(np.sum(voiced_mask)),
            'voicing_ratio': float(np.sum(voiced_mask) / len(voiced_mask)),
            'avg_confidence': float(np.mean(pitch_data['confidence'])),
            'pitch_range': {
                'min': float(np.min(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
                'max': float(np.max(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
                'mean': float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0,
                'std': float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            }
        }

        # Add vibrato statistics if available
        if 'vibrato_presence' in pitch_data:
            vibrato_frames = np.sum(pitch_data['vibrato_presence'])
            stats['vibrato'] = {
                'frames_with_vibrato': int(vibrato_frames),
                'vibrato_ratio': float(vibrato_frames / len(pitch_data['time'])),
                'avg_vibrato_rate': float(np.mean(pitch_data['vibrato_rate'][pitch_data['vibrato_presence']])) if vibrato_frames > 0 else 0.0,
                'avg_vibrato_extent': float(np.mean(pitch_data['vibrato_extent'][pitch_data['vibrato_presence']])) if vibrato_frames > 0 else 0.0
            }

        # Store analysis results
        session['pitch_data'] = pitch_data
        session['analysis_stats'] = stats
        session['status'] = 'analyzed'

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'time': pitch_data['time'].tolist(),
            'frequency': pitch_data['frequency'].tolist(),
            'confidence': pitch_data['confidence'].tolist(),
            'voiced_mask': pitch_data['voiced_mask'].tolist(),
            'stats': stats
        }

        if 'vibrato_presence' in pitch_data:
            serializable_data.update({
                'vibrato_presence': pitch_data['vibrato_presence'].tolist(),
                'vibrato_rate': pitch_data['vibrato_rate'].tolist(),
                'vibrato_extent': pitch_data['vibrato_extent'].tolist()
            })

        logger.info(f"Pitch analysis completed: {session_id}, voiced ratio: {stats['voicing_ratio']:.2%}")

        # Send completion notification via WebSocket
        completion_status = ProcessingStatus(
            session_id=session_id,
            status='analyzed',
            progress=1.0,
            message="Pitch analysis completed successfully",
            data=serializable_data
        )

        await manager.send_update(session_id, completion_status)

        return completion_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pitch analysis failed for {session_id}: {e}")
        traceback.print_exc()
        if session_id in processing_sessions:
            processing_sessions[session_id]['status'] = 'error'
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/correct/{session_id}", response_model=ProcessingStatus)
async def correct_pitch(session_id: str, request: PitchCorrectionRequest):
    """
    Apply pitch correction to analyzed audio

    Args:
        session_id: Session ID
        request: Pitch correction parameters

    Returns:
        Correction results
    """
    try:
        if session_id not in processing_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = processing_sessions[session_id]
        if session['status'] != 'analyzed':
            raise HTTPException(status_code=400, detail="Audio must be analyzed first")

        # Update status
        session['status'] = 'correcting'
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='correcting',
            progress=0.0,
            message="Initializing pitch correction..."
        ))

        # Get data
        audio_data = session['audio_data']
        sample_rate = session['sample_rate']
        pitch_data = session['pitch_data']
        temp_dir = Path(session['temp_dir'])

        # Initialize musical scale and quantizer
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='correcting',
            progress=0.1,
            message=f"Setting up {request.key} {request.scale_type} scale..."
        ))

        scale = MusicalScale(
            key=request.key,
            scale_type=request.scale_type,
            reference_freq=440.0
        )

        vibrato_analyzer = VibratoAnalyzer()
        quantizer = PitchQuantizer(scale, vibrato_analyzer)

        # Quantize pitch trajectory
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='correcting',
            progress=0.3,
            message="Quantizing pitch to musical scale..."
        ))

        quantization_result = quantizer.quantize_pitch_trajectory(
            time=pitch_data['time'],
            frequency=pitch_data['frequency'],
            confidence=pitch_data['confidence'],
            correction_strength=request.correction_strength,
            preserve_vibrato=request.preserve_vibrato,
            smoothing_factor=request.smoothing_factor
        )

        target_f0 = quantization_result['quantized_frequency']

        # Apply pitch correction using audio synthesizer
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='correcting',
            progress=0.6,
            message="Synthesizing corrected audio..."
        ))

        correction_result = correction_engine.correct_pitch(
            audio=audio_data,
            target_f0=target_f0,
            correction_strength=request.correction_strength,
            preserve_formants=request.preserve_formants,
            smoothing_factor=request.smoothing_factor
        )

        corrected_audio = correction_result['corrected_audio']

        # Save corrected audio
        await manager.send_update(session_id, ProcessingStatus(
            session_id=session_id,
            status='correcting',
            progress=0.9,
            message="Saving corrected audio..."
        ))

        corrected_file = temp_dir / "corrected.wav"
        sf.write(corrected_file, corrected_audio, sample_rate)

        # Calculate correction statistics
        original_f0 = pitch_data['frequency']
        voiced_mask = original_f0 > 0

        correction_stats = {
            'frames_corrected': int(np.sum(quantization_result['correction_applied'])),
            'frames_with_vibrato_preserved': int(np.sum(quantization_result['vibrato_preserved'])),
            'correction_ratio': float(np.sum(quantization_result['correction_applied']) / len(original_f0))
        }

        if np.sum(voiced_mask) > 0:
            original_deviation = np.abs(quantization_result['pitch_deviation'][voiced_mask])
            avg_original_deviation = float(np.mean(original_deviation))

            corrected_deviation = []
            for i in range(len(target_f0)):
                if target_f0[i] > 0:
                    dev = scale.get_pitch_deviation(target_f0[i])
                    corrected_deviation.append(abs(dev))

            if len(corrected_deviation) > 0:
                avg_corrected_deviation = float(np.mean(corrected_deviation))
                improvement = avg_original_deviation - avg_corrected_deviation

                correction_stats.update({
                    'pitch_accuracy_improvement_cents': improvement,
                    'avg_original_deviation_cents': avg_original_deviation,
                    'avg_corrected_deviation_cents': avg_corrected_deviation
                })

        # Store results
        session['corrected_file'] = str(corrected_file)
        session['correction_params'] = request.model_dump()
        session['correction_stats'] = correction_stats
        session['quantization_result'] = quantization_result
        session['status'] = 'corrected'

        # Prepare serializable response data
        response_data = {
            'correction_stats': correction_stats,
            'quantized_frequency': target_f0.tolist(),
            'correction_applied': quantization_result['correction_applied'].tolist(),
            'vibrato_preserved': quantization_result['vibrato_preserved'].tolist(),
            'pitch_deviation': quantization_result['pitch_deviation'].tolist()
        }

        logger.info(f"Pitch correction completed: {session_id}, "
                   f"improvement: {correction_stats.get('pitch_accuracy_improvement_cents', 0):.1f} cents")

        # Send completion notification via WebSocket
        completion_status = ProcessingStatus(
            session_id=session_id,
            status='corrected',
            progress=1.0,
            message="Pitch correction completed successfully",
            data=response_data
        )

        await manager.send_update(session_id, completion_status)

        return completion_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pitch correction failed for {session_id}: {e}")
        traceback.print_exc()
        if session_id in processing_sessions:
            processing_sessions[session_id]['status'] = 'error'
        raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")


@router.get("/download/{session_id}/{file_type}")
async def download_audio(session_id: str, file_type: str, format: str = "wav"):
    """
    Download processed audio files in various formats

    Args:
        session_id: Session ID
        file_type: 'original' or 'corrected'
        format: Output format ('wav', 'mp3', 'flac', 'm4a', 'ogg')

    Returns:
        Audio file download in requested format
    """
    try:
        # Validate format
        supported_formats = ['wav', 'mp3', 'flac', 'm4a', 'ogg']
        if format not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported formats: {', '.join(supported_formats)}"
            )

        if session_id not in processing_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = processing_sessions[session_id]

        # Get source file
        if file_type == 'original':
            source_path = session['original_file']
            base_filename = f"original_{session_id}"
        elif file_type == 'corrected':
            if 'corrected_file' not in session:
                raise HTTPException(status_code=400, detail="No corrected audio available")
            source_path = session['corrected_file']
            base_filename = f"corrected_{session_id}"
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        if not os.path.exists(source_path):
            raise HTTPException(status_code=404, detail="File not found")

        # If requested format is WAV and source is already WAV, return directly
        if format == 'wav' and source_path.lower().endswith('.wav'):
            return FileResponse(
                source_path,
                media_type="audio/wav",
                filename=f"{base_filename}.wav"
            )

        # Convert to requested format
        converted_path = await convert_audio_format(source_path, format, base_filename)

        # Get appropriate MIME type
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4',
            'ogg': 'audio/ogg'
        }

        return FileResponse(
            converted_path,
            media_type=mime_types[format],
            filename=f"{base_filename}.{format}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


async def convert_audio_format(source_path: str, target_format: str, base_filename: str) -> str:
    """
    Convert audio file to target format using soundfile and subprocess

    Args:
        source_path: Path to source audio file
        target_format: Target format ('mp3', 'flac', 'm4a', 'ogg')
        base_filename: Base filename for output

    Returns:
        Path to converted file
    """
    try:
        # Create output directory for converted files
        output_dir = os.path.join(settings.temp_dir, "converted")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{base_filename}.{target_format}")

        # Load audio with librosa
        audio_data, sample_rate = librosa.load(source_path, sr=None, mono=False)

        # Ensure audio is in proper shape (channels, samples)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        elif audio_data.ndim == 2 and audio_data.shape[0] > audio_data.shape[1]:
            audio_data = audio_data.T

        if target_format == 'wav':
            # Direct save with soundfile
            sf.write(output_path, audio_data.T, sample_rate, format='WAV')

        elif target_format == 'flac':
            # Save as FLAC
            sf.write(output_path, audio_data.T, sample_rate, format='FLAC')

        else:
            # For MP3, M4A, OGG - use ffmpeg via subprocess
            import subprocess

            # First save as temporary WAV
            temp_wav = os.path.join(output_dir, f"temp_{base_filename}.wav")
            sf.write(temp_wav, audio_data.T, sample_rate, format='WAV')

            # Convert with ffmpeg
            if target_format == 'mp3':
                cmd = [
                    'ffmpeg', '-i', temp_wav, '-codec:a', 'libmp3lame',
                    '-b:a', '320k', '-y', output_path
                ]
            elif target_format == 'm4a':
                cmd = [
                    'ffmpeg', '-i', temp_wav, '-codec:a', 'aac',
                    '-b:a', '256k', '-y', output_path
                ]
            elif target_format == 'ogg':
                cmd = [
                    'ffmpeg', '-i', temp_wav, '-codec:a', 'libvorbis',
                    '-b:a', '320k', '-y', output_path
                ]

            # Execute conversion
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                logger.error(f"FFmpeg conversion failed: {error_msg}")
                raise Exception(f"Audio conversion failed: {error_msg}")

            # Clean up temporary WAV file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        return output_path

    except Exception as e:
        logger.error(f"Audio format conversion failed: {e}")
        raise Exception(f"Format conversion failed: {str(e)}")


@router.get("/formats")
async def get_supported_formats():
    """
    Get list of supported audio export formats

    Returns:
        List of supported formats with metadata
    """
    return {
        "formats": [
            {
                "id": "wav",
                "name": "WAV",
                "description": "Uncompressed audio, best quality",
                "extension": ".wav",
                "mime_type": "audio/wav",
                "is_lossless": True
            },
            {
                "id": "flac",
                "name": "FLAC",
                "description": "Lossless compression, smaller than WAV",
                "extension": ".flac",
                "mime_type": "audio/flac",
                "is_lossless": True
            },
            {
                "id": "mp3",
                "name": "MP3",
                "description": "Compressed audio, widely compatible",
                "extension": ".mp3",
                "mime_type": "audio/mpeg",
                "is_lossless": False
            },
            {
                "id": "m4a",
                "name": "M4A (AAC)",
                "description": "High-quality compressed audio",
                "extension": ".m4a",
                "mime_type": "audio/mp4",
                "is_lossless": False
            },
            {
                "id": "ogg",
                "name": "OGG Vorbis",
                "description": "Open-source compressed audio",
                "extension": ".ogg",
                "mime_type": "audio/ogg",
                "is_lossless": False
            }
        ]
    }


@router.get("/status/{session_id}", response_model=ProcessingStatus)
async def get_status(session_id: str):
    """Get processing status for a session"""
    if session_id not in processing_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = processing_sessions[session_id]

    return ProcessingStatus(
        session_id=session_id,
        status=session['status'],
        progress=1.0 if session['status'] in ['analyzed', 'corrected'] else 0.0,
        message=f"Session status: {session['status']}"
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up session data and temporary files"""
    try:
        if session_id not in processing_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = processing_sessions[session_id]

        # Clean up temporary files
        temp_dir = Path(session['temp_dir'])
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)

        # Remove from sessions
        del processing_sessions[session_id]

        # Disconnect WebSocket
        manager.disconnect(session_id)

        return {"message": f"Session {session_id} cleaned up successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(session_id, websocket)
    try:
        while True:
            # Keep connection alive and handle any client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        manager.disconnect(session_id)


@router.get("/scales")
async def get_available_scales():
    """Get list of available musical scales"""
    return {
        'keys': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        'scale_types': list(MusicalScale.SCALES.keys())
    }


@router.get("/sessions")
async def list_active_sessions():
    """List all active processing sessions (for debugging)"""
    return {
        'active_sessions': len(processing_sessions),
        'sessions': {
            sid: {
                'status': session['status'],
                'duration': session.get('duration', 0),
                'sample_rate': session.get('sample_rate', 0)
            }
            for sid, session in processing_sessions.items()
        }
    }