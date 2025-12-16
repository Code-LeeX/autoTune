#!/usr/bin/env python3
"""
Test script for the FastAPI audio processing endpoints
"""
import requests
import json
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
import time

API_BASE = "http://localhost:8000/api/audio"

def generate_test_audio(filename: str, duration: float = 2.0, sample_rate: int = 44100):
    """Generate a test audio file with known pitch characteristics"""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Create a melody that's slightly out of tune
    base_freq = 440.0  # A4
    melody = [1.0, 1.125, 1.25, 1.375, 1.5]  # Major scale ratios
    note_duration = duration / len(melody)

    signal = np.zeros_like(t)

    for i, freq_ratio in enumerate(melody):
        start_time = i * note_duration
        end_time = (i + 1) * note_duration

        mask = (t >= start_time) & (t < end_time)
        note_t = t[mask]

        # Base frequency with small pitch error
        freq = base_freq * freq_ratio
        pitch_error = freq * 0.05 * np.sin(2 * np.pi * 2.0 * note_t)  # ±5% error

        # Add some vibrato
        vibrato = freq * 0.02 * np.sin(2 * np.pi * 6.0 * note_t)  # 2% vibrato at 6Hz

        final_freq = freq + pitch_error + vibrato

        # Generate harmonic signal
        note_signal = np.sin(2 * np.pi * final_freq * note_t)
        note_signal += 0.3 * np.sin(2 * np.pi * 2 * final_freq * note_t)  # 2nd harmonic

        signal[mask] = note_signal

    # Add envelope and noise
    envelope = np.exp(-t * 0.3)  # Decay envelope
    signal *= envelope
    signal += 0.01 * np.random.randn(len(signal))  # Small amount of noise

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    sf.write(filename, signal.astype(np.float32), sample_rate)
    print(f"Generated test audio: {filename} ({duration:.1f}s, {sample_rate}Hz)")
    return filename

def test_audio_upload(filename: str):
    """Test audio upload endpoint"""
    print("\n=== Testing Audio Upload ===")

    analysis_params = {
        "confidence_threshold": 0.8,
        "analyze_vibrato": True
    }

    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'audio/wav')}
        data = {'analysis_request': json.dumps(analysis_params)}

        response = requests.post(f"{API_BASE}/upload", files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Upload successful: session_id = {result['session_id']}")
        print(f"  Audio info: {result['audio_info']['duration']:.2f}s, {result['audio_info']['sample_rate']}Hz")
        return result['session_id']
    else:
        print(f"✗ Upload failed: {response.status_code} - {response.text}")
        return None

def test_pitch_analysis(session_id: str):
    """Test pitch analysis endpoint"""
    print("\n=== Testing Pitch Analysis ===")

    response = requests.post(f"{API_BASE}/analyze/{session_id}")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Analysis successful")
        print(f"  Status: {result['status']}")
        print(f"  Progress: {result['progress']}")

        if 'data' in result:
            data = result['data']
            stats = data['stats']
            print(f"  Voiced frames: {stats['voiced_frames']}/{stats['total_frames']} ({stats['voicing_ratio']:.1%})")
            print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
            print(f"  Pitch range: {stats['pitch_range']['min']:.1f} - {stats['pitch_range']['max']:.1f} Hz")

            if 'vibrato' in stats:
                vibrato = stats['vibrato']
                print(f"  Vibrato detected: {vibrato['frames_with_vibrato']} frames ({vibrato['vibrato_ratio']:.1%})")

        return True
    else:
        print(f"✗ Analysis failed: {response.status_code} - {response.text}")
        return False

def test_pitch_correction(session_id: str):
    """Test pitch correction endpoint"""
    print("\n=== Testing Pitch Correction ===")

    correction_params = {
        "session_id": session_id,
        "key": "A",
        "scale_type": "major",
        "correction_strength": 0.8,
        "preserve_vibrato": True,
        "preserve_formants": True,
        "smoothing_factor": 0.1
    }

    response = requests.post(f"{API_BASE}/correct/{session_id}", json=correction_params)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Correction successful")
        print(f"  Status: {result['status']}")
        print(f"  Progress: {result['progress']}")

        if 'data' in result:
            stats = result['data']['correction_stats']
            print(f"  Frames corrected: {stats['frames_corrected']}")
            print(f"  Correction ratio: {stats['correction_ratio']:.1%}")
            print(f"  Vibrato preserved: {stats['frames_with_vibrato_preserved']} frames")

            if 'pitch_accuracy_improvement_cents' in stats:
                improvement = stats['pitch_accuracy_improvement_cents']
                print(f"  Pitch improvement: {improvement:.1f} cents")

        return True
    else:
        print(f"✗ Correction failed: {response.status_code} - {response.text}")
        return False

def test_download_audio(session_id: str, file_type: str = 'corrected'):
    """Test audio download endpoint"""
    print(f"\n=== Testing Audio Download ({file_type}) ===")

    response = requests.get(f"{API_BASE}/download/{session_id}/{file_type}")

    if response.status_code == 200:
        output_file = f"test_{file_type}_{session_id[:8]}.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)

        print(f"✓ Download successful: {output_file} ({len(response.content)} bytes)")
        return output_file
    else:
        print(f"✗ Download failed: {response.status_code} - {response.text}")
        return None

def test_session_cleanup(session_id: str):
    """Test session cleanup endpoint"""
    print("\n=== Testing Session Cleanup ===")

    response = requests.delete(f"{API_BASE}/session/{session_id}")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Cleanup successful: {result['message']}")
        return True
    else:
        print(f"✗ Cleanup failed: {response.status_code} - {response.text}")
        return False

def main():
    """Run complete API test suite"""
    print("FastAPI Audio Processing Test Suite")
    print("=" * 50)

    try:
        # Generate test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_audio_file = tmp.name

        generate_test_audio(test_audio_file, duration=1.5, sample_rate=22050)

        # Test the complete pipeline
        session_id = test_audio_upload(test_audio_file)
        if not session_id:
            return False

        # Wait a moment for processing
        time.sleep(0.5)

        # Test analysis
        if not test_pitch_analysis(session_id):
            return False

        # Test correction
        if not test_pitch_correction(session_id):
            return False

        # Test downloads
        corrected_file = test_download_audio(session_id, 'corrected')
        original_file = test_download_audio(session_id, 'original')

        # Test cleanup
        test_session_cleanup(session_id)

        print("\n" + "=" * 50)
        print("✓ All API tests passed successfully!")

        # Clean up test files
        for file in [test_audio_file, corrected_file, original_file]:
            if file and os.path.exists(file):
                os.unlink(file)

        return True

    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)