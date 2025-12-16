#!/usr/bin/env python3
"""
Integration test to demonstrate frontend-backend connectivity
"""
import os
import sys
import requests
import json
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
import time

# Add backend path
sys.path.insert(0, '/Users/xiang.li/Melodyne/backend')

def generate_test_audio():
    """Generate a simple test audio file"""
    duration = 1.5
    sample_rate = 22050
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Create a simple melody
    freqs = [440, 494, 523, 587]  # A4, B4, C5, D5
    note_duration = duration / len(freqs)
    signal = np.zeros_like(t)

    for i, freq in enumerate(freqs):
        start_idx = int(i * note_duration * sample_rate)
        end_idx = int((i + 1) * note_duration * sample_rate)
        note_t = t[start_idx:end_idx]
        signal[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * freq * note_t)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, signal.astype(np.float32), sample_rate)
    return temp_file.name

def test_backend_health():
    """Test backend health endpoint"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend healthy: {data['service']} v{data['version']}")
            return True
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def test_scales_endpoint():
    """Test scales endpoint"""
    print("\n🎵 Testing scales endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/audio/scales", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Available keys: {len(data['keys'])} keys")
            print(f"✅ Available scales: {data['scale_types']}")
            return True
        else:
            print(f"❌ Scales endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Scales endpoint error: {e}")
        return False

def test_upload_workflow():
    """Test complete upload and analysis workflow"""
    print("\n📤 Testing upload and analysis workflow...")

    # Generate test audio
    print("  📝 Generating test audio...")
    audio_file = generate_test_audio()

    try:
        # Step 1: Upload file
        print("  📤 Uploading file...")
        analysis_params = {
            "confidence_threshold": 0.8,
            "analyze_vibrato": True
        }

        with open(audio_file, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            data = {'analysis_request': json.dumps(analysis_params)}
            response = requests.post("http://localhost:8000/api/audio/upload", files=files, data=data, timeout=10)

        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False

        upload_result = response.json()
        session_id = upload_result['session_id']
        audio_info = upload_result['audio_info']

        print(f"✅ Upload successful: {session_id}")
        print(f"   📊 Duration: {audio_info['duration']:.1f}s")
        print(f"   📊 Sample Rate: {audio_info['sample_rate']}Hz")
        print(f"   📊 File Size: {audio_info['file_size']} bytes")

        # Step 2: Start analysis
        print("  🔍 Starting pitch analysis...")
        response = requests.post(f"http://localhost:8000/api/audio/analyze/{session_id}", timeout=30)

        if response.status_code != 200:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
            return False

        analysis_result = response.json()

        if analysis_result['status'] == 'analyzed':
            stats = analysis_result['data']['stats']
            print(f"✅ Analysis complete:")
            print(f"   🎤 Voiced frames: {stats['voiced_frames']}/{stats['total_frames']} ({stats['voicing_ratio']*100:.1f}%)")
            print(f"   📊 Avg confidence: {stats['avg_confidence']:.3f}")
            print(f"   🎵 Pitch range: {stats['pitch_range']['min']:.1f}-{stats['pitch_range']['max']:.1f}Hz")

            if 'vibrato' in stats:
                vibrato = stats['vibrato']
                print(f"   🌊 Vibrato: {vibrato['frames_with_vibrato']} frames ({vibrato['vibrato_ratio']*100:.1f}%)")

        # Step 3: Test correction
        print("  🔧 Testing pitch correction...")
        correction_params = {
            "session_id": session_id,
            "key": "A",
            "scale_type": "major",
            "correction_strength": 0.8,
            "preserve_vibrato": True,
            "preserve_formants": True,
            "smoothing_factor": 0.1
        }

        response = requests.post(f"http://localhost:8000/api/audio/correct/{session_id}",
                               json=correction_params, timeout=30)

        if response.status_code != 200:
            print(f"❌ Correction failed: {response.status_code} - {response.text}")
            return False

        correction_result = response.json()

        if correction_result['status'] == 'corrected':
            correction_stats = correction_result['data']['correction_stats']
            print(f"✅ Correction complete:")
            print(f"   🔧 Frames corrected: {correction_stats['frames_corrected']}")
            print(f"   📊 Correction ratio: {correction_stats['correction_ratio']*100:.1f}%")

            if 'pitch_accuracy_improvement_cents' in correction_stats:
                improvement = correction_stats['pitch_accuracy_improvement_cents']
                print(f"   🎯 Pitch improvement: {improvement:.1f} cents")

        # Step 4: Test download
        print("  📥 Testing audio download...")
        response = requests.get(f"http://localhost:8000/api/audio/download/{session_id}/corrected", timeout=10)

        if response.status_code == 200:
            corrected_size = len(response.content)
            print(f"✅ Download successful: {corrected_size} bytes")
        else:
            print(f"❌ Download failed: {response.status_code}")
            return False

        # Step 5: Cleanup
        print("  🧹 Cleaning up session...")
        response = requests.delete(f"http://localhost:8000/api/audio/session/{session_id}", timeout=5)

        if response.status_code == 200:
            print(f"✅ Session cleanup successful")
        else:
            print(f"⚠️  Cleanup warning: {response.status_code}")

        return True

    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return False

    finally:
        # Clean up test file
        if os.path.exists(audio_file):
            os.unlink(audio_file)

def main():
    """Run integration tests"""
    print("🚀 Frontend-Backend Integration Test")
    print("=" * 50)

    tests = [
        ("Backend Health Check", test_backend_health),
        ("Scales Endpoint", test_scales_endpoint),
        ("Complete Upload Workflow", test_upload_workflow)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All integration tests PASSED!")
        print("🔗 Frontend-backend integration is working correctly!")
        return True
    else:
        print("⚠️  Some tests failed - check backend status")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)