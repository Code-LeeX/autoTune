#!/usr/bin/env python3
"""
Test script for CREPE pitch detection
"""
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.pitch_detector import CreepePitchDetector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def generate_test_signal(duration: float = 3.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a test audio signal with known pitch characteristics

    Args:
        duration: Signal duration in seconds
        sample_rate: Audio sample rate

    Returns:
        Generated audio signal
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Create a signal with varying pitch
    # Start at A4 (440Hz), go to C5 (523Hz), back to A4
    f0_base = 440.0

    # Pitch modulation (melody)
    pitch_curve = f0_base * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))  # Slow melody

    # Add vibrato (5Hz rate, 2% depth)
    vibrato = 0.02 * np.sin(2 * np.pi * 5.0 * t)
    pitch_with_vibrato = pitch_curve * (1 + vibrato)

    # Generate the audio signal
    signal = np.sin(2 * np.pi * pitch_with_vibrato * t)

    # Add some harmonics for realism
    signal += 0.3 * np.sin(2 * np.pi * 2 * pitch_with_vibrato * t)  # 2nd harmonic
    signal += 0.2 * np.sin(2 * np.pi * 3 * pitch_with_vibrato * t)  # 3rd harmonic

    # Add envelope (attack/decay)
    envelope = np.exp(-t * 0.5)  # Exponential decay
    signal *= envelope

    # Add a bit of noise
    noise = 0.01 * np.random.randn(len(signal))
    signal += noise

    return signal.astype(np.float32)


def test_crepe_basic():
    """Test basic CREPE functionality"""
    print("=== Testing Basic CREPE Functionality ===")

    # Generate test signal
    sample_rate = 44100
    audio = generate_test_signal(duration=2.0, sample_rate=sample_rate)

    print(f"Generated test signal: {len(audio)} samples at {sample_rate}Hz")

    # Initialize CREPE detector
    detector = CreepePitchDetector(model_capacity='small', step_size=10)

    # Extract pitch
    pitch_data = detector.extract_pitch(audio, sample_rate)

    print(f"Extracted pitch data:")
    print(f"  Time frames: {len(pitch_data['time'])}")
    print(f"  Frequency range: {np.min(pitch_data['frequency'][pitch_data['frequency'] > 0]):.1f} - "
          f"{np.max(pitch_data['frequency']):.1f} Hz")
    print(f"  Average confidence: {np.mean(pitch_data['confidence']):.3f}")
    print(f"  Voiced frames: {np.sum(pitch_data['voiced_mask'])}/{len(pitch_data['voiced_mask'])} "
          f"({np.sum(pitch_data['voiced_mask'])/len(pitch_data['voiced_mask'])*100:.1f}%)")

    return pitch_data


def test_vibrato_analysis():
    """Test vibrato analysis functionality"""
    print("\n=== Testing Vibrato Analysis ===")

    # Generate signal with prominent vibrato
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Fixed pitch with strong vibrato
    f0 = 440.0  # A4
    vibrato_rate = 6.0  # Hz
    vibrato_depth = 0.05  # 5%

    pitch = f0 * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    audio = np.sin(2 * np.pi * pitch * t).astype(np.float32)

    print(f"Generated vibrato test signal: f0={f0}Hz, rate={vibrato_rate}Hz, depth={vibrato_depth*100:.1f}%")

    # Initialize detector
    detector = CreepePitchDetector(model_capacity='small', step_size=10)

    # Extract pitch with vibrato analysis
    pitch_data = detector.extract_pitch_with_vibrato_analysis(audio, sample_rate)

    print(f"Vibrato analysis results:")
    print(f"  Frames with vibrato: {np.sum(pitch_data['vibrato_presence'])} "
          f"({np.sum(pitch_data['vibrato_presence'])/len(pitch_data['vibrato_presence'])*100:.1f}%)")

    if np.sum(pitch_data['vibrato_presence']) > 0:
        vibrato_rates = pitch_data['vibrato_rate'][pitch_data['vibrato_presence']]
        vibrato_extents = pitch_data['vibrato_extent'][pitch_data['vibrato_presence']]

        print(f"  Detected vibrato rate: {np.mean(vibrato_rates):.1f}Hz (expected: {vibrato_rate}Hz)")
        print(f"  Detected vibrato extent: {np.mean(vibrato_extents)*100:.1f}% (expected: {vibrato_depth*100:.1f}%)")

    return pitch_data


def main():
    """Run all tests"""
    print("CREPE Pitch Detection Test Suite")
    print("=" * 40)

    try:
        # Test basic functionality
        basic_data = test_crepe_basic()

        # Test vibrato analysis
        vibrato_data = test_vibrato_analysis()

        print("\n=== All Tests Completed Successfully! ===")

        # Save some results for inspection
        print("\nSample pitch data (first 10 frames):")
        for i in range(min(10, len(basic_data['time']))):
            print(f"  t={basic_data['time'][i]:.3f}s: "
                  f"f0={basic_data['frequency'][i]:.1f}Hz, "
                  f"conf={basic_data['confidence'][i]:.3f}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)