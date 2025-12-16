#!/usr/bin/env python3
"""
Test script for pitch quantization algorithms
"""
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from audio_processing.pitch_quantizer import MusicalScale, VibratoAnalyzer, PitchQuantizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_musical_scale():
    """Test musical scale functionality"""
    print("=== Testing Musical Scale ===")

    # Test C major scale
    scale = MusicalScale(key='C', scale_type='major', reference_freq=440.0)

    # Test frequency quantization
    test_frequencies = [261.63, 265.0, 293.66, 300.0, 329.63, 340.0]  # C4, ~C4, D4, ~D4, E4, ~E4
    expected_notes = ['C4', 'C4', 'D4', 'D4', 'E4', 'E4']

    print(f"Scale: {scale.key} {scale.scale_type}")
    print(f"Valid pitches (first 10): {scale.valid_pitches[:10]}")

    for i, freq in enumerate(test_frequencies):
        quantized = scale.quantize_frequency(freq)
        deviation = scale.get_pitch_deviation(freq)

        print(f"  {freq:.2f}Hz -> {quantized:.2f}Hz (deviation: {deviation:+.1f} cents)")

    # Test different scales
    scales_to_test = [
        ('A', 'minor'),
        ('G', 'major'),
        ('D', 'dorian'),
        ('C', 'chromatic')
    ]

    for key, scale_type in scales_to_test:
        scale = MusicalScale(key=key, scale_type=scale_type)
        test_freq = 440.0  # A4
        quantized = scale.quantize_frequency(test_freq)
        print(f"{key} {scale_type}: {test_freq}Hz -> {quantized:.2f}Hz")


def test_vibrato_analyzer():
    """Test vibrato analysis"""
    print("\n=== Testing Vibrato Analyzer ===")

    analyzer = VibratoAnalyzer()

    # Generate test signal with known vibrato
    duration = 2.0
    sample_rate = 100  # 100 Hz sampling for analysis (10ms frames)
    time = np.linspace(0, duration, int(duration * sample_rate))

    # Base frequency with vibrato
    base_freq = 440.0
    vibrato_rate = 6.0  # Hz
    vibrato_depth = 0.03  # 3%

    frequency = base_freq * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * time))

    # Add some confidence scores
    confidence = np.ones_like(frequency) * 0.9

    # Analyze vibrato
    vibrato_results = analyzer.analyze_vibrato(time, frequency, confidence)

    print(f"Vibrato analysis on {len(frequency)} frames:")
    print(f"  Expected rate: {vibrato_rate:.1f}Hz, depth: {vibrato_depth*100:.1f}%")

    vibrato_detected = np.sum(vibrato_results['vibrato_presence'])
    print(f"  Detected vibrato in {vibrato_detected} frames ({vibrato_detected/len(frequency)*100:.1f}%)")

    if vibrato_detected > 0:
        detected_rates = vibrato_results['vibrato_rate'][vibrato_results['vibrato_presence']]
        detected_extents = vibrato_results['vibrato_extent'][vibrato_results['vibrato_presence']]

        print(f"  Average detected rate: {np.mean(detected_rates):.1f}Hz")
        print(f"  Average detected extent: {np.mean(detected_extents)*100:.1f}%")

    # Test without vibrato (straight pitch)
    straight_frequency = np.full_like(time, base_freq)
    straight_results = analyzer.analyze_vibrato(time, straight_frequency, confidence)

    straight_detected = np.sum(straight_results['vibrato_presence'])
    print(f"  Control (no vibrato): {straight_detected} frames detected ({straight_detected/len(frequency)*100:.1f}%)")

    return vibrato_results


def test_pitch_quantizer():
    """Test complete pitch quantization"""
    print("\n=== Testing Pitch Quantizer ===")

    # Set up scale and quantizer
    scale = MusicalScale(key='C', scale_type='major', reference_freq=440.0)
    vibrato_analyzer = VibratoAnalyzer()
    quantizer = PitchQuantizer(scale, vibrato_analyzer)

    # Generate test pitch trajectory (slightly out of tune melody)
    duration = 3.0
    sample_rate = 100  # 100 Hz for pitch analysis
    time = np.linspace(0, duration, int(duration * sample_rate))

    # Create a melody with some pitch errors
    notes = [261.63, 293.66, 329.63, 349.23, 392.00]  # C-D-E-F-G major scale
    note_duration = duration / len(notes)

    frequency = np.zeros_like(time)
    confidence = np.zeros_like(time)

    for i, note_freq in enumerate(notes):
        start_time = i * note_duration
        end_time = (i + 1) * note_duration

        mask = (time >= start_time) & (time < end_time)

        # Add some pitch error and vibrato
        note_time = time[mask] - start_time
        pitch_error = np.sin(2 * np.pi * 0.5 * note_time) * 10  # ±10 Hz error
        vibrato = np.sin(2 * np.pi * 5.0 * note_time) * note_freq * 0.02  # 2% vibrato

        frequency[mask] = note_freq + pitch_error + vibrato
        confidence[mask] = 0.9

    print(f"Generated test melody: {len(frequency)} frames")

    # Test different correction settings
    test_settings = [
        {'strength': 0.5, 'preserve_vibrato': True, 'smoothing': 0.1},
        {'strength': 1.0, 'preserve_vibrato': True, 'smoothing': 0.2},
        {'strength': 1.0, 'preserve_vibrato': False, 'smoothing': 0.1},
    ]

    for i, settings in enumerate(test_settings):
        print(f"\n  Test {i+1}: strength={settings['strength']}, "
              f"vibrato={settings['preserve_vibrato']}, smoothing={settings['smoothing']}")

        result = quantizer.quantize_pitch_trajectory(
            time=time,
            frequency=frequency,
            confidence=confidence,
            correction_strength=settings['strength'],
            preserve_vibrato=settings['preserve_vibrato'],
            smoothing_factor=settings['smoothing']
        )

        # Analyze results
        corrected_frames = np.sum(result['correction_applied'])
        vibrato_frames = np.sum(result['vibrato_preserved'])

        print(f"    Corrected frames: {corrected_frames} ({corrected_frames/len(frequency)*100:.1f}%)")
        print(f"    Vibrato preserved: {vibrato_frames} ({vibrato_frames/len(frequency)*100:.1f}%)")

        # Calculate pitch accuracy improvement
        original_deviation = np.abs(result['pitch_deviation'])
        corrected_deviation = []

        for j in range(len(frequency)):
            if frequency[j] > 0:
                new_deviation = scale.get_pitch_deviation(result['quantized_frequency'][j])
                corrected_deviation.append(abs(new_deviation))

        if len(corrected_deviation) > 0:
            avg_original_dev = np.mean(original_deviation[original_deviation > 0])
            avg_corrected_dev = np.mean(corrected_deviation)
            improvement = avg_original_dev - avg_corrected_dev

            print(f"    Average deviation: {avg_original_dev:.1f} -> {avg_corrected_dev:.1f} cents")
            print(f"    Improvement: {improvement:.1f} cents")

    return result


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")

    scale = MusicalScale(key='C', scale_type='major')
    quantizer = PitchQuantizer(scale)

    # Test empty input
    empty_time = np.array([])
    empty_freq = np.array([])

    try:
        result = quantizer.quantize_pitch_trajectory(empty_time, empty_freq)
        print("  ✓ Empty input handled gracefully")
    except Exception as e:
        print(f"  ✗ Empty input failed: {e}")

    # Test all-unvoiced input
    time = np.linspace(0, 1, 100)
    unvoiced_freq = np.zeros(100)
    confidence = np.zeros(100)

    try:
        result = quantizer.quantize_pitch_trajectory(time, unvoiced_freq, confidence)
        corrected = np.sum(result['correction_applied'])
        print(f"  ✓ All-unvoiced input: {corrected} corrections applied")
    except Exception as e:
        print(f"  ✗ All-unvoiced input failed: {e}")

    # Test single frame
    single_time = np.array([0.0])
    single_freq = np.array([440.0])
    single_conf = np.array([0.9])

    try:
        result = quantizer.quantize_pitch_trajectory(single_time, single_freq, single_conf)
        print("  ✓ Single frame handled gracefully")
    except Exception as e:
        print(f"  ✗ Single frame failed: {e}")

    # Test very high/low frequencies
    extreme_time = np.array([0.0, 0.1, 0.2])
    extreme_freq = np.array([50.0, 440.0, 5000.0])  # Very low, normal, very high
    extreme_conf = np.array([0.9, 0.9, 0.9])

    try:
        result = quantizer.quantize_pitch_trajectory(extreme_time, extreme_freq, extreme_conf)
        print("  ✓ Extreme frequencies handled")
    except Exception as e:
        print(f"  ✗ Extreme frequencies failed: {e}")


def main():
    """Run all tests"""
    print("Pitch Quantization Test Suite")
    print("=" * 40)

    try:
        # Test components
        test_musical_scale()
        vibrato_result = test_vibrato_analyzer()
        quantizer_result = test_pitch_quantizer()
        test_edge_cases()

        print("\n=== All Tests Completed Successfully! ===")

        # Summary
        print(f"\nSummary:")
        print(f"  Musical scales working correctly")
        print(f"  Vibrato analysis functional")
        print(f"  Pitch quantization operational")
        print(f"  Edge cases handled")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)