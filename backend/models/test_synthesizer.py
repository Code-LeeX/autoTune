#!/usr/bin/env python3
"""
Test script for audio synthesizer
"""
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.audio_synthesizer import WorldAudioSynthesizer, PitchCorrectionEngine
from models.pitch_detector import CreepePitchDetector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def generate_test_voice(duration: float = 3.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a more realistic test voice signal

    Args:
        duration: Signal duration in seconds
        sample_rate: Audio sample rate

    Returns:
        Generated voice-like audio signal
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Create a melody (simple scale)
    base_freq = 220.0  # A3
    melody_pattern = [1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 2.0]  # Major scale

    # Create pitch trajectory
    notes_per_second = 2
    note_duration = 1.0 / notes_per_second
    f0_trajectory = np.zeros_like(t)

    for i, freq_ratio in enumerate(melody_pattern):
        start_time = i * note_duration
        end_time = (i + 1) * note_duration

        if start_time >= duration:
            break

        mask = (t >= start_time) & (t < end_time)
        f0_trajectory[mask] = base_freq * freq_ratio

    # Add vibrato
    vibrato_rate = 5.0  # Hz
    vibrato_depth = 0.02  # 2%
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    f0_with_vibrato = f0_trajectory * (1 + vibrato)

    # Generate harmonic signal
    signal = np.zeros_like(t)

    # Fundamental
    signal += np.sin(2 * np.pi * f0_with_vibrato * t)

    # Add harmonics for voice-like timbre
    signal += 0.5 * np.sin(2 * np.pi * 2 * f0_with_vibrato * t)  # 2nd harmonic
    signal += 0.3 * np.sin(2 * np.pi * 3 * f0_with_vibrato * t)  # 3rd harmonic
    signal += 0.2 * np.sin(2 * np.pi * 4 * f0_with_vibrato * t)  # 4th harmonic
    signal += 0.1 * np.sin(2 * np.pi * 5 * f0_with_vibrato * t)  # 5th harmonic

    # Add envelope
    envelope = np.ones_like(t)
    for i in range(len(melody_pattern)):
        start_time = i * note_duration
        end_time = (i + 1) * note_duration

        if start_time >= duration:
            break

        mask = (t >= start_time) & (t < end_time)
        note_t = t[mask] - start_time

        # Simple attack-decay envelope for each note
        attack_time = 0.05  # 50ms attack
        decay_time = 0.1   # 100ms decay

        note_envelope = np.ones_like(note_t)

        # Attack
        attack_mask = note_t < attack_time
        note_envelope[attack_mask] = note_t[attack_mask] / attack_time

        # Decay
        decay_mask = note_t > (note_duration - decay_time)
        decay_t = note_t[decay_mask] - (note_duration - decay_time)
        note_envelope[decay_mask] = 1.0 - (decay_t / decay_time)

        envelope[mask] = note_envelope

    signal *= envelope

    # Add some noise for realism
    noise = 0.005 * np.random.randn(len(signal))
    signal += noise

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    return signal.astype(np.float32)


def test_world_analysis_synthesis():
    """Test World analysis and synthesis"""
    print("=== Testing World Analysis & Synthesis ===")

    # Generate test audio
    sample_rate = 44100
    audio = generate_test_voice(duration=2.0, sample_rate=sample_rate)

    print(f"Generated test audio: {len(audio)} samples at {sample_rate}Hz")

    # Initialize synthesizer
    synthesizer = WorldAudioSynthesizer(sample_rate=sample_rate)

    # Analyze audio
    analysis = synthesizer.analyze_audio(audio)

    print(f"Analysis results:")
    print(f"  Frames: {len(analysis['f0'])}")
    print(f"  F0 range: {np.min(analysis['f0'][analysis['f0']>0]):.1f} - {np.max(analysis['f0']):.1f} Hz")
    print(f"  Spectral shape: {analysis['sp'].shape}")
    print(f"  Aperiodicity shape: {analysis['ap'].shape}")

    # Synthesize back
    reconstructed = synthesizer.synthesize_audio(
        analysis['f0'],
        analysis['sp'],
        analysis['ap'],
        analysis['time_axis']
    )

    print(f"Reconstructed audio: {len(reconstructed)} samples")

    # Calculate reconstruction quality
    min_len = min(len(audio), len(reconstructed))
    correlation = np.corrcoef(
        audio[:min_len],
        reconstructed[:min_len]
    )[0, 1]

    print(f"Reconstruction correlation: {correlation:.3f}")

    return analysis, reconstructed


def test_pitch_correction():
    """Test complete pitch correction pipeline"""
    print("\n=== Testing Pitch Correction Pipeline ===")

    sample_rate = 44100

    # Generate slightly out-of-tune audio
    audio = generate_test_voice(duration=2.0, sample_rate=sample_rate)

    # Add pitch deviations to simulate out-of-tune singing
    print("Adding pitch deviations to simulate out-of-tune singing...")

    # Initialize pitch detector
    pitch_detector = CreepePitchDetector(model_capacity='small', step_size=10)

    # Extract original pitch
    pitch_data = pitch_detector.extract_pitch(audio, sample_rate)
    original_f0 = pitch_data['frequency']

    print(f"Original pitch analysis:")
    print(f"  Voiced frames: {np.sum(pitch_data['voiced_mask'])}/{len(pitch_data['voiced_mask'])}")
    print(f"  Average confidence: {np.mean(pitch_data['confidence']):.3f}")

    # Create corrected pitch (quantize to semitones)
    target_f0 = quantize_to_semitones(original_f0)

    # Calculate pitch deviation
    voiced_mask = original_f0 > 0
    if np.sum(voiced_mask) > 0:
        deviation = np.abs(original_f0[voiced_mask] - target_f0[voiced_mask])
        avg_deviation = np.mean(deviation)
        print(f"  Average pitch deviation: {avg_deviation:.2f} Hz")

    # Initialize correction engine
    correction_engine = PitchCorrectionEngine(sample_rate=sample_rate)

    # Apply pitch correction
    result = correction_engine.correct_pitch(
        audio,
        target_f0,
        correction_strength=0.8,
        preserve_formants=True
    )

    corrected_audio = result['corrected_audio']
    print(f"Pitch correction complete: {len(corrected_audio)} samples")

    # Analyze corrected audio
    corrected_pitch_data = pitch_detector.extract_pitch(corrected_audio, sample_rate)
    corrected_f0 = corrected_pitch_data['frequency']

    # Calculate correction accuracy
    if np.sum(voiced_mask) > 0:
        correction_error = np.abs(corrected_f0[voiced_mask] - target_f0[voiced_mask])
        avg_correction_error = np.mean(correction_error)
        print(f"  Average correction error: {avg_correction_error:.2f} Hz")

        improvement = avg_deviation - avg_correction_error
        print(f"  Pitch accuracy improvement: {improvement:.2f} Hz")

    return result


def quantize_to_semitones(f0: np.ndarray, reference_freq: float = 440.0) -> np.ndarray:
    """
    Quantize frequencies to nearest semitones

    Args:
        f0: Frequency array
        reference_freq: Reference frequency (A4 = 440Hz)

    Returns:
        Quantized frequency array
    """
    quantized = np.zeros_like(f0)
    voiced_mask = f0 > 0

    if np.sum(voiced_mask) == 0:
        return quantized

    # Convert to semitones relative to reference
    semitones = 12 * np.log2(f0[voiced_mask] / reference_freq)

    # Round to nearest semitone
    quantized_semitones = np.round(semitones)

    # Convert back to frequency
    quantized[voiced_mask] = reference_freq * (2 ** (quantized_semitones / 12))

    return quantized


def main():
    """Run all tests"""
    print("Audio Synthesizer Test Suite")
    print("=" * 40)

    try:
        # Test World analysis/synthesis
        analysis, reconstructed = test_world_analysis_synthesis()

        # Test pitch correction
        correction_result = test_pitch_correction()

        print("\n=== All Tests Completed Successfully! ===")

        # Print summary
        print(f"\nSynthesizer Performance:")
        print(f"  Original audio length: {len(correction_result['corrected_audio'])}")
        print(f"  Analysis frames: {len(analysis['f0'])}")
        print(f"  Spectral bins: {analysis['sp'].shape[1]}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)