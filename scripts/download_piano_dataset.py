#!/usr/bin/env python3
"""
Piano Dataset Generator for WaveGAN

This script creates a dataset of piano samples by:
1. Generating synthetic piano sounds with harmonics
2. Processing them for WaveGAN training

Usage:
    python scripts/download_piano_dataset.py --output_dir ./datasets/piano --num_samples 500
"""

import os
import sys
import argparse
import numpy as np
import random
import soundfile as sf
from tqdm import tqdm
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_piano_sample(sample_rate=16000, duration_range=(0.5, 3.0)):
    """
    Generate a synthetic piano-like tone

    Args:
        sample_rate: Sample rate in Hz
        duration_range: Range of durations (min, max) in seconds

    Returns:
        numpy.ndarray: Generated audio sample
    """
    # Create synthetic piano-like tone
    duration = random.uniform(*duration_range)  # Random duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Random note frequency (A0 to C7)
    base_freq = random.uniform(27.5, 2093)

    # Create piano-like envelope with fast attack, longer decay
    attack = int(0.005 * sample_rate)
    decay = int(random.uniform(0.1, 0.5) * sample_rate)
    release = int(random.uniform(0.5, 1.5) * sample_rate)

    # Ensure we don't exceed the signal length
    total_length = len(t)
    sustain_point = attack + decay
    release_point = max(sustain_point, total_length - release)

    # Create envelope
    envelope = np.zeros_like(t)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if sustain_point > attack:
        decay_part = np.linspace(1, 0.7, min(decay, sustain_point - attack))
        envelope[attack:sustain_point] = decay_part
    if release_point < total_length:
        release_part = np.linspace(
            envelope[release_point], 0, total_length - release_point)
        envelope[release_point:] = release_part

    # Generate piano-like sound with harmonics
    audio = np.zeros_like(t)

    # Fundamental
    audio += 0.7 * np.sin(2 * np.pi * base_freq * t) * envelope

    # Harmonics with decreasing amplitude
    for h in range(2, 20):
        harmonic_amp = 0.7 * (1/h) * random.uniform(0.2, 1.0)
        audio += harmonic_amp * \
            np.sin(2 * np.pi * base_freq * h * t) * envelope

    # Add slight inharmonicity (stretched harmonics) - characteristic of piano strings
    for h in range(2, 10):
        stretch_factor = 1 + (0.0005 * h * h)
        stretched_freq = base_freq * h * stretch_factor
        stretch_amp = 0.1 * (1/h) * random.uniform(0.1, 0.5)
        audio += stretch_amp * \
            np.sin(2 * np.pi * stretched_freq * t) * envelope

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Add subtle noise to simulate hammer and resonance
    noise_level = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(audio)) * noise_level * envelope
    audio += noise

    # Final normalization
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    return audio


def generate_piano_chord(sample_rate=16000, duration_range=(1.0, 4.0)):
    """
    Generate a synthetic piano chord

    Args:
        sample_rate: Sample rate in Hz
        duration_range: Range of durations (min, max) in seconds

    Returns:
        numpy.ndarray: Generated audio sample
    """
    # Create synthetic piano chord
    duration = random.uniform(*duration_range)  # Random duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Base frequency for the chord (C2 to C5)
    base_freq = random.uniform(65.41, 523.25)

    # Create chord intervals (major, minor, etc.)
    chord_types = [
        [1, 5/4, 3/2],  # Major
        [1, 6/5, 3/2],  # Minor
        [1, 5/4, 3/2, 2],  # Major 7th
        [1, 6/5, 3/2, 9/5],  # Minor 7th
        [1, 5/4, 3/2, 7/4],  # Dominant 7th
        [1, 5/4, 3/2, 7/4, 2],  # Major 9th
    ]

    chord_intervals = random.choice(chord_types)

    # Create piano-like envelope
    attack = int(0.01 * sample_rate)
    decay = int(random.uniform(0.2, 0.8) * sample_rate)
    release = int(random.uniform(1.0, 2.0) * sample_rate)

    # Ensure we don't exceed the signal length
    total_length = len(t)
    sustain_point = attack + decay
    release_point = max(sustain_point, total_length - release)

    # Create envelope
    envelope = np.zeros_like(t)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if sustain_point > attack:
        decay_part = np.linspace(1, 0.6, min(decay, sustain_point - attack))
        envelope[attack:sustain_point] = decay_part
    if release_point < total_length:
        release_part = np.linspace(
            envelope[release_point], 0, total_length - release_point)
        envelope[release_point:] = release_part

    # Generate chord
    audio = np.zeros_like(t)

    # Add each note of the chord
    for interval in chord_intervals:
        note_freq = base_freq * interval

        # Slightly vary the timing of each note
        note_offset = int(random.uniform(0, 0.05) * sample_rate)
        note_env = np.zeros_like(envelope)
        if note_offset < len(note_env):
            note_env[note_offset:] = envelope[:(len(envelope)-note_offset)]

        # Add fundamental
        audio += 0.3 * np.sin(2 * np.pi * note_freq * t) * note_env

        # Add harmonics
        for h in range(2, 10):
            harmonic_amp = 0.3 * (1/h) * random.uniform(0.2, 1.0)
            audio += harmonic_amp * \
                np.sin(2 * np.pi * note_freq * h * t) * note_env

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Add subtle noise
    noise_level = random.uniform(0.001, 0.008)
    noise = np.random.randn(len(audio)) * noise_level * envelope
    audio += noise

    # Final normalization
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    return audio


def generate_piano_melody(sample_rate=16000, duration_range=(3.0, 8.0), n_notes=None):
    """
    Generate a synthetic piano melody

    Args:
        sample_rate: Sample rate in Hz
        duration_range: Range of durations (min, max) in seconds
        n_notes: Number of notes in the melody (if None, will be calculated based on duration)

    Returns:
        numpy.ndarray: Generated audio sample
    """
    # Create synthetic piano melody
    duration = random.uniform(*duration_range)  # Random duration

    if n_notes is None:
        # Calculate number of notes based on duration (2-4 notes per second)
        n_notes = int(duration * random.uniform(2, 4))

    # Create scales to choose from
    scales = [
        [0, 2, 4, 5, 7, 9, 11],  # Major
        [0, 2, 3, 5, 7, 8, 10],  # Minor natural
        [0, 2, 3, 5, 7, 9, 11],  # Minor melodic (ascending)
        [0, 2, 3, 5, 7, 8, 11],  # Harmonic minor
        [0, 2, 4, 6, 7, 9, 11],  # Lydian
        [0, 2, 4, 5, 7, 9, 10],  # Mixolydian
    ]

    # Choose a scale and root note
    scale = random.choice(scales)
    root_freq = random.uniform(110, 440)  # A2 to A4

    # Generate melody
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    # Divide duration into segments for notes
    note_duration = duration / n_notes

    for i in range(n_notes):
        # Select note from scale (can jump up/down by octaves occasionally)
        scale_degree = random.choice(scale)
        # Mostly stay in same octave
        octave_shift = random.choice([-1, 0, 0, 0, 1])

        # Calculate frequency using equal temperament
        semitones = scale_degree + octave_shift * 12
        freq = root_freq * (2 ** (semitones / 12))

        # Note timing
        start_time = i * note_duration
        note_actual_duration = random.uniform(
            0.7, 1.0) * note_duration  # Some articulation

        # Convert to samples
        start_sample = int(start_time * sample_rate)
        note_samples = int(note_actual_duration * sample_rate)

        # Create note envelope
        attack = int(0.01 * sample_rate)
        decay = int(0.1 * sample_rate)
        release = int(0.2 * sample_rate)

        # Create per-note envelope
        envelope = np.zeros(note_samples)
        if attack > 0 and attack < note_samples:
            envelope[:attack] = np.linspace(0, 1, attack)
        sustain_point = min(attack + decay, note_samples)
        if sustain_point > attack:
            decay_part = np.linspace(
                1, 0.7, min(decay, sustain_point - attack))
            envelope[attack:sustain_point] = decay_part
        release_point = max(sustain_point, note_samples - release)
        if release_point < note_samples:
            release_part = np.linspace(
                envelope[release_point-1], 0, note_samples - release_point)
            envelope[release_point:] = release_part
        if sustain_point < release_point:
            envelope[sustain_point:release_point] = 0.7

        # Generate the note with its envelope
        t_note = np.linspace(0, note_actual_duration,
                             note_samples, endpoint=False)
        note = np.zeros_like(t_note)

        # Fundamental
        note += 0.7 * np.sin(2 * np.pi * freq * t_note)

        # Harmonics
        for h in range(2, 15):
            harmonic_amp = 0.7 * (1/h) * random.uniform(0.2, 1.0)
            note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note)

        # Apply envelope
        note = note * envelope

        # Add to main audio array
        end_sample = min(start_sample + note_samples, len(audio))
        audio_segment = audio[start_sample:end_sample]
        note_segment = note[:len(audio_segment)]
        audio[start_sample:end_sample] += note_segment

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Add subtle noise for realism
    noise_level = random.uniform(0.001, 0.005)
    noise = np.random.randn(len(audio)) * noise_level
    audio += noise

    # Final normalization
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    return audio


def generate_piano_dataset(output_dir, num_samples, sample_rate=16000):
    """
    Generate a dataset of piano samples

    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
    """
    # Create output directories
    raw_dir = os.path.join(output_dir, 'raw')
    processed_dir = os.path.join(output_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Calculate samples per type
    n_single_notes = int(num_samples * 0.5)  # 50% single notes
    n_chords = int(num_samples * 0.3)        # 30% chords
    n_melodies = num_samples - n_single_notes - n_chords  # 20% melodies

    logger.info(
        f"Generating {n_single_notes} single notes, {n_chords} chords, and {n_melodies} melodies")

    # Generate single note samples
    for i in tqdm(range(n_single_notes), desc="Generating single notes"):
        audio = generate_piano_sample(sample_rate=sample_rate)
        filename = f"piano_note_{i:04d}.wav"
        sf.write(os.path.join(processed_dir, filename), audio, sample_rate)

    # Generate chord samples
    for i in tqdm(range(n_chords), desc="Generating chords"):
        audio = generate_piano_chord(sample_rate=sample_rate)
        filename = f"piano_chord_{i:04d}.wav"
        sf.write(os.path.join(processed_dir, filename), audio, sample_rate)

    # Generate melody samples
    for i in tqdm(range(n_melodies), desc="Generating melodies"):
        audio = generate_piano_melody(sample_rate=sample_rate)
        filename = f"piano_melody_{i:04d}.wav"
        sf.write(os.path.join(processed_dir, filename), audio, sample_rate)

    # Save examples of each type to raw directory
    sf.write(os.path.join(raw_dir, "example_note.wav"),
             generate_piano_sample(sample_rate=sample_rate), sample_rate)
    sf.write(os.path.join(raw_dir, "example_chord.wav"),
             generate_piano_chord(sample_rate=sample_rate), sample_rate)
    sf.write(os.path.join(raw_dir, "example_melody.wav"),
             generate_piano_melody(sample_rate=sample_rate), sample_rate)

    logger.info(f"Generated {num_samples} piano samples at {processed_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate a dataset of piano samples for WaveGAN training')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the dataset')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to generate')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate in Hz')

    args = parser.parse_args()

    try:
        # Generate dataset
        output_dir = generate_piano_dataset(
            args.output_dir, args.num_samples, args.sample_rate)

        logger.info("Piano dataset generation complete")
        logger.info(f"Dataset location: {output_dir}")
        logger.info(f"Total samples: {args.num_samples}")

        return 0
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
