#!/usr/bin/env python3
"""
WaveGAN Baseline Comparison Script

This script compares WaveGAN-generated samples with simple baseline methods:
- White noise
- Pink noise
- Sine waves
- Random sine combinations
- Random selection from training data

Usage:
    python scripts/compare_baselines.py --model_dir ./models/[timestamp] --data_dir ./datasets/piano/processed
"""

import os
import sys
import argparse
import logging
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
from pathlib import Path
import datetime
import json
from scipy import signal

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_audio_samples(directory, max_samples=20, sample_rate=16000):
    """
    Load audio samples from directory

    Args:
        directory (str): Directory containing audio files
        max_samples (int): Maximum number of samples to load
        sample_rate (int): Target sample rate

    Returns:
        list: List of audio samples as numpy arrays
    """
    logger.info(f"Loading audio samples from {directory}")

    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(list(Path(directory).glob(f"*{ext}")))

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {directory}")

    # Randomly select files if there are more than max_samples
    if len(audio_files) > max_samples:
        import random
        audio_files = random.sample(audio_files, max_samples)

    # Load audio files
    samples = []
    for file_path in tqdm(audio_files, desc="Loading audio"):
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate)
            # Ensure consistent length
            if len(audio) > sample_rate * 4:  # Cap at 4 seconds
                audio = audio[:sample_rate * 4]
            samples.append(audio)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(samples)} audio samples")
    return samples


def generate_white_noise(num_samples, sample_rate=16000, duration=3.0):
    """
    Generate white noise samples

    Args:
        num_samples (int): Number of samples to generate
        sample_rate (int): Sample rate
        duration (float): Duration in seconds

    Returns:
        list: List of white noise samples
    """
    logger.info(f"Generating {num_samples} white noise samples")

    samples = []
    for i in range(num_samples):
        # Generate white noise
        audio = np.random.normal(0, 0.1, int(sample_rate * duration))
        # Normalize
        audio = audio / np.max(np.abs(audio))
        samples.append(audio)

    return samples


def generate_pink_noise(num_samples, sample_rate=16000, duration=3.0):
    """
    Generate pink noise samples

    Args:
        num_samples (int): Number of samples to generate
        sample_rate (int): Sample rate
        duration (float): Duration in seconds

    Returns:
        list: List of pink noise samples
    """
    logger.info(f"Generating {num_samples} pink noise samples")

    samples = []
    for i in range(num_samples):
        # Generate white noise
        white_noise = np.random.normal(0, 0.1, int(sample_rate * duration))

        # Create pink noise filter (1/f filter)
        n = len(white_noise)
        X = np.fft.rfft(white_noise)
        S = np.arange(X.size) + 1
        S[0] = 1  # Avoid division by zero
        pink = X / np.sqrt(S)
        audio = np.fft.irfft(pink, n)

        # Normalize
        audio = audio / np.max(np.abs(audio))
        samples.append(audio)

    return samples


def generate_sine_waves(num_samples, sample_rate=16000, duration=3.0):
    """
    Generate simple sine wave samples

    Args:
        num_samples (int): Number of samples to generate
        sample_rate (int): Sample rate
        duration (float): Duration in seconds

    Returns:
        list: List of sine wave samples
    """
    logger.info(f"Generating {num_samples} sine wave samples")

    samples = []
    for i in range(num_samples):
        # Random frequency between 110Hz and 880Hz (A2 to A5)
        freq = np.random.uniform(110, 880)
        t = np.linspace(0, duration, int(
            sample_rate * duration), endpoint=False)
        audio = 0.8 * np.sin(2 * np.pi * freq * t)

        # Apply simple envelope
        envelope = np.ones_like(audio)
        attack = int(0.01 * sample_rate)
        release = int(0.1 * sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        audio = audio * envelope
        samples.append(audio)

    return samples


def generate_random_sines(num_samples, sample_rate=16000, duration=3.0):
    """
    Generate samples with multiple random sine waves

    Args:
        num_samples (int): Number of samples to generate
        sample_rate (int): Sample rate
        duration (float): Duration in seconds

    Returns:
        list: List of random sine combination samples
    """
    logger.info(f"Generating {num_samples} random sine combination samples")

    samples = []
    for i in range(num_samples):
        t = np.linspace(0, duration, int(
            sample_rate * duration), endpoint=False)
        audio = np.zeros_like(t)

        # Add 3-7 sine waves
        num_sines = np.random.randint(3, 8)

        for j in range(num_sines):
            # Random frequency between 110Hz and 1760Hz (A2 to A6)
            freq = np.random.uniform(110, 1760)
            # Random amplitude
            amp = np.random.uniform(0.1, 0.3)
            # Random phase
            phase = np.random.uniform(0, 2 * np.pi)

            audio += amp * np.sin(2 * np.pi * freq * t + phase)

        # Apply simple envelope
        envelope = np.ones_like(audio)
        attack = int(0.05 * sample_rate)
        release = int(0.2 * sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        audio = audio * envelope

        # Normalize
        audio = audio / np.max(np.abs(audio))
        samples.append(audio)

    return samples


def random_selection_from_training(data_dir, num_samples, sample_rate=16000):
    """
    Randomly select samples from the training data

    Args:
        data_dir (str): Directory containing training data
        num_samples (int): Number of samples to select
        sample_rate (int): Sample rate

    Returns:
        list: List of randomly selected samples
    """
    logger.info(f"Selecting {num_samples} random samples from training data")

    # Load all training samples
    all_samples = load_audio_samples(
        data_dir, max_samples=1000, sample_rate=sample_rate)

    if len(all_samples) < num_samples:
        logger.warning(
            f"Only {len(all_samples)} samples available in training data")
        return all_samples

    # Randomly select num_samples
    import random
    selected_samples = random.sample(all_samples, num_samples)

    return selected_samples


def visualize_waveforms(samples_dict, output_dir, num_to_plot=3):
    """
    Visualize waveforms from different methods

    Args:
        samples_dict (dict): Dictionary mapping method names to sample lists
        output_dir (str): Directory to save visualizations
        num_to_plot (int): Number of samples to plot per method
    """
    logger.info("Visualizing waveforms")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot waveforms
    plt.figure(figsize=(15, 10))

    num_methods = len(samples_dict)
    for i, (method, samples) in enumerate(samples_dict.items()):
        for j in range(min(num_to_plot, len(samples))):
            plt.subplot(num_methods, num_to_plot, i * num_to_plot + j + 1)
            plt.plot(samples[j])
            if j == 0:
                plt.ylabel(method, fontsize=12)
            plt.xticks([])
            if i == num_methods - 1:
                plt.xlabel('Time', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=300)
    plt.close()


def visualize_spectrograms(samples_dict, output_dir, sample_rate=16000, num_to_plot=3):
    """
    Visualize spectrograms from different methods

    Args:
        samples_dict (dict): Dictionary mapping method names to sample lists
        output_dir (str): Directory to save visualizations
        sample_rate (int): Sample rate
        num_to_plot (int): Number of samples to plot per method
    """
    logger.info("Visualizing spectrograms")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot spectrograms
    plt.figure(figsize=(15, 12))

    num_methods = len(samples_dict)
    for i, (method, samples) in enumerate(samples_dict.items()):
        for j in range(min(num_to_plot, len(samples))):
            plt.subplot(num_methods, num_to_plot, i * num_to_plot + j + 1)

            # Calculate spectrogram
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(samples[j])), ref=np.max)

            librosa.display.specshow(D, sr=sample_rate,
                                     x_axis='time', y_axis='log')

            if j == 0:
                plt.ylabel(method, fontsize=12)
            if i == 0:
                plt.title(f'Sample {j+1}', fontsize=10)
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'spectrogram_comparison.png'), dpi=300)
    plt.close()


def save_audio_samples(samples_dict, output_dir, sample_rate=16000):
    """
    Save audio samples from different methods

    Args:
        samples_dict (dict): Dictionary mapping method names to sample lists
        output_dir (str): Directory to save samples
        sample_rate (int): Sample rate
    """
    logger.info("Saving audio samples")

    # Create directories for each method
    for method in samples_dict.keys():
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Save samples
        for i, audio in enumerate(samples_dict[method]):
            sf.write(os.path.join(method_dir, f"{method}_{i+1}.wav"),
                     audio, sample_rate)


def compute_statistics(samples_dict, sample_rate=16000):
    """
    Compute statistical measures for samples from different methods

    Args:
        samples_dict (dict): Dictionary mapping method names to sample lists
        sample_rate (int): Sample rate

    Returns:
        dict: Dictionary of statistics for each method
    """
    logger.info("Computing statistics")

    stats = {}

    for method, samples in samples_dict.items():
        method_stats = {}

        # Calculate basic statistics
        audio_lengths = [len(s) for s in samples]
        method_stats['mean_length'] = float(np.mean(audio_lengths))
        method_stats['min_length'] = int(np.min(audio_lengths))
        method_stats['max_length'] = int(np.max(audio_lengths))

        # Calculate audio statistics
        amplitudes = []
        zero_crossings = []
        spectral_centroids = []
        spectral_bandwidths = []

        for audio in samples:
            amplitudes.append(np.mean(np.abs(audio)))
            zero_crossings.append(
                np.sum(librosa.zero_crossings(audio)) / len(audio))
            centroid = librosa.feature.spectral_centroid(
                y=audio, sr=sample_rate)[0]
            spectral_centroids.append(np.mean(centroid))
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=sample_rate)[0]
            spectral_bandwidths.append(np.mean(bandwidth))

        method_stats['mean_amplitude'] = float(np.mean(amplitudes))
        method_stats['mean_zero_crossing_rate'] = float(
            np.mean(zero_crossings))
        method_stats['mean_spectral_centroid'] = float(
            np.mean(spectral_centroids))
        method_stats['mean_spectral_bandwidth'] = float(
            np.mean(spectral_bandwidths))

        stats[method] = method_stats

    return stats


def generate_comparison_report(stats, output_file):
    """
    Generate a report comparing different methods

    Args:
        stats (dict): Dictionary of statistics for each method
        output_file (str): Path to output report file
    """
    logger.info(f"Generating comparison report: {output_file}")

    with open(output_file, 'w') as f:
        f.write("# WaveGAN Baseline Comparison Report\n\n")
        f.write(
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Statistical Comparison\n\n")
        f.write(
            "| Method | Mean Amplitude | Zero Crossing Rate | Spectral Centroid | Spectral Bandwidth |\n")
        f.write(
            "| ------ | -------------- | ------------------ | ----------------- | ------------------ |\n")

        for method, method_stats in stats.items():
            f.write(
                f"| {method} | {method_stats['mean_amplitude']:.4f} | {method_stats['mean_zero_crossing_rate']:.4f} | {method_stats['mean_spectral_centroid']:.1f} | {method_stats['mean_spectral_bandwidth']:.1f} |\n")

        f.write("\n## Analysis\n\n")

        # Compare WaveGAN with baselines
        if 'wavegan' in stats:
            wavegan_stats = stats['wavegan']

            # Find most similar baseline
            similarities = {}
            for method, method_stats in stats.items():
                if method == 'wavegan':
                    continue

                # Calculate similarity based on all metrics
                diff_amp = abs(
                    wavegan_stats['mean_amplitude'] - method_stats['mean_amplitude'])
                diff_zcr = abs(
                    wavegan_stats['mean_zero_crossing_rate'] - method_stats['mean_zero_crossing_rate'])
                diff_sc = abs(
                    wavegan_stats['mean_spectral_centroid'] - method_stats['mean_spectral_centroid'])
                diff_bw = abs(
                    wavegan_stats['mean_spectral_bandwidth'] - method_stats['mean_spectral_bandwidth'])

                # Normalize differences by dividing by the mean of the metrics
                norm_diff_amp = diff_amp / \
                    (wavegan_stats['mean_amplitude'] + 1e-6)
                norm_diff_zcr = diff_zcr / \
                    (wavegan_stats['mean_zero_crossing_rate'] + 1e-6)
                norm_diff_sc = diff_sc / \
                    (wavegan_stats['mean_spectral_centroid'] + 1e-6)
                norm_diff_bw = diff_bw / \
                    (wavegan_stats['mean_spectral_bandwidth'] + 1e-6)

                similarity = 1.0 / (1e-6 + norm_diff_amp +
                                    norm_diff_zcr + norm_diff_sc + norm_diff_bw)
                similarities[method] = similarity

            # Find most and least similar
            most_similar = max(similarities.items(), key=lambda x: x[1])[0]
            least_similar = min(similarities.items(), key=lambda x: x[1])[0]

            f.write("### WaveGAN vs Baselines\n\n")
            f.write(
                f"The WaveGAN-generated samples are most similar to **{most_similar}** in terms of audio characteristics, and least similar to **{least_similar}**.\n\n")

            # Compare specific metrics
            if wavegan_stats['mean_spectral_centroid'] > stats[most_similar]['mean_spectral_centroid'] * 1.2:
                f.write(
                    "WaveGAN samples have a **higher spectral centroid** than similar baselines, indicating more high-frequency content.\n\n")
            elif wavegan_stats['mean_spectral_centroid'] < stats[most_similar]['mean_spectral_centroid'] * 0.8:
                f.write(
                    "WaveGAN samples have a **lower spectral centroid** than similar baselines, indicating more low-frequency content.\n\n")

            if wavegan_stats['mean_spectral_bandwidth'] > stats[most_similar]['mean_spectral_bandwidth'] * 1.2:
                f.write(
                    "WaveGAN samples have a **wider spectral bandwidth** than similar baselines, indicating greater frequency diversity.\n\n")
            elif wavegan_stats['mean_spectral_bandwidth'] < stats[most_similar]['mean_spectral_bandwidth'] * 0.8:
                f.write(
                    "WaveGAN samples have a **narrower spectral bandwidth** than similar baselines, indicating less frequency diversity.\n\n")

        f.write("## Visual Comparison\n\n")
        f.write("### Waveforms\n\n")
        f.write("![Waveform Comparison](./waveform_comparison.png)\n\n")
        f.write("### Spectrograms\n\n")
        f.write("![Spectrogram Comparison](./spectrogram_comparison.png)\n\n")

        f.write("## Conclusion\n\n")
        if 'wavegan' in stats and 'random_selection' in stats:
            # Compare to random selection from training data
            if abs(wavegan_stats['mean_spectral_centroid'] - stats['random_selection']['mean_spectral_centroid']) < stats['random_selection']['mean_spectral_centroid'] * 0.2:
                f.write(
                    "The WaveGAN model successfully produces samples with spectral characteristics similar to the training data.\n\n")
            else:
                f.write(
                    "The WaveGAN model produces samples with spectral characteristics that differ from the training data.\n\n")

            # Make recommendations
            f.write("### Recommendations\n\n")
            if wavegan_stats['mean_amplitude'] < stats['random_selection']['mean_amplitude'] * 0.5:
                f.write(
                    "- Consider adjusting the model to produce samples with higher amplitude.\n")
            if wavegan_stats['mean_spectral_bandwidth'] < stats['random_selection']['mean_spectral_bandwidth'] * 0.7:
                f.write(
                    "- The model might benefit from techniques to increase spectral diversity.\n")
            if wavegan_stats['mean_spectral_centroid'] < stats['random_selection']['mean_spectral_centroid'] * 0.7:
                f.write(
                    "- Consider techniques to enhance high-frequency reproduction.\n")
        else:
            f.write(
                "Compare the audio samples from different methods to evaluate the WaveGAN's performance against baselines.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare WaveGAN samples with baseline methods')

    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing real audio samples')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation/baselines',
                        help='Directory to save comparison results')

    # Evaluation options
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples per method')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['white_noise', 'pink_noise', 'sine_wave',
                                 'random_sines', 'random_selection'],
                        help='Baseline methods to compare')
    parser.add_argument('--generated_dir', type=str, default=None,
                        help='Directory containing pre-generated WaveGAN samples')

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Dictionary to store samples from different methods
        all_samples = {}

        # Get WaveGAN samples
        if args.generated_dir:
            logger.info(
                f"Using pre-generated samples from {args.generated_dir}")
            wavegan_samples = load_audio_samples(
                args.generated_dir, max_samples=args.num_samples, sample_rate=args.sample_rate)
        else:
            logger.warning(
                "No generated_dir provided, cannot compare WaveGAN samples")
            wavegan_samples = []

        if wavegan_samples:
            all_samples['wavegan'] = wavegan_samples

        # Generate baseline samples
        if 'white_noise' in args.methods:
            all_samples['white_noise'] = generate_white_noise(
                args.num_samples, args.sample_rate)

        if 'pink_noise' in args.methods:
            all_samples['pink_noise'] = generate_pink_noise(
                args.num_samples, args.sample_rate)

        if 'sine_wave' in args.methods:
            all_samples['sine_wave'] = generate_sine_waves(
                args.num_samples, args.sample_rate)

        if 'random_sines' in args.methods:
            all_samples['random_sines'] = generate_random_sines(
                args.num_samples, args.sample_rate)

        if 'random_selection' in args.methods:
            all_samples['random_selection'] = random_selection_from_training(
                args.data_dir, args.num_samples, args.sample_rate)

        # Save audio samples
        samples_dir = os.path.join(args.output_dir, 'samples')
        save_audio_samples(all_samples, samples_dir, args.sample_rate)

        # Visualize waveforms
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        visualize_waveforms(all_samples, vis_dir)
        visualize_spectrograms(all_samples, vis_dir, args.sample_rate)

        # Compute statistics
        stats = compute_statistics(all_samples, args.sample_rate)

        # Save statistics
        stats_file = os.path.join(args.output_dir, 'baseline_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Generate comparison report
        report_file = os.path.join(
            args.output_dir, 'baseline_comparison_report.md')
        generate_comparison_report(stats, report_file)

        logger.info(
            f"Comparison complete! Results saved to: {args.output_dir}")
        logger.info(f"Report: {report_file}")

        return 0

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
