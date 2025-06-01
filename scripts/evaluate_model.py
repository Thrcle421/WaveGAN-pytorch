#!/usr/bin/env python3
"""
WaveGAN Objective Evaluation Metrics

This script calculates objective metrics for a WaveGAN model:
- Frechet Audio Distance (FAD)
- Inception Score
- Diversity Score
- Feature visualization

Usage:
    python scripts/evaluate_model.py --model_dir ./models/[timestamp] --data_dir ./datasets/piano/processed
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_audio_samples(directory, max_samples=100, sample_rate=16000):
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


def extract_features(audio_samples, feature_type='mfcc', n_mfcc=20):
    """
    Extract audio features from samples

    Args:
        audio_samples (list): List of audio samples
        feature_type (str): Type of feature to extract ('mfcc', 'mel', 'chroma')
        n_mfcc (int): Number of MFCC coefficients

    Returns:
        numpy.ndarray: Feature matrix
    """
    logger.info(f"Extracting {feature_type} features")

    features = []
    for audio in tqdm(audio_samples, desc=f"Extracting {feature_type}"):
        if feature_type == 'mfcc':
            feat = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc)
        elif feature_type == 'mel':
            feat = librosa.feature.melspectrogram(y=audio)
            feat = librosa.power_to_db(feat)
        elif feature_type == 'chroma':
            feat = librosa.feature.chroma_stft(y=audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Average over time to get a fixed-size representation
        feat_mean = np.mean(feat, axis=1)
        features.append(feat_mean)

    features = np.array(features)
    logger.info(f"Extracted features with shape {features.shape}")
    return features


def calculate_fad(real_features, generated_features):
    """
    Calculate Frechet Audio Distance between real and generated features

    Args:
        real_features (numpy.ndarray): Features from real samples
        generated_features (numpy.ndarray): Features from generated samples

    Returns:
        float: Frechet distance
    """
    logger.info("Calculating Frechet Audio Distance (FAD)")

    # Calculate mean and covariance for real and generated features
    real_mean = np.mean(real_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)

    gen_mean = np.mean(generated_features, axis=0)
    gen_cov = np.cov(generated_features, rowvar=False)

    # Calculate Frechet distance
    mean_diff = real_mean - gen_mean
    mean_term = np.sum(mean_diff ** 2)

    # Handle numerical stability
    eps = 1e-6
    real_cov = real_cov + np.eye(real_cov.shape[0]) * eps
    gen_cov = gen_cov + np.eye(gen_cov.shape[0]) * eps

    # Calculate the matrix square root term
    cov_term = np.trace(real_cov + gen_cov -
                        2 * np.sqrt(np.matmul(real_cov, gen_cov)))

    fad = mean_term + cov_term

    logger.info(f"FAD: {fad:.4f}")
    return fad


def calculate_inception_score(features, n_splits=10):
    """
    Calculate Inception Score for generated samples

    Args:
        features (numpy.ndarray): Features from generated samples
        n_splits (int): Number of splits for score calculation

    Returns:
        tuple: (mean, std) of inception scores
    """
    logger.info("Calculating Inception Score")

    # Normalize features for probability interpretation
    features = features - np.min(features, axis=1, keepdims=True)
    features = features / np.sum(features, axis=1, keepdims=True)

    # Add small constant to avoid log(0)
    features = features + 1e-10

    # Split features into n_splits
    split_size = features.shape[0] // n_splits
    if split_size == 0:
        n_splits = 1
        split_size = features.shape[0]

    scores = []
    for i in range(n_splits):
        start = i * split_size
        end = min((i + 1) * split_size, features.shape[0])
        split_features = features[start:end]

        # Calculate marginal distribution
        p_y = np.mean(split_features, axis=0)

        # Calculate KL divergence
        kl = []
        for j in range(split_features.shape[0]):
            kl.append(entropy(split_features[j], p_y))

        # Calculate score for this split
        score = np.exp(np.mean(kl))
        scores.append(score)

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    logger.info(f"Inception Score: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score


def calculate_diversity(features):
    """
    Calculate diversity of generated samples

    Args:
        features (numpy.ndarray): Features from generated samples

    Returns:
        float: Diversity score
    """
    logger.info("Calculating diversity score")

    # Calculate pairwise distances
    distances = pairwise_distances(features, metric='euclidean')

    # Calculate diversity as the average distance
    diversity = np.mean(distances)
    # Normalize by feature dimension
    diversity = diversity / np.sqrt(features.shape[1])

    logger.info(f"Diversity: {diversity:.4f}")
    return diversity


def visualize_features(real_features, generated_features, output_dir):
    """
    Visualize feature distributions

    Args:
        real_features (numpy.ndarray): Features from real samples
        generated_features (numpy.ndarray): Features from generated samples
        output_dir (str): Directory to save visualizations
    """
    logger.info("Visualizing feature distributions")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Flatten features for histogram
    real_flat = real_features.flatten()
    gen_flat = generated_features.flatten()

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(real_flat, bins=50, alpha=0.5, label='Real', density=True)
    plt.hist(gen_flat, bins=50, alpha=0.5, label='Generated', density=True)
    plt.legend()
    plt.title('Feature Distribution Comparison')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, 'feature_distribution.png'), dpi=300)
    plt.close()

    # Plot feature means
    real_mean = np.mean(real_features, axis=0)
    gen_mean = np.mean(generated_features, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(real_mean, label='Real', alpha=0.7)
    plt.plot(gen_mean, label='Generated', alpha=0.7)
    plt.legend()
    plt.title('Feature Means Comparison')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.savefig(os.path.join(output_dir, 'feature_means.png'), dpi=300)
    plt.close()

    # Plot feature correlation matrices
    real_corr = np.corrcoef(real_features, rowvar=False)
    gen_corr = np.corrcoef(generated_features, rowvar=False)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Real Features Correlation')

    plt.subplot(1, 2, 2)
    plt.imshow(gen_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Generated Features Correlation')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300)
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def generate_audio_samples(model_dir, output_dir, num_samples=100, sample_rate=16000):
    """
    Generate audio samples from model

    Args:
        model_dir (str): Directory containing trained model
        output_dir (str): Directory to save generated samples
        num_samples (int): Number of samples to generate
        sample_rate (int): Audio sample rate

    Returns:
        list: List of generated audio samples
    """
    logger.info(f"Generating {num_samples} audio samples...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Call our custom generation script
    try:
        import subprocess
        cmd = [
            sys.executable, 'scripts/generate_piano_samples.py',
            '--model_dir', model_dir,
            '--output_dir', output_dir,
            '--num_samples', str(num_samples),
            '--sample_rate', str(sample_rate)
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Load the generated samples
        samples = load_audio_samples(
            output_dir, max_samples=num_samples, sample_rate=sample_rate)
        return samples

    except Exception as e:
        logger.error(f"Error generating samples: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Calculate objective metrics for WaveGAN model')

    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing real audio samples')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Directory to save evaluation results')

    # Evaluation options
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'mel', 'chroma'],
                        help='Type of audio feature to extract')
    parser.add_argument('--generated_dir', type=str, default=None,
                        help='Directory containing pre-generated samples (skips generation step)')

    args = parser.parse_args()

    try:
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)

        # Load real samples
        real_samples = load_audio_samples(
            args.data_dir, max_samples=args.max_samples, sample_rate=args.sample_rate)

        # Generate or load generated samples
        if args.generated_dir:
            logger.info(
                f"Using pre-generated samples from {args.generated_dir}")
            generated_samples = load_audio_samples(
                args.generated_dir, max_samples=args.max_samples, sample_rate=args.sample_rate)
        else:
            # Create subdirectory for generated samples
            gen_dir = os.path.join(eval_dir, 'generated_samples')
            generated_samples = generate_audio_samples(
                model_dir=args.model_dir,
                output_dir=gen_dir,
                num_samples=args.max_samples,
                sample_rate=args.sample_rate
            )

        # Extract features
        real_features = extract_features(
            real_samples, feature_type=args.feature_type)
        generated_features = extract_features(
            generated_samples, feature_type=args.feature_type)

        # Calculate metrics
        fad = calculate_fad(real_features, generated_features)
        inception_score, inception_std = calculate_inception_score(
            generated_features)
        diversity = calculate_diversity(generated_features)

        # Visualize features
        vis_dir = os.path.join(eval_dir, 'visualizations')
        visualize_features(real_features, generated_features, vis_dir)

        # Save metrics
        metrics = {
            'fad': float(fad),
            'inception_score': float(inception_score),
            'inception_std': float(inception_std),
            'diversity': float(diversity),
            'feature_type': args.feature_type,
            'n_real_samples': len(real_samples),
            'n_generated_samples': len(generated_samples),
            'sample_rate': args.sample_rate,
            'timestamp': timestamp
        }

        metrics_file = os.path.join(eval_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create report
        report_file = os.path.join(eval_dir, 'evaluation_report.md')
        with open(report_file, 'w') as f:
            f.write("# WaveGAN Objective Evaluation Report\n\n")
            f.write(
                f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Objective Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("| ------ | ----- |\n")
            f.write(f"| Frechet Audio Distance (FAD) | {fad:.4f} |\n")
            f.write(
                f"| Inception Score | {inception_score:.4f} ± {inception_std:.4f} |\n")
            f.write(f"| Diversity Score | {diversity:.4f} |\n\n")

            f.write("## Interpretation\n\n")

            # FAD interpretation
            if fad < 5.0:
                fad_quality = "excellent"
            elif fad < 10.0:
                fad_quality = "good"
            elif fad < 20.0:
                fad_quality = "fair"
            else:
                fad_quality = "poor"

            f.write(
                f"- **FAD Score** ({fad:.2f}) indicates {fad_quality} quality compared to real samples.\n")

            # Inception Score interpretation
            if inception_score > 3.0:
                is_quality = "excellent"
            elif inception_score > 2.0:
                is_quality = "good"
            elif inception_score > 1.5:
                is_quality = "fair"
            else:
                is_quality = "poor"

            f.write(
                f"- **Inception Score** ({inception_score:.2f}) indicates {is_quality} diversity and quality.\n")

            # Diversity interpretation
            if diversity > 0.7:
                div_quality = "very high"
            elif diversity > 0.5:
                div_quality = "high"
            elif diversity > 0.3:
                div_quality = "moderate"
            else:
                div_quality = "low"

            f.write(
                f"- **Sample Diversity** ({diversity:.2f}) is {div_quality}.\n\n")

            f.write("## Visualizations\n\n")
            f.write(
                "![Feature Distribution](./visualizations/feature_distribution.png)\n\n")
            f.write("![Feature Means](./visualizations/feature_means.png)\n\n")
            f.write(
                "![Feature Correlation](./visualizations/feature_correlation.png)\n\n")

        logger.info(f"Evaluation complete! Results saved to: {eval_dir}")
        logger.info(f"Metrics: {metrics}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
