"""
Music Generation Script for WaveGAN

This script loads a trained WaveGAN model and generates audio samples.
Supports batch generation, different sample lengths, and audio export.
"""

import argparse
import os
import json
import logging
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import pickle as pk
from datetime import datetime

from wavegan import WaveGANGenerator
from utils import save_samples, write_audio

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_config(model_dir):
    """
    Load model configuration from saved config file

    Args:
        model_dir (str): Directory containing trained model

    Returns:
        dict: Model configuration
    """
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def load_trained_generator(model_dir, use_cuda=True):
    """
    Load a trained WaveGAN generator from saved weights

    Args:
        model_dir (str): Directory containing trained model
        use_cuda (bool): Whether to use CUDA if available

    Returns:
        WaveGANGenerator: Loaded generator model
        dict: Model configuration
    """
    logger.info(f"Loading model from {model_dir}")

    # Load configuration
    config = load_model_config(model_dir)

    # Create generator model
    generator = WaveGANGenerator(
        model_size=config['model_size'],
        ngpus=1 if use_cuda and torch.cuda.is_available() else 0,
        latent_dim=config['latent_dim'],
        post_proc_filt_len=config.get('post_proc_filt_len', 512),
        upsample=True
    )

    # Load trained weights
    model_path = os.path.join(model_dir, 'model_gen.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()

    if use_cuda and torch.cuda.is_available():
        generator = generator.cuda()
        logger.info("Using CUDA for generation")
    else:
        logger.info("Using CPU for generation")

    logger.info(f"Generator loaded successfully")
    logger.info(
        f"Model parameters: size={config['model_size']}, latent_dim={config['latent_dim']}")

    return generator, config


def generate_noise(batch_size, latent_dim, use_cuda=True):
    """
    Generate random noise for the generator

    Args:
        batch_size (int): Number of samples to generate
        latent_dim (int): Dimension of latent space
        use_cuda (bool): Whether to use CUDA

    Returns:
        torch.Tensor: Random noise tensor
    """
    noise = torch.randn(batch_size, latent_dim)

    if use_cuda and torch.cuda.is_available():
        noise = noise.cuda()

    return noise


def generate_audio_samples(generator, num_samples, latent_dim, batch_size=32, use_cuda=True):
    """
    Generate audio samples using the trained generator

    Args:
        generator (WaveGANGenerator): Trained generator model
        num_samples (int): Number of samples to generate
        latent_dim (int): Dimension of latent space
        batch_size (int): Batch size for generation
        use_cuda (bool): Whether to use CUDA

    Returns:
        list: Generated audio samples as numpy arrays
    """
    logger.info(f"Generating {num_samples} audio samples")

    generated_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Calculate batch size for this iteration
            current_batch_size = min(
                batch_size, num_samples - batch_idx * batch_size)

            # Generate random noise
            noise = generate_noise(current_batch_size, latent_dim, use_cuda)

            # Generate audio
            generated_audio = generator(noise)

            # Convert to numpy and add to results
            if use_cuda and torch.cuda.is_available():
                generated_audio = generated_audio.cpu()

            generated_audio = generated_audio.numpy()

            for i in range(current_batch_size):
                # Remove channel dimension
                generated_samples.append(generated_audio[i, 0])

    logger.info(f"Generated {len(generated_samples)} audio samples")
    return generated_samples


def save_audio_samples(samples, output_dir, sample_rate=16000, prefix="generated"):
    """
    Save generated audio samples to files

    Args:
        samples (list): List of audio samples as numpy arrays
        output_dir (str): Directory to save files
        sample_rate (int): Sample rate for audio files
        prefix (str): Prefix for output filenames
    """
    logger.info(f"Saving {len(samples)} audio samples to {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(samples, desc="Saving audio files")):
        # Normalize audio to prevent clipping
        max_amp = np.max(np.abs(sample))
        if max_amp > 0:
            sample = sample / max_amp * 0.95

        # Generate filename
        filename = f"{prefix}_{i:04d}.wav"
        filepath = os.path.join(output_dir, filename)

        # Save audio file using utility function
        try:
            write_audio(filepath, sample.astype(np.float32), sample_rate)
        except Exception as e:
            logger.warning(f"Failed to save {filepath}: {str(e)}")

    logger.info(f"Audio samples saved to {output_dir}")


def create_audio_collage(samples, output_path, sample_rate=16000, samples_per_row=5, pause_duration=0.1):
    """
    Create a collage of multiple audio samples in a single file

    Args:
        samples (list): List of audio samples
        output_path (str): Path for output collage file
        sample_rate (int): Sample rate
        samples_per_row (int): Number of samples per row in collage
        pause_duration (float): Pause duration between samples in seconds
    """
    logger.info(f"Creating audio collage with {len(samples)} samples")

    pause_samples = int(pause_duration * sample_rate)
    pause_audio = np.zeros(pause_samples)

    collage_audio = []

    for i, sample in enumerate(samples):
        # Normalize sample
        max_amp = np.max(np.abs(sample))
        if max_amp > 0:
            sample = sample / max_amp * 0.8

        collage_audio.append(sample)

        # Add pause between samples
        if i < len(samples) - 1:
            collage_audio.append(pause_audio)

        # Add longer pause at end of row
        if (i + 1) % samples_per_row == 0 and i < len(samples) - 1:
            collage_audio.append(pause_audio * 3)  # Longer pause between rows

    # Concatenate all audio
    final_audio = np.concatenate(collage_audio)

    # Save collage using utility function
    try:
        write_audio(output_path, final_audio.astype(np.float32), sample_rate)
        logger.info(f"Audio collage saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save collage: {str(e)}")


def analyze_generated_samples(samples, sample_rate=16000):
    """
    Analyze generated samples and print statistics

    Args:
        samples (list): List of generated audio samples
        sample_rate (int): Sample rate of the audio
    """
    logger.info("Analyzing generated samples...")

    durations = [len(sample) / sample_rate for sample in samples]
    amplitudes = [np.max(np.abs(sample)) for sample in samples]
    rms_values = [np.sqrt(np.mean(sample**2)) for sample in samples]

    print(f"\nGenerated Samples Analysis:")
    print(f"Number of samples: {len(samples)}")
    print(
        f"Average duration: {np.mean(durations):.3f}s (±{np.std(durations):.3f}s)")
    print(f"Sample rate: {sample_rate}Hz")
    print(
        f"Average amplitude: {np.mean(amplitudes):.3f} (±{np.std(amplitudes):.3f})")
    print(
        f"Average RMS: {np.mean(rms_values):.3f} (±{np.std(rms_values):.3f})")
    print(f"Length range: {min(durations):.3f}s - {max(durations):.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Generate audio samples using trained WaveGAN model')

    parser.add_argument('model_dir', type=str,
                        help='Path to directory containing trained model')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of audio samples to generate')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Directory to save generated audio')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--sample_rate', type=int, default=None,
                        help='Sample rate for output audio (uses model default if not specified)')
    parser.add_argument('--prefix', type=str, default='generated',
                        help='Prefix for output filenames')
    parser.add_argument('--create_collage', action='store_true',
                        help='Create an audio collage of all samples')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze and print statistics of generated samples')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible generation')

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # Load trained model
    use_cuda = not args.no_cuda
    generator, config = load_trained_generator(args.model_dir, use_cuda)

    # Determine sample rate
    sample_rate = args.sample_rate or 16000  # Default to 16kHz if not specified

    # Generate audio samples
    generated_samples = generate_audio_samples(
        generator=generator,
        num_samples=args.num_samples,
        latent_dim=config['latent_dim'],
        batch_size=args.batch_size,
        use_cuda=use_cuda
    )

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.prefix}_{timestamp}")

    # Save individual audio files
    save_audio_samples(
        samples=generated_samples,
        output_dir=output_dir,
        sample_rate=sample_rate,
        prefix=args.prefix
    )

    # Create audio collage if requested
    if args.create_collage:
        collage_path = os.path.join(output_dir, f"{args.prefix}_collage.wav")
        create_audio_collage(
            samples=generated_samples,
            output_path=collage_path,
            sample_rate=sample_rate
        )

    # Analyze samples if requested
    if args.analyze:
        analyze_generated_samples(generated_samples, sample_rate)

    # Save generation metadata
    metadata = {
        'model_dir': args.model_dir,
        'num_samples': args.num_samples,
        'sample_rate': sample_rate,
        'batch_size': args.batch_size,
        'generation_time': timestamp,
        'model_config': config,
        'seed': args.seed
    }

    metadata_path = os.path.join(output_dir, 'generation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generation complete! Files saved to: {output_dir}")
    logger.info(f"Generated {len(generated_samples)} audio samples")

    if args.create_collage:
        logger.info(
            f"Audio collage available at: {os.path.join(output_dir, f'{args.prefix}_collage.wav')}")


if __name__ == '__main__':
    main()
