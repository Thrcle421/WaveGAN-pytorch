"""
Generate Longer Music Script for WaveGAN

This script generates longer audio pieces by concatenating multiple WaveGAN samples
and adding smooth transitions between them.
"""

import argparse
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle as pk
from datetime import datetime

from wavegan import WaveGANGenerator
from utils import write_audio

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trained_generator(model_dir, use_cuda=True):
    """Load a trained WaveGAN generator"""
    logger.info(f"Loading model from {model_dir}")

    # Load configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

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
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()

    if use_cuda and torch.cuda.is_available():
        generator = generator.cuda()
        logger.info("Using CUDA for generation")
    else:
        logger.info("Using CPU for generation")

    return generator, config


def generate_single_segment(generator, latent_dim, use_cuda=True):
    """Generate a single audio segment"""
    noise = torch.randn(1, latent_dim)
    if use_cuda and torch.cuda.is_available():
        noise = noise.cuda()

    with torch.no_grad():
        audio = generator(noise)

    if use_cuda and torch.cuda.is_available():
        audio = audio.cpu()

    return audio.numpy()[0, 0]  # Remove batch and channel dimensions


def create_crossfade(audio1, audio2, fade_length=1000):
    """Create a smooth crossfade between two audio segments"""
    # Create fade curves
    fade_out = np.linspace(1, 0, fade_length)
    fade_in = np.linspace(0, 1, fade_length)

    # Apply crossfade
    audio1_end = audio1[-fade_length:] * fade_out
    audio2_start = audio2[:fade_length] * fade_in
    crossfade = audio1_end + audio2_start

    # Combine the segments
    result = np.concatenate([
        audio1[:-fade_length],
        crossfade,
        audio2[fade_length:]
    ])

    return result


def generate_long_audio(generator, config, duration_seconds=10, sample_rate=16000,
                        crossfade_duration=0.1, use_cuda=True):
    """
    Generate a longer audio piece by concatenating multiple segments

    Args:
        generator: Trained WaveGAN generator
        config: Model configuration
        duration_seconds: Desired duration in seconds
        sample_rate: Audio sample rate
        crossfade_duration: Duration of crossfade between segments in seconds
        use_cuda: Whether to use CUDA

    Returns:
        numpy.ndarray: Generated long audio
    """
    segment_length = 16384  # WaveGAN default output length
    segment_duration = segment_length / sample_rate
    crossfade_samples = int(crossfade_duration * sample_rate)

    # Calculate how many segments we need
    effective_segment_length = segment_length - crossfade_samples
    num_segments = int(
        np.ceil(duration_seconds / (effective_segment_length / sample_rate))) + 1

    logger.info(
        f"Generating {num_segments} segments for {duration_seconds}s audio")

    # Generate first segment
    segments = []
    for i in tqdm(range(num_segments), desc="Generating segments"):
        segment = generate_single_segment(
            generator, config['latent_dim'], use_cuda)
        segments.append(segment)

    # Combine segments with crossfades
    logger.info("Combining segments with crossfades...")
    result = segments[0]

    for i in range(1, len(segments)):
        result = create_crossfade(result, segments[i], crossfade_samples)

    # Trim to desired length
    target_samples = int(duration_seconds * sample_rate)
    if len(result) > target_samples:
        result = result[:target_samples]

    # Normalize
    max_amp = np.max(np.abs(result))
    if max_amp > 0:
        result = result / max_amp * 0.95

    logger.info(
        f"Generated audio: {len(result)/sample_rate:.2f}s ({len(result)} samples)")
    return result


def add_reverb_effect(audio, sample_rate=16000, room_size=0.3, damping=0.5):
    """Add a simple reverb effect to audio"""
    # Simple reverb using delayed and attenuated copies
    delays = [0.03, 0.06, 0.09, 0.12]  # Delay times in seconds
    gains = [0.4, 0.3, 0.2, 0.1]       # Gain for each delay

    result = audio.copy()

    for delay, gain in zip(delays, gains):
        delay_samples = int(delay * sample_rate)
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * gain
            result += delayed

    # Normalize
    max_amp = np.max(np.abs(result))
    if max_amp > 0:
        result = result / max_amp * 0.95

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate longer music pieces using trained WaveGAN model')

    parser.add_argument('model_dir', type=str,
                        help='Path to directory containing trained model')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration of generated audio in seconds')
    parser.add_argument('--num_pieces', type=int, default=3,
                        help='Number of different pieces to generate')
    parser.add_argument('--output_dir', type=str, default='./long_generated',
                        help='Directory to save generated audio')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate for output audio')
    parser.add_argument('--crossfade', type=float, default=0.1,
                        help='Crossfade duration between segments in seconds')
    parser.add_argument('--add_reverb', action='store_true',
                        help='Add reverb effect to generated audio')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
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

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"long_music_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating {args.num_pieces} pieces of {args.duration}s each")

    # Generate multiple pieces
    for piece_idx in range(args.num_pieces):
        logger.info(f"Generating piece {piece_idx + 1}/{args.num_pieces}")

        # Generate long audio
        long_audio = generate_long_audio(
            generator=generator,
            config=config,
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
            crossfade_duration=args.crossfade,
            use_cuda=use_cuda
        )

        # Add reverb if requested
        if args.add_reverb:
            logger.info("Adding reverb effect...")
            long_audio = add_reverb_effect(long_audio, args.sample_rate)

        # Save audio file
        filename = f"long_music_{piece_idx:02d}.wav"
        filepath = os.path.join(output_dir, filename)

        try:
            write_audio(filepath, long_audio.astype(
                np.float32), args.sample_rate)
            logger.info(f"Saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {str(e)}")

    # Save generation metadata
    metadata = {
        'model_dir': args.model_dir,
        'duration_seconds': args.duration,
        'num_pieces': args.num_pieces,
        'sample_rate': args.sample_rate,
        'crossfade_duration': args.crossfade,
        'reverb_enabled': args.add_reverb,
        'generation_time': timestamp,
        'model_config': config,
        'seed': args.seed
    }

    metadata_path = os.path.join(output_dir, 'generation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generation complete! Files saved to: {output_dir}")
    logger.info(f"Generated {args.num_pieces} pieces of {args.duration}s each")

    print(f"\n🎵 Long Music Generation Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎶 Generated {args.num_pieces} pieces, each {args.duration}s long")
    print(f"🔊 Sample rate: {args.sample_rate}Hz")
    print(f"✨ Reverb effect: {'Enabled' if args.add_reverb else 'Disabled'}")


if __name__ == '__main__':
    main()
