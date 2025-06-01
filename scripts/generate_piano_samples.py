#!/usr/bin/env python3
"""
Piano Sample Generator from Trained WaveGAN Model

This script loads a trained WaveGAN model and generates piano samples.
It doesn't import wavegan as a module but loads model files directly.

Usage:
    python scripts/generate_piano_samples.py --model_dir ./models/[timestamp] --output_dir ./output/samples
"""

import os
import sys
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ConstantPad1d(
                reflection_padding, value=0)
            self.conv1d = torch.nn.Conv1d(
                in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv1d(out)
        return out


# Define WaveGAN Generator architecture
class WaveGANGenerator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, latent_dim=100,
                 post_proc_filt_len=512, verbose=False, upsample=True):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.latent_dim = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        self.fc1 = nn.DataParallel(nn.Linear(latent_dim, 256 * model_size))

        self.tconv1 = None
        self.tconv2 = None
        self.tconv3 = None
        self.tconv4 = None
        self.tconv5 = None

        self.upSampConv1 = None
        self.upSampConv2 = None
        self.upSampConv3 = None
        self.upSampConv4 = None
        self.upSampConv5 = None

        self.upsample = upsample

        if self.upsample:
            self.upSampConv1 = nn.DataParallel(
                UpsampleConvLayer(16 * model_size, 8 * model_size, 25, stride=1, upsample=4))
            self.upSampConv2 = nn.DataParallel(
                UpsampleConvLayer(8 * model_size, 4 * model_size, 25, stride=1, upsample=4))
            self.upSampConv3 = nn.DataParallel(
                UpsampleConvLayer(4 * model_size, 2 * model_size, 25, stride=1, upsample=4))
            self.upSampConv4 = nn.DataParallel(
                UpsampleConvLayer(2 * model_size, model_size, 25, stride=1, upsample=4))
            self.upSampConv5 = nn.DataParallel(
                UpsampleConvLayer(model_size, num_channels, 25, stride=1, upsample=4))

        else:
            self.tconv1 = nn.DataParallel(
                nn.ConvTranspose1d(16 * model_size, 8 * model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv2 = nn.DataParallel(
                nn.ConvTranspose1d(8 * model_size, 4 * model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv3 = nn.DataParallel(
                nn.ConvTranspose1d(4 * model_size, 2 * model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv4 = nn.DataParallel(
                nn.ConvTranspose1d(2 * model_size, model_size, 25, stride=4, padding=11,
                                   output_padding=1))
            self.tconv5 = nn.DataParallel(
                nn.ConvTranspose1d(model_size, num_channels, 25, stride=4, padding=11,
                                   output_padding=1))

        if post_proc_filt_len:
            self.ppfilter1 = nn.DataParallel(
                nn.Conv1d(num_channels, num_channels, post_proc_filt_len))

    def forward(self, x):
        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        x = F.relu(x)
        output = None

        if self.upsample:
            x = F.relu(self.upSampConv1(x))
            x = F.relu(self.upSampConv2(x))
            x = F.relu(self.upSampConv3(x))
            x = F.relu(self.upSampConv4(x))
            output = F.tanh(self.upSampConv5(x))
        else:
            x = F.relu(self.tconv1(x))
            x = F.relu(self.tconv2(x))
            x = F.relu(self.tconv3(x))
            x = F.relu(self.tconv4(x))
            output = F.tanh(self.tconv5(x))

        if self.post_proc_filt_len:
            # Pad for "same" filtering
            if (self.post_proc_filt_len % 2) == 0:
                pad_left = self.post_proc_filt_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filt_len - 1) // 2
                pad_right = pad_left
            output = self.ppfilter1(F.pad(output, (pad_left, pad_right)))

        return output


def load_model(model, model_path):
    """
    Load model weights from file

    Args:
        model: PyTorch model
        model_path: Path to model weights
    """
    logger.info(f"Loading model from {model_path}")

    try:
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(
                model_path, map_location=torch.device('cpu'))

        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_samples(model, num_samples, device, latent_dim=100):
    """
    Generate audio samples from model

    Args:
        model: PyTorch generator model
        num_samples: Number of samples to generate
        device: Torch device
        latent_dim: Dimension of latent vector

    Returns:
        list: List of generated audio samples as numpy arrays
    """
    logger.info(f"Generating {num_samples} samples...")

    model.eval()
    samples = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating"):
            # Sample random latent vector
            z = torch.randn(1, latent_dim, device=device)

            # Generate audio
            output = model(z)

            # Convert to numpy array
            audio = output.cpu().numpy()[0, 0, :]
            samples.append(audio)

    return samples


def save_samples(samples, output_dir, sample_rate=16000, add_suffix=True):
    """
    Save audio samples to files

    Args:
        samples: List of audio samples
        output_dir: Directory to save samples
        sample_rate: Audio sample rate
        add_suffix: Whether to add a suffix to the filename
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, audio in enumerate(tqdm(samples, desc="Saving samples")):
        if add_suffix:
            filename = f"generated_{i+1}.wav"
        else:
            filename = f"piano_sample_{i+1}.wav"

        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, audio, sample_rate)

    # Also create a combined audio file
    concat_audio = np.concatenate(samples[:min(10, len(samples))])
    sf.write(os.path.join(output_dir, "combined_samples.wav"),
             concat_audio, sample_rate)

    logger.info(f"Saved {len(samples)} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate piano samples from a trained WaveGAN model')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--model_size', type=int, default=64,
                        help='Model size parameter')

    args = parser.parse_args()

    try:
        # Locate model files
        model_gen_path = os.path.join(args.model_dir, 'model_gen.pkl')
        config_path = os.path.join(args.model_dir, 'config.json')

        if not os.path.exists(model_gen_path):
            logger.error(f"Generator model not found at {model_gen_path}")
            return 1

        # Load model configuration
        model_size = args.model_size
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_size = config.get('model_size', args.model_size)
                logger.info(f"Using model size from config: {model_size}")

        # Set device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Create generator model
        generator = WaveGANGenerator(model_size=model_size)
        generator.to(device)

        # Load model weights
        load_model(generator, model_gen_path)

        # Generate samples
        samples = generate_samples(generator, args.num_samples, device)

        # Save samples
        save_samples(samples, args.output_dir, args.sample_rate)

        logger.info("Sample generation complete!")
        return 0

    except Exception as e:
        logger.error(f"Error generating samples: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
