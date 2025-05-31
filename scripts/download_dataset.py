"""
Dataset Download Script for WaveGAN Training

This script downloads and prepares popular audio datasets for WaveGAN training.
Supports multiple datasets including speech commands, drum samples, and piano samples.
"""

import argparse
import os
import requests
import zipfile
import tarfile
import logging
from tqdm import tqdm
import librosa
import numpy as np
from pathlib import Path

# Import local utilities
try:
    from utils import write_audio
except ImportError:
    def write_audio(filepath, audio, sample_rate):
        """Fallback audio writing function"""
        try:
            import soundfile as sf
            sf.write(filepath, audio, sample_rate)
        except ImportError:
            from scipy.io import wavfile
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(filepath, sample_rate, audio_int)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    'sc09': {
        'name': 'Speech Commands v0.02',
        'url': 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        'description': 'Google Speech Commands dataset with single-word commands',
        'size': '2.3GB',
        'sample_rate': 16000,
        'extract_func': 'extract_speech_commands'
    },
    'drums': {
        'name': 'Drum Sample Collection',
        'url': 'https://archive.org/download/freesound_drums/freesound_drums.zip',
        'description': 'Collection of drum samples from Freesound.org',
        'size': '500MB',
        'sample_rate': 44100,
        'extract_func': 'extract_drums'
    },
    'piano': {
        'name': 'Piano Sample Collection',
        'url': 'https://archive.org/download/piano_samples_collection/piano_samples.zip',
        'description': 'Collection of piano note samples',
        'size': '1GB',
        'sample_rate': 44100,
        'extract_func': 'extract_piano'
    }
}


def download_file(url, destination, chunk_size=8192):
    """
    Download a file with progress bar

    Args:
        url (str): URL to download from
        destination (str): Path to save the file
        chunk_size (int): Chunk size for downloading
    """
    logger.info(f"Downloading from {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def extract_archive(archive_path, extract_path):
    """
    Extract an archive file (zip, tar.gz, etc.)

    Args:
        archive_path (str): Path to the archive file
        extract_path (str): Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_path}")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def convert_audio_files(input_dir, output_dir, target_sr=16000, target_format='wav'):
    """
    Convert audio files to consistent format and sample rate

    Args:
        input_dir (str): Input directory with audio files
        output_dir (str): Output directory for converted files
        target_sr (int): Target sample rate
        target_format (str): Target audio format
    """
    logger.info(
        f"Converting audio files to {target_sr}Hz {target_format} format")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Supported audio extensions
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aiff', '.au'}

    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    logger.info(f"Found {len(audio_files)} audio files to convert")

    converted_count = 0
    for audio_file in tqdm(audio_files, desc="Converting audio"):
        try:
            # Load audio file
            audio_data, original_sr = librosa.load(audio_file, sr=target_sr)

            # Skip very short files (less than 0.1 seconds)
            if len(audio_data) < target_sr * 0.1:
                continue

            # Normalize audio
            max_amp = np.max(np.abs(audio_data))
            if max_amp > 0:
                audio_data = audio_data / max_amp

            # Generate output filename
            relative_path = os.path.relpath(audio_file, input_dir)
            output_filename = Path(relative_path).stem + f'.{target_format}'
            output_path = os.path.join(output_dir, output_filename)

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save converted audio using utility function
            write_audio(output_path, audio_data, target_sr)
            converted_count += 1

        except Exception as e:
            logger.warning(f"Failed to convert {audio_file}: {str(e)}")
            continue

    logger.info(f"Successfully converted {converted_count} audio files")


def extract_speech_commands(archive_path, output_dir, target_sr=16000):
    """
    Extract and process Speech Commands dataset

    Args:
        archive_path (str): Path to downloaded archive
        output_dir (str): Output directory
        target_sr (int): Target sample rate
    """
    extract_dir = os.path.join(output_dir, 'extracted')
    extract_archive(archive_path, extract_dir)

    # Process the extracted files
    final_output_dir = os.path.join(output_dir, 'processed')
    convert_audio_files(extract_dir, final_output_dir, target_sr)

    # Clean up extracted directory
    import shutil
    shutil.rmtree(extract_dir)

    logger.info(f"Speech Commands dataset prepared in {final_output_dir}")


def extract_drums(archive_path, output_dir, target_sr=44100):
    """
    Extract and process Drum samples dataset

    Args:
        archive_path (str): Path to downloaded archive
        output_dir (str): Output directory
        target_sr (int): Target sample rate
    """
    extract_dir = os.path.join(output_dir, 'extracted')
    extract_archive(archive_path, extract_dir)

    # Process the extracted files
    final_output_dir = os.path.join(output_dir, 'processed')
    convert_audio_files(extract_dir, final_output_dir, target_sr)

    # Clean up extracted directory
    import shutil
    shutil.rmtree(extract_dir)

    logger.info(f"Drum samples dataset prepared in {final_output_dir}")


def extract_piano(archive_path, output_dir, target_sr=44100):
    """
    Extract and process Piano samples dataset

    Args:
        archive_path (str): Path to downloaded archive
        output_dir (str): Output directory
        target_sr (int): Target sample rate
    """
    extract_dir = os.path.join(output_dir, 'extracted')
    extract_archive(archive_path, extract_dir)

    # Process the extracted files
    final_output_dir = os.path.join(output_dir, 'processed')
    convert_audio_files(extract_dir, final_output_dir, target_sr)

    # Clean up extracted directory
    import shutil
    shutil.rmtree(extract_dir)

    logger.info(f"Piano samples dataset prepared in {final_output_dir}")


def create_sample_dataset(output_dir, num_samples=100, sample_length=1.0, sample_rate=16000):
    """
    Create a small sample dataset for testing purposes

    Args:
        output_dir (str): Output directory
        num_samples (int): Number of sample files to create
        sample_length (float): Length of each sample in seconds
        sample_rate (int): Sample rate
    """
    logger.info(f"Creating sample dataset with {num_samples} files")

    # Create processed directory structure to match other datasets
    processed_dir = os.path.join(output_dir, 'processed')
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Generate simple synthetic audio (sine waves with random frequencies)
        duration = int(sample_length * sample_rate)
        t = np.linspace(0, sample_length, duration)

        # Random frequency between 200-2000 Hz
        freq = np.random.uniform(200, 2000)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        # Add some noise
        noise = 0.1 * np.random.normal(0, 1, duration)
        audio = audio + noise

        # Save as WAV file using utility function
        output_path = os.path.join(processed_dir, f"sample_{i:04d}.wav")
        write_audio(output_path, audio.astype(np.float32), sample_rate)

    logger.info(f"Sample dataset created in {processed_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare audio datasets for WaveGAN training')

    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()) + ['sample'],
                        required=True, help='Dataset to download')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the dataset')
    parser.add_argument('--sample_rate', type=int, default=None,
                        help='Target sample rate (uses dataset default if not specified)')
    parser.add_argument('--keep_archive', action='store_true',
                        help='Keep the downloaded archive file')
    parser.add_argument('--list_datasets', action='store_true',
                        help='List available datasets and exit')

    args = parser.parse_args()

    if args.list_datasets:
        print("\nAvailable datasets:")
        print("-" * 50)
        for key, info in DATASETS.items():
            print(f"{key}: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Default sample rate: {info['sample_rate']}Hz")
            print()
        print("sample: Create a small synthetic dataset for testing")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'sample':
        create_sample_dataset(args.output_dir)
        return

    # Get dataset info
    dataset_info = DATASETS[args.dataset]
    sample_rate = args.sample_rate or dataset_info['sample_rate']

    logger.info(f"Downloading dataset: {dataset_info['name']}")
    logger.info(f"Description: {dataset_info['description']}")
    logger.info(f"Size: {dataset_info['size']}")

    # Download the dataset
    archive_filename = os.path.basename(dataset_info['url'])
    archive_path = os.path.join(args.output_dir, archive_filename)

    if not os.path.exists(archive_path):
        download_file(dataset_info['url'], archive_path)
    else:
        logger.info(f"Archive already exists: {archive_path}")

    # Extract and process the dataset
    extract_func = globals()[dataset_info['extract_func']]
    extract_func(archive_path, args.output_dir, sample_rate)

    # Remove archive if not keeping it
    if not args.keep_archive:
        os.remove(archive_path)
        logger.info(f"Removed archive file: {archive_path}")

    logger.info(
        f"Dataset preparation complete! Dataset saved to: {args.output_dir}/processed")


if __name__ == '__main__':
    main()
