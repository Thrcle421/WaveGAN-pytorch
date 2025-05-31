"""
Complete WaveGAN Pipeline Script

This script runs the complete pipeline:
1. Download and prepare dataset
2. Train the WaveGAN model
3. Generate audio samples from the trained model

Designed for easy end-to-end execution.
"""

import argparse
import os
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description, check=True):
    """
    Run a command and handle errors

    Args:
        cmd (list): Command to run as list of strings
        description (str): Description of what the command does
        check (bool): Whether to raise exception on error

    Returns:
        subprocess.CompletedProcess: Result of command execution
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=check, capture_output=True, text=True)

        if result.stdout:
            logger.info(f"Output: {result.stdout}")

        if result.stderr and result.returncode == 0:
            logger.info(f"Warnings: {result.stderr}")

        logger.info(f"✓ {description} completed successfully")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_dependencies():
    """
    Check if required dependencies are installed

    Returns:
        bool: True if all dependencies are available
    """
    logger.info("Checking dependencies...")

    required_packages = ['torch', 'numpy',
                         'librosa', 'pescador', 'requests', 'tqdm']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            logger.error(f"✗ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install " +
                     ' '.join(missing_packages))
        return False

    logger.info("All dependencies are available")
    return True


def download_dataset(dataset_name, data_dir, sample_rate=None):
    """
    Download and prepare dataset

    Args:
        dataset_name (str): Name of dataset to download
        data_dir (str): Directory to save dataset
        sample_rate (int): Target sample rate

    Returns:
        str: Path to processed dataset directory
    """
    logger.info(f"Starting dataset download: {dataset_name}")

    cmd = [sys.executable, 'download_dataset.py',
           '--dataset', dataset_name, '--output_dir', data_dir]

    if sample_rate:
        cmd.extend(['--sample_rate', str(sample_rate)])

    run_command(cmd, f"Download {dataset_name} dataset")

    processed_dir = os.path.join(data_dir, 'processed')
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dir}")

    # Count audio files
    audio_files = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(root, file))

    logger.info(f"Dataset prepared with {len(audio_files)} audio files")
    return processed_dir


def train_model(audio_dir, output_dir, epochs=50, batch_size=32, model_size=64):
    """
    Train the WaveGAN model

    Args:
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory to save trained model
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        model_size (int): Model size parameter

    Returns:
        str: Path to trained model directory
    """
    logger.info(f"Starting model training for {epochs} epochs")

    cmd = [
        sys.executable, 'train_wavegan.py',
        audio_dir, output_dir,
        '--num-epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--model-size', str(model_size),
        '--ngpus', '1' if check_cuda_available() else '0'
    ]

    run_command(cmd, f"Train WaveGAN model for {epochs} epochs")

    # Find the most recent model directory
    model_dirs = [d for d in os.listdir(
        output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {output_dir}")

    # Get the most recent directory (assuming timestamp naming)
    latest_model_dir = os.path.join(output_dir, sorted(model_dirs)[-1])

    # Check if model files exist
    model_gen_path = os.path.join(latest_model_dir, 'model_gen.pkl')
    config_path = os.path.join(latest_model_dir, 'config.json')

    if not os.path.exists(model_gen_path):
        raise FileNotFoundError(
            f"Generator model not found at {model_gen_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    logger.info(
        f"Model training completed. Model saved to: {latest_model_dir}")
    return latest_model_dir


def generate_music(model_dir, output_dir, num_samples=20, sample_rate=16000):
    """
    Generate music samples from trained model

    Args:
        model_dir (str): Directory containing trained model
        output_dir (str): Directory to save generated audio
        num_samples (int): Number of samples to generate
        sample_rate (int): Sample rate for output audio

    Returns:
        str: Path to generated audio directory
    """
    logger.info(f"Generating {num_samples} audio samples")

    cmd = [
        sys.executable, 'generate_music.py',
        model_dir,
        '--num_samples', str(num_samples),
        '--output_dir', output_dir,
        '--sample_rate', str(sample_rate),
        '--create_collage',
        '--analyze'
    ]

    run_command(cmd, f"Generate {num_samples} audio samples")

    # Find the generated audio directory
    generation_dirs = [d for d in os.listdir(
        output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not generation_dirs:
        raise FileNotFoundError(
            f"No generation directories found in {output_dir}")

    latest_generation_dir = os.path.join(
        output_dir, sorted(generation_dirs)[-1])

    # Count generated files
    audio_files = [f for f in os.listdir(
        latest_generation_dir) if f.endswith('.wav')]
    logger.info(
        f"Generated {len(audio_files)} audio files in: {latest_generation_dir}")

    return latest_generation_dir


def check_cuda_available():
    """
    Check if CUDA is available for training

    Returns:
        bool: True if CUDA is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def print_summary(dataset_name, data_dir, model_dir, generated_dir, total_time):
    """
    Print a summary of the completed pipeline

    Args:
        dataset_name (str): Name of dataset used
        data_dir (str): Path to dataset directory
        model_dir (str): Path to trained model directory
        generated_dir (str): Path to generated audio directory
        total_time (float): Total execution time in seconds
    """
    print("\n" + "="*60)
    print("WAVEGAN PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Data directory: {data_dir}")
    print(f"Trained model: {model_dir}")
    print(f"Generated audio: {generated_dir}")
    print(
        f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\nWhat's next?")
    print("1. Listen to the generated audio samples")
    print("2. Try generating more samples with different parameters")
    print("3. Experiment with different datasets")
    print("4. Adjust training parameters for better results")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Complete WaveGAN pipeline: download → train → generate')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='sample',
                        choices=['sample', 'sc09', 'drums', 'piano'],
                        help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Base directory for datasets')

    # Training options
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--model_size', type=int, default=64,
                        help='Model size parameter')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')

    # Generation options
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of audio samples to generate')
    parser.add_argument('--generate_long', action='store_true',
                        help='Also generate long music pieces')
    parser.add_argument('--long_duration', type=float, default=15.0,
                        help='Duration for long music pieces')

    # Output options
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save generated audio')

    # Control options
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip dataset download step')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training step')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip generation step')
    parser.add_argument('--use_existing_model', type=str, default=None,
                        help='Path to existing model directory to use for generation')

    args = parser.parse_args()

    # Setup paths
    dataset_map = {
        'sample': 'sample',
        'sc09': 'speech_commands',
        'drums': 'music/drums',
        'piano': 'music/piano'
    }

    dataset_path = os.path.join(args.data_dir, dataset_map[args.dataset])
    processed_path = os.path.join(dataset_path, 'processed')

    start_time = datetime.now()

    try:
        # Check dependencies
        if not check_dependencies():
            logger.error(
                "Dependency check failed. Please install missing packages.")
            sys.exit(1)

        # Create directories
        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        # Step 1: Download dataset
        if not args.skip_download:
            processed_data_dir = download_dataset(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                sample_rate=None
            )
        else:
            processed_data_dir = processed_path
            if not os.path.exists(processed_data_dir):
                logger.error(f"Data directory not found: {processed_data_dir}")
                logger.error(
                    "Either run without --skip_download or ensure data exists")
                sys.exit(1)

        # Step 2: Train model
        if not args.skip_training and not args.use_existing_model:
            model_dir = train_model(
                audio_dir=processed_data_dir,
                output_dir=args.models_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_size=args.model_size
            )
        elif args.use_existing_model:
            model_dir = args.use_existing_model
            if not os.path.exists(model_dir):
                logger.error(
                    f"Existing model directory not found: {model_dir}")
                sys.exit(1)
        else:
            # Find most recent model
            model_dirs = [d for d in os.listdir(args.models_dir) if os.path.isdir(
                os.path.join(args.models_dir, d))]
            if not model_dirs:
                logger.error(f"No models found in {args.models_dir}")
                logger.error(
                    "Either run without --skip_training or train a model first")
                sys.exit(1)
            model_dir = os.path.join(args.models_dir, sorted(model_dirs)[-1])

        # Step 3: Generate music
        if not args.skip_generation:
            generated_dir = generate_music(
                model_dir=model_dir,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                sample_rate=None
            )
        else:
            generated_dir = "Skipped"

        # Calculate total time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Print summary
        print_summary(
            dataset_name=args.dataset,
            data_dir=processed_data_dir,
            model_dir=model_dir,
            generated_dir=generated_dir,
            total_time=total_time
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error("Check the logs above for detailed error information")
        sys.exit(1)


if __name__ == '__main__':
    main()
