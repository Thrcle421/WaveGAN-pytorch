#!/usr/bin/env python3
"""
WaveGAN Piano Model Training and Evaluation

This script provides an end-to-end pipeline for:
1. Generating a piano dataset
2. Training a WaveGAN model on the dataset
3. Evaluating the model using objective metrics

Usage:
    python scripts/train_and_evaluate_piano.py --output_dir ./output/piano_experiment
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import datetime
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


def generate_piano_dataset(output_dir, num_samples, sample_rate):
    """
    Generate a piano dataset

    Args:
        output_dir (str): Directory to save the dataset
        num_samples (int): Number of samples to generate
        sample_rate (int): Audio sample rate

    Returns:
        str: Path to dataset directory
    """
    logger.info(f"Generating piano dataset with {num_samples} samples...")

    dataset_dir = os.path.join(output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # Run piano dataset generator
    cmd = [
        sys.executable, 'scripts/download_piano_dataset.py',
        '--output_dir', dataset_dir,
        '--num_samples', str(num_samples),
        '--sample_rate', str(sample_rate)
    ]

    run_command(cmd, "Generate piano dataset")

    processed_dir = os.path.join(dataset_dir, 'processed')
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(
            f"Processed dataset directory not found: {processed_dir}")

    logger.info(f"Piano dataset generated at {processed_dir}")
    return processed_dir


def train_wavegan(dataset_dir, output_dir, num_epochs, batch_size, model_size, learning_rate):
    """
    Train WaveGAN model on the dataset

    Args:
        dataset_dir (str): Path to processed dataset directory
        output_dir (str): Directory to save trained model
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_size (int): Model size parameter
        learning_rate (float): Learning rate

    Returns:
        str: Path to trained model directory
    """
    logger.info(f"Training WaveGAN model for {num_epochs} epochs...")

    # Create models directory
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Run training script
    cmd = [
        sys.executable, 'train_wavegan.py',
        dataset_dir, models_dir,
        '--num-epochs', str(num_epochs),
        '--batch-size', str(batch_size),
        '--model-size', str(model_size),
        '--learning-rate', str(learning_rate)
    ]

    run_command(cmd, "Train WaveGAN model")

    # Find the most recent model directory
    model_dirs = [d for d in os.listdir(
        models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {models_dir}")

    latest_model_dir = os.path.join(models_dir, sorted(model_dirs)[-1])

    logger.info(f"Model trained and saved to {latest_model_dir}")
    return latest_model_dir


def generate_samples(model_dir, output_dir, num_samples, sample_rate):
    """
    Generate audio samples from trained model using our custom script

    Args:
        model_dir (str): Directory containing trained model
        output_dir (str): Directory to save generated samples
        num_samples (int): Number of samples to generate
        sample_rate (int): Audio sample rate

    Returns:
        str: Path to generated samples directory
    """
    logger.info(f"Generating {num_samples} audio samples...")

    # Create output directory
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Run our custom generation script instead of generate_music.py
    cmd = [
        sys.executable, 'scripts/generate_piano_samples.py',
        '--model_dir', model_dir,
        '--output_dir', samples_dir,
        '--num_samples', str(num_samples),
        '--sample_rate', str(sample_rate)
    ]

    run_command(cmd, "Generate audio samples")

    logger.info(f"Samples generated and saved to {samples_dir}")
    return samples_dir


def run_objective_evaluation(model_dir, dataset_dir, output_dir, max_samples, sample_rate):
    """
    Run objective evaluation on the model

    Args:
        model_dir (str): Directory containing trained model
        dataset_dir (str): Directory containing real audio samples
        output_dir (str): Directory to save evaluation results
        max_samples (int): Maximum number of samples to evaluate
        sample_rate (int): Audio sample rate

    Returns:
        str: Path to evaluation directory
    """
    logger.info("Starting objective evaluation...")

    # Create output directory
    eval_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    # First generate samples directly into the evaluation directory
    samples_dir = os.path.join(eval_dir, 'generated_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Generate samples for evaluation
    generate_samples(
        model_dir=model_dir,
        output_dir=samples_dir,
        num_samples=max_samples,
        sample_rate=sample_rate
    )

    # Run evaluation script with the generated samples directory
    cmd = [
        sys.executable, 'scripts/run_evaluation.py',
        '--model_dir', model_dir,
        '--data_dir', dataset_dir,
        '--output_dir', eval_dir,
        '--objective_samples', str(max_samples),
        # Use fewer samples for baseline comparison
        '--baseline_samples', str(max_samples // 5),
        '--sample_rate', str(sample_rate),
        '--generated_dir', samples_dir  # Use our pre-generated samples
    ]

    run_command(cmd, "Run objective evaluation")

    # Find the most recent evaluation directory
    eval_dirs = [d for d in os.listdir(
        eval_dir) if d.startswith('objective_evaluation_')]
    if not eval_dirs:
        raise FileNotFoundError(
            f"No evaluation directories found in {eval_dir}")

    latest_eval_dir = os.path.join(eval_dir, sorted(eval_dirs)[-1])

    logger.info(f"Evaluation completed and saved to {latest_eval_dir}")
    return latest_eval_dir


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate WaveGAN model on piano dataset')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./output/piano_experiment',
                        help='Base directory for all outputs')

    # Dataset options
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of piano samples to generate')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')

    # Training options
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--model_size', type=int, default=64,
                        help='WaveGAN model size parameter')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')

    # Evaluation options
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Number of samples for evaluation')

    # Control options
    parser.add_argument('--skip_dataset', action='store_true',
                        help='Skip dataset generation (use existing dataset)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training (use existing model)')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip sample generation')
    parser.add_argument('--existing_dataset', type=str, default=None,
                        help='Path to existing dataset (if skipping generation)')
    parser.add_argument('--existing_model', type=str, default=None,
                        help='Path to existing model (if skipping training)')

    args = parser.parse_args()

    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Step 1: Generate dataset or use existing one
        if args.skip_dataset and args.existing_dataset:
            dataset_dir = args.existing_dataset
            logger.info(f"Using existing dataset: {dataset_dir}")
        else:
            dataset_dir = generate_piano_dataset(
                output_dir=run_dir,
                num_samples=args.num_samples,
                sample_rate=args.sample_rate
            )

        # Step 2: Train model or use existing one
        if args.skip_training and args.existing_model:
            model_dir = args.existing_model
            logger.info(f"Using existing model: {model_dir}")
        else:
            model_dir = train_wavegan(
                dataset_dir=dataset_dir,
                output_dir=run_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                model_size=args.model_size,
                learning_rate=args.learning_rate
            )

        # Step 3: Generate samples (optional)
        if not args.skip_generation:
            samples_dir = generate_samples(
                model_dir=model_dir,
                output_dir=run_dir,
                num_samples=args.eval_samples,
                sample_rate=args.sample_rate
            )

        # Step 4: Run objective evaluation
        eval_dir = run_objective_evaluation(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            output_dir=run_dir,
            max_samples=args.eval_samples,
            sample_rate=args.sample_rate
        )

        # Create summary report
        summary_path = os.path.join(run_dir, "summary.md")
        with open(summary_path, 'w') as f:
            f.write("# WaveGAN Piano Model Training and Evaluation\n\n")
            f.write(
                f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Pipeline Summary\n\n")
            f.write(f"- Dataset: {dataset_dir}\n")
            f.write(f"- Model: {model_dir}\n")
            f.write(f"- Evaluation: {eval_dir}\n\n")

            f.write("## Training Parameters\n\n")
            f.write(f"- Epochs: {args.num_epochs}\n")
            f.write(f"- Batch size: {args.batch_size}\n")
            f.write(f"- Model size: {args.model_size}\n")
            f.write(f"- Learning rate: {args.learning_rate}\n\n")

            f.write("## Evaluation Results\n\n")
            f.write("See the full evaluation report in the evaluation directory.\n")
            f.write(
                f"[Evaluation Report]({os.path.relpath(os.path.join(eval_dir, 'evaluation_report.md'), run_dir)})\n\n")

        logger.info(
            f"Pipeline completed successfully! Results saved to: {run_dir}")
        logger.info(f"Summary report: {summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
