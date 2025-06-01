#!/usr/bin/env python3
"""
Piano WaveGAN Model Evaluation Script

This script evaluates a trained WaveGAN model on piano audio generation:
1. Generates samples from the model
2. Runs objective evaluation metrics
3. Compares with baseline methods
4. Produces evaluation report

Usage:
    python scripts/evaluate_piano_model.py --model_dir ./models/[timestamp] --data_dir ./datasets/piano/processed
"""

import os
import sys
import argparse
import logging
import subprocess
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


def generate_samples(model_dir, output_dir, num_samples, sample_rate):
    """
    Generate audio samples from trained model

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
    os.makedirs(output_dir, exist_ok=True)

    # Run our custom generation script
    cmd = [
        sys.executable, 'scripts/generate_piano_samples.py',
        '--model_dir', model_dir,
        '--output_dir', output_dir,
        '--num_samples', str(num_samples),
        '--sample_rate', str(sample_rate)
    ]

    run_command(cmd, "Generate audio samples")

    logger.info(f"Samples generated and saved to {output_dir}")
    return output_dir


def run_objective_evaluation(model_dir, data_dir, generated_dir, output_dir, num_baseline_samples, sample_rate):
    """
    Run objective evaluation comparing real data, generated samples, and baselines

    Args:
        model_dir (str): Directory containing trained model
        data_dir (str): Directory containing real audio samples
        generated_dir (str): Directory containing generated audio samples
        output_dir (str): Directory to save evaluation results
        num_baseline_samples (int): Number of baseline samples to generate
        sample_rate (int): Audio sample rate

    Returns:
        str: Path to evaluation directory
    """
    logger.info("Running objective evaluation...")

    # Create evaluation directory
    os.makedirs(output_dir, exist_ok=True)

    # Run objective evaluation
    cmd = [
        sys.executable, 'scripts/run_evaluation.py',
        '--model_dir', model_dir,
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--baseline_samples', str(num_baseline_samples),
        '--sample_rate', str(sample_rate),
        '--generated_dir', generated_dir,
        '--skip_subjective'  # Skip subjective evaluation
    ]

    run_command(cmd, "Run objective evaluation")

    # Find the most recent evaluation directory
    eval_dirs = [d for d in os.listdir(
        output_dir) if d.startswith('objective_evaluation_')]
    if not eval_dirs:
        raise FileNotFoundError(
            f"No evaluation directories found in {output_dir}")

    latest_eval_dir = os.path.join(output_dir, sorted(eval_dirs)[-1])
    logger.info(f"Evaluation completed and saved to {latest_eval_dir}")

    return latest_eval_dir


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained WaveGAN piano model')

    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing real audio samples for comparison')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation/piano',
                        help='Directory to save evaluation results')

    # Evaluation options
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--num_baseline_samples', type=int, default=20,
                        help='Number of baseline samples to generate')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')

    args = parser.parse_args()

    try:
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)

        # Generate samples
        samples_dir = os.path.join(eval_dir, 'generated_samples')
        generate_samples(
            model_dir=args.model_dir,
            output_dir=samples_dir,
            num_samples=args.num_samples,
            sample_rate=args.sample_rate
        )

        # Run objective evaluation
        evaluation_dir = os.path.join(eval_dir, 'evaluation')
        results_dir = run_objective_evaluation(
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            generated_dir=samples_dir,
            output_dir=evaluation_dir,
            num_baseline_samples=args.num_baseline_samples,
            sample_rate=args.sample_rate
        )

        # Create summary report
        summary_path = os.path.join(eval_dir, "evaluation_summary.md")
        with open(summary_path, 'w') as f:
            f.write("# WaveGAN Piano Model Evaluation Summary\n\n")
            f.write(
                f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Evaluation Overview\n\n")
            f.write(f"- Model: {args.model_dir}\n")
            f.write(f"- Real data: {args.data_dir}\n")
            f.write(f"- Generated samples: {samples_dir}\n")
            f.write(f"- Evaluation results: {results_dir}\n\n")

            f.write("## Generated Samples\n\n")
            f.write(f"- Number of samples: {args.num_samples}\n")
            f.write(f"- Sample rate: {args.sample_rate}Hz\n\n")

            f.write("## Full Evaluation Report\n\n")
            f.write(
                "For detailed metrics and analysis, see the full evaluation report:\n\n")

            report_path = os.path.join(results_dir, "evaluation_report.md")
            if os.path.exists(report_path):
                rel_path = os.path.relpath(report_path, eval_dir)
                f.write(f"[Evaluation Report]({rel_path})\n\n")
            else:
                f.write(
                    "Evaluation report not found. Check the evaluation directory for results.\n\n")

        logger.info(f"Evaluation completed! Results saved to: {eval_dir}")
        logger.info(f"Summary report: {summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
