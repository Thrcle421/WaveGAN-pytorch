#!/usr/bin/env python3
"""
WaveGAN Objective Evaluation Script

This script runs the objective evaluation pipeline:
1. Calculate objective metrics on a trained model
2. Compare with baseline methods
3. Generate evaluation report

Usage:
    python scripts/run_evaluation.py --model_dir ./models/[timestamp] --data_dir ./datasets/sample/processed
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


def check_dependencies():
    """
    Check if required dependencies are installed

    Returns:
        bool: True if all dependencies are available
    """
    logger.info("Checking dependencies...")

    required_packages = ['torch', 'numpy', 'librosa', 'matplotlib',
                         'scipy', 'sklearn', 'soundfile', 'seaborn']
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


def run_objective_evaluation(model_dir, data_dir, output_dir, max_samples=100, sample_rate=16000, generated_dir=None):
    """
    Run objective evaluation on the model

    Args:
        model_dir (str): Directory containing trained model
        data_dir (str): Directory containing real audio samples
        output_dir (str): Directory to save evaluation results
        max_samples (int): Maximum number of samples to evaluate
        sample_rate (int): Audio sample rate
        generated_dir (str, optional): Directory containing pre-generated samples

    Returns:
        str: Path to evaluation directory
    """
    logger.info("Starting objective evaluation...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build command based on whether we have pre-generated samples
    cmd = [
        sys.executable, 'scripts/evaluate_model.py',
        '--model_dir', model_dir,
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--max_samples', str(max_samples),
        '--sample_rate', str(sample_rate)
    ]

    # Add generated samples directory if provided
    if generated_dir:
        cmd.extend(['--generated_dir', generated_dir])
        logger.info(f"Using pre-generated samples from: {generated_dir}")

    run_command(cmd, "Objective evaluation")

    # Find the most recent evaluation directory
    eval_dirs = [d for d in os.listdir(
        output_dir) if d.startswith('evaluation_')]
    if not eval_dirs:
        raise FileNotFoundError(
            f"No evaluation directories found in {output_dir}")

    latest_eval_dir = os.path.join(output_dir, sorted(eval_dirs)[-1])

    logger.info(
        f"Objective evaluation completed. Results saved to: {latest_eval_dir}")
    return latest_eval_dir


def run_baseline_comparison(model_dir, data_dir, output_dir, methods=None, num_samples=20, sample_rate=16000, generated_dir=None):
    """
    Compare model with baseline methods

    Args:
        model_dir (str): Directory containing trained model
        data_dir (str): Directory containing real audio samples
        output_dir (str): Directory to save comparison results
        methods (list): List of baseline methods to compare
        num_samples (int): Number of samples to generate
        sample_rate (int): Audio sample rate
        generated_dir (str, optional): Directory containing pre-generated model samples

    Returns:
        str: Path to baseline comparison directory
    """
    logger.info("Starting baseline comparison...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Default methods if not specified
    if methods is None:
        methods = ['white_noise', 'pink_noise',
                   'sine_wave', 'random_sines', 'random_selection']

    # Build command
    cmd = [
        sys.executable, 'scripts/compare_baselines.py',
        '--model_dir', model_dir,
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--num_samples', str(num_samples),
        '--sample_rate', str(sample_rate),
        '--methods'
    ] + methods

    # Add generated samples directory if provided
    if generated_dir:
        cmd.extend(['--generated_dir', generated_dir])
        logger.info(f"Using pre-generated samples from: {generated_dir}")

    run_command(cmd, "Baseline comparison")

    logger.info(
        f"Baseline comparison completed. Results saved to: {output_dir}")
    return output_dir


def generate_evaluation_report(objective_dir, baseline_dir, output_file):
    """
    Generate comprehensive evaluation report

    Args:
        objective_dir (str): Directory with objective evaluation results
        baseline_dir (str): Directory with baseline comparison results
        output_file (str): Path to output report file

    Returns:
        str: Path to report file
    """
    logger.info("Generating evaluation report...")

    # Load metrics from objective evaluation
    metrics_file = os.path.join(objective_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
        logger.warning(f"Metrics file not found: {metrics_file}")

    # Load baseline comparison report
    baseline_report_file = os.path.join(
        baseline_dir, 'baseline_comparison_report.md')
    if os.path.exists(baseline_report_file):
        with open(baseline_report_file, 'r') as f:
            baseline_report = f.read()
    else:
        baseline_report = "Baseline comparison report not found."
        logger.warning(
            f"Baseline report file not found: {baseline_report_file}")

    # Create report
    with open(output_file, 'w') as f:
        f.write("# WaveGAN Objective Evaluation Report\n\n")
        f.write(
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Objective Metrics Summary\n\n")

        if metrics:
            f.write("### Key Metrics\n\n")
            f.write(
                f"- **Frechet Audio Distance (FAD)**: {metrics.get('fad', 'N/A'):.4f} (lower is better)\n")
            f.write(
                f"- **Inception Score**: {metrics.get('inception_score', 'N/A'):.4f} ± {metrics.get('inception_std', 'N/A'):.4f} (higher is better)\n")
            f.write(
                f"- **Diversity Score**: {metrics.get('diversity', 'N/A'):.4f} (higher is better)\n\n")

            # Interpret metrics
            fad = metrics.get('fad', float('inf'))
            is_score = metrics.get('inception_score', 0)
            diversity = metrics.get('diversity', 0)

            f.write("### Interpretation\n\n")

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
                f"- **FAD Score** indicates {fad_quality} quality compared to real samples.\n")

            # Inception Score interpretation
            if is_score > 3.0:
                is_quality = "excellent"
            elif is_score > 2.0:
                is_quality = "good"
            elif is_score > 1.5:
                is_quality = "fair"
            else:
                is_quality = "poor"

            f.write(
                f"- **Inception Score** indicates {is_quality} diversity and quality.\n")

            # Diversity interpretation
            if diversity > 0.7:
                div_quality = "very high"
            elif diversity > 0.5:
                div_quality = "high"
            elif diversity > 0.3:
                div_quality = "moderate"
            else:
                div_quality = "low"

            f.write(f"- **Sample Diversity** is {div_quality}.\n\n")
        else:
            f.write("No metrics data available.\n\n")

        f.write("## 2. Baseline Comparison\n\n")

        # Include summary from baseline comparison
        if "WaveGAN Baseline Comparison Report" in baseline_report:
            # Extract summary section
            import re
            analysis_section = re.search(
                r"## Analysis(.*?)##", baseline_report, re.DOTALL)
            if analysis_section:
                f.write(analysis_section.group(1))
            else:
                f.write("See detailed baseline comparison report for analysis.\n\n")
        else:
            f.write(baseline_report + "\n\n")

        f.write("## 3. Recommended Next Steps\n\n")

        # Make recommendations based on metrics
        if metrics:
            if fad > 15 or is_score < 1.5:
                f.write(
                    "Based on the objective metrics, the model quality could be significantly improved:\n\n")
                f.write(
                    "- **Increase training time**: Train for more epochs (100+)\n")
                f.write(
                    "- **Use a larger dataset**: Current results suggest underfitting or limited data variety\n")
                f.write(
                    "- **Adjust model architecture**: Consider increasing model capacity\n")
            elif fad > 8 or is_score < 2.0:
                f.write(
                    "The model shows promising results but could benefit from refinement:\n\n")
                f.write(
                    "- **Fine-tune hyperparameters**: Experiment with learning rate and batch size\n")
                f.write(
                    "- **Add data augmentation**: Increase effective dataset size\n")
                f.write(
                    "- **Improve post-processing**: Enhance audio quality with effects like reverb\n")
            else:
                f.write(
                    "The model is performing well according to objective metrics:\n\n")
                f.write(
                    "- **Experiment with conditioning**: Try conditional generation for more control\n")
                f.write(
                    "- **Extend generation length**: Implement techniques for longer coherent samples\n")
        else:
            f.write(
                "Complete objective evaluation to get specific recommendations.\n\n")

        f.write("## 4. Conclusion\n\n")
        f.write("This evaluation provides a comprehensive assessment of the WaveGAN model's performance using objective metrics. These quantitative measures offer insights into the technical performance of the model in terms of audio quality, diversity, and similarity to real samples.\n\n")

        f.write("For full details, refer to the individual evaluation directories:\n\n")
        f.write(f"- Objective metrics: `{objective_dir}`\n")
        f.write(f"- Baseline comparisons: `{baseline_dir}`\n")

    logger.info(f"Evaluation report generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Run objective WaveGAN evaluation pipeline')

    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing real audio samples')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Base directory for evaluation outputs')

    # Evaluation options
    parser.add_argument('--objective_samples', type=int, default=100,
                        help='Number of samples for objective evaluation')
    parser.add_argument('--baseline_samples', type=int, default=20,
                        help='Number of samples for baseline comparison')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')

    # Control options
    parser.add_argument('--skip_objective', action='store_true',
                        help='Skip objective evaluation')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip baseline comparison')
    parser.add_argument('--baseline_methods', type=str, nargs='+',
                        default=['white_noise', 'pink_noise', 'sine_wave',
                                 'random_sines', 'random_selection'],
                        help='Baseline methods to compare')
    parser.add_argument('--generated_dir', type=str, default=None,
                        help='Directory containing pre-generated samples (skips generation step)')
    parser.add_argument('--skip_subjective', action='store_true',
                        help='Skip subjective evaluation (always skipped in this version)')

    args = parser.parse_args()

    # Create timestamp for this evaluation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(
        args.output_dir, f"objective_evaluation_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    try:
        # Check dependencies
        if not check_dependencies():
            logger.error(
                "Dependency check failed. Please install missing packages.")
            sys.exit(1)

        # Create subdirectories
        objective_dir = os.path.join(eval_dir, 'objective')
        baseline_dir = os.path.join(eval_dir, 'baselines')

        # Step 1: Run objective evaluation
        if not args.skip_objective:
            objective_results_dir = run_objective_evaluation(
                model_dir=args.model_dir,
                data_dir=args.data_dir,
                output_dir=objective_dir,
                max_samples=args.objective_samples,
                sample_rate=args.sample_rate,
                generated_dir=args.generated_dir
            )
        else:
            objective_results_dir = objective_dir
            logger.info("Skipping objective evaluation")

        # Step 2: Run baseline comparison
        if not args.skip_baselines:
            baseline_results_dir = run_baseline_comparison(
                model_dir=args.model_dir,
                data_dir=args.data_dir,
                output_dir=baseline_dir,
                methods=args.baseline_methods,
                num_samples=args.baseline_samples,
                sample_rate=args.sample_rate,
                generated_dir=args.generated_dir
            )
        else:
            baseline_results_dir = baseline_dir
            logger.info("Skipping baseline comparison")

        # Step 3: Generate comprehensive report
        report_file = os.path.join(eval_dir, 'evaluation_report.md')
        generate_evaluation_report(
            objective_dir=objective_results_dir,
            baseline_dir=baseline_results_dir,
            output_file=report_file
        )

        # Create README for the evaluation directory
        with open(os.path.join(eval_dir, 'README.md'), 'w') as f:
            f.write("# WaveGAN Objective Evaluation Results\n\n")
            f.write(
                f"Evaluation conducted on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model: {args.model_dir}\n")
            f.write(f"Data: {args.data_dir}\n\n")

            f.write("## Contents\n\n")
            f.write(
                "1. **Objective Metrics**: Statistical evaluation of model performance\n")
            f.write(
                "2. **Baseline Comparisons**: Comparison with simple generation methods\n")
            f.write(
                "3. **Evaluation Report**: Comprehensive analysis and recommendations\n\n")

            f.write("## Quick Links\n\n")
            f.write(f"- [Evaluation Report](./evaluation_report.md)\n")
            if not args.skip_objective:
                f.write(f"- [Objective Metrics](./objective)\n")
            if not args.skip_baselines:
                f.write(
                    f"- [Baseline Comparisons](./baselines/baseline_comparison_report.md)\n")

        logger.info(f"Evaluation complete! Results available in: {eval_dir}")
        logger.info(f"Main report: {report_file}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
