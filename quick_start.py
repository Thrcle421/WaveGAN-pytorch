#!/usr/bin/env python3
"""
WaveGAN Quick Start Script

This script provides an easy way to get started with WaveGAN.
Choose from predefined configurations or run the complete pipeline.
"""

import os
import sys
import subprocess
import argparse


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🎵 WaveGAN - Audio Generation with Deep Learning")
    print("=" * 60)
    print()


def print_options():
    """Print available options"""
    print("Available Quick Start Options:")
    print()
    print("1. 🚀 Quick Test (5 minutes)")
    print("   - Sample dataset, 10 epochs, 5 samples")
    print("   - Perfect for testing the setup")
    print()
    print("2. 🎯 Standard Training (30 minutes)")
    print("   - Sample dataset, 50 epochs, 10 samples")
    print("   - Good balance of quality and time")
    print()
    print("3. 🏆 High Quality (2+ hours)")
    print("   - Speech Commands dataset, 100 epochs")
    print("   - Best quality results")
    print()
    print("4. 🎼 Long Music Generation")
    print("   - Generate 15-30 second music pieces")
    print("   - Requires existing trained model")
    print()
    print("5. 🛠️  Custom Configuration")
    print("   - Specify your own parameters")
    print()


def run_quick_test():
    """Run quick test configuration"""
    print("🚀 Running Quick Test...")
    cmd = [
        sys.executable, "scripts/run_complete_pipeline.py",
        "--dataset", "sample",
        "--epochs", "10",
        "--batch_size", "16",
        "--num_samples", "5"
    ]
    subprocess.run(cmd)


def run_standard_training():
    """Run standard training configuration"""
    print("🎯 Running Standard Training...")
    cmd = [
        sys.executable, "scripts/run_complete_pipeline.py",
        "--dataset", "sample",
        "--epochs", "50",
        "--batch_size", "32",
        "--num_samples", "10"
    ]
    subprocess.run(cmd)


def run_high_quality():
    """Run high quality training configuration"""
    print("🏆 Running High Quality Training...")
    print("This will download ~2GB dataset and train for 100+ epochs...")
    confirm = input("Continue? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Cancelled.")
        return

    cmd = [
        sys.executable, "scripts/run_complete_pipeline.py",
        "--dataset", "sc09",
        "--epochs", "100",
        "--batch_size", "32",
        "--num_samples", "15"
    ]
    subprocess.run(cmd)


def run_long_music():
    """Generate long music pieces"""
    print("🎼 Generating Long Music...")

    # Check for existing models
    models_dir = "./models"
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        print("❌ No trained models found!")
        print("Please train a model first using options 1, 2, or 3.")
        return

    # Find latest model
    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]
    if not model_dirs:
        print("❌ No valid model directories found!")
        return

    latest_model = os.path.join(models_dir, sorted(model_dirs)[-1])
    print(f"Using model: {latest_model}")

    duration = input("Duration in seconds (default 15): ").strip()
    if not duration:
        duration = "15"

    num_pieces = input("Number of pieces (default 3): ").strip()
    if not num_pieces:
        num_pieces = "3"

    cmd = [
        sys.executable, "scripts/generate_longer_music.py",
        latest_model,
        "--duration", duration,
        "--num_pieces", num_pieces,
        "--add_reverb"
    ]
    subprocess.run(cmd)


def run_custom():
    """Run with custom configuration"""
    print("🛠️  Custom Configuration")
    print()

    # Get parameters
    dataset = input(
        "Dataset (sample/sc09/drums/piano) [sample]: ").strip() or "sample"
    epochs = input("Number of epochs [20]: ").strip() or "20"
    batch_size = input("Batch size [32]: ").strip() or "32"
    num_samples = input("Number of samples to generate [10]: ").strip() or "10"

    cmd = [
        sys.executable, "scripts/run_complete_pipeline.py",
        "--dataset", dataset,
        "--epochs", epochs,
        "--batch_size", batch_size,
        "--num_samples", num_samples
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='WaveGAN Quick Start')
    parser.add_argument('--option', type=int, choices=[1, 2, 3, 4, 5],
                        help='Quick start option (1-5)')
    args = parser.parse_args()

    print_banner()

    if args.option:
        option = args.option
    else:
        print_options()
        try:
            option = int(input("Choose an option (1-5): ").strip())
        except (ValueError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

    print()

    if option == 1:
        run_quick_test()
    elif option == 2:
        run_standard_training()
    elif option == 3:
        run_high_quality()
    elif option == 4:
        run_long_music()
    elif option == 5:
        run_custom()
    else:
        print("❌ Invalid option. Please choose 1-5.")
        return

    print()
    print("✅ Done! Check the output/ directory for generated audio.")
    print("📁 Models are saved in the models/ directory.")
    print("📖 See README.md for more detailed usage instructions.")


if __name__ == '__main__':
    main()
