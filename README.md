# WaveGAN - Audio Generation with Deep Learning

WaveGAN is a deep learning project for audio generation based on Generative Adversarial Network (GAN) architecture, capable of learning and generating various types of audio data.

## Project Overview

This project uses the WaveGAN architecture to generate piano audio. WaveGAN learns to create realistic audio samples through adversarial training, useful for music generation and sound synthesis applications.

## Features

- Generate piano audio samples, including single notes, chords, and melodies
- Evaluate generation quality using objective metrics
- Compare with baseline methods
- Visualize results with analysis tools

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Piano Dataset

Generate a synthetic piano audio dataset for training:

```bash
python scripts/download_piano_dataset.py --output_dir ./datasets/piano --num_samples 500
```

### 2. Train Model

Train the WaveGAN model using the piano dataset:

```bash
python scripts/train_and_evaluate_piano.py --output_dir ./output/piano_experiment --num_samples 500 --num_epochs 50
```

### 3. Generate Samples from Trained Model

Generate piano audio samples from a trained model:

```bash
python scripts/generate_piano_samples.py --model_dir ./output/piano_experiment/run_[timestamp]/models/[timestamp] --output_dir ./generated_samples --num_samples 20
```

### 4. Evaluate Model

Evaluate model performance using objective metrics:

```bash
python scripts/evaluate_piano_model.py --model_dir ./output/piano_experiment/run_[timestamp]/models/[timestamp] --data_dir ./datasets/piano/processed
```

## Objective Evaluation Metrics

We use the following objective metrics to evaluate the quality of generated samples:

### Frechet Audio Distance (FAD)

FAD measures the distance between generated and real samples in audio feature space. Lower FAD indicates generated samples have a distribution more similar to real samples.

### Inception Score (IS)

Inception Score measures the quality and diversity of generated samples. Higher IS indicates samples are both high quality and diverse.

### Diversity Score

Diversity Score measures the degree of difference between generated samples. Higher diversity scores indicate more varied samples rather than simply copying training data.

## Baseline Comparison

We compare WaveGAN-generated samples with the following simple baseline methods:

- White Noise
- Pink Noise
- Simple Sine Waves
- Random Sine Wave Combinations
- Random Selection from Training Data

## Project Structure

```
wavegan/
├── datasets/              # Dataset directory
├── scripts/               # Script files
│   ├── download_piano_dataset.py     # Generate piano dataset
│   ├── train_and_evaluate_piano.py   # Train and evaluate model
│   ├── generate_piano_samples.py     # Generate audio samples
│   ├── evaluate_piano_model.py       # Evaluate model
│   ├── evaluate_model.py             # Calculate objective metrics
│   ├── compare_baselines.py          # Compare with baseline methods
│   └── run_evaluation.py             # Run evaluation pipeline
├── models/                # Save trained models
├── output/                # Output directory
├── evaluation/            # Evaluation results
└── README.md              # Project documentation
```

## References

- [WaveGAN: Synthesizing Audio with Generative Adversarial Networks](https://arxiv.org/abs/1802.04208)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
