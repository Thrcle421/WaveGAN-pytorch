# WaveGAN - Audio Generation with Deep Learning

A PyTorch implementation of WaveGAN [(Donahue, et al. 2018)](https://arxiv.org/abs/1802.04208) for generating realistic audio using Generative Adversarial Networks.

## 🎵 Project Overview

WaveGAN is a deep learning model that generates high-quality audio by learning patterns from training data. This implementation provides a complete end-to-end pipeline for audio generation.

### ✨ Key Features

- **Complete Pipeline**: Download datasets → Train models → Generate audio
- **Multiple Datasets**: Support for speech, music, and custom audio datasets
- **Flexible Generation**: Generate both short samples and long music pieces
- **GPU Acceleration**: CUDA support for faster training and generation
- **Audio Effects**: Built-in reverb and audio processing
- **Easy to Use**: Simple command-line interface

## 📁 Project Structure

```
wavegan/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── Core Model Files/
│   ├── wavegan.py              # WaveGAN model architecture
│   ├── wgan.py                 # WGAN training logic
│   ├── train_wavegan.py        # Main training script
│   ├── sample.py               # Data loading utilities
│   ├── utils.py                # Helper functions
│   └── log.py                  # Logging utilities
│
├── scripts/                    # Utility Scripts
│   ├── download_dataset.py     # Dataset download and preparation
│   ├── generate_music.py       # Generate short audio samples
│   ├── generate_longer_music.py # Generate long music pieces
│   └── run_complete_pipeline.py # Complete end-to-end pipeline
│
├── datasets/                   # Training Data
│   ├── sample/                 # Sample synthetic dataset
│   │   └── processed/          # Processed audio files
│   ├── speech_commands/        # Speech commands dataset
│   └── music/                  # Music datasets
│
├── models/                     # Trained Models
│   └── [timestamp]/            # Model checkpoints and configs
│       ├── model_gen.pkl       # Generator weights
│       ├── model_dis.pkl       # Discriminator weights
│       ├── config.json         # Training configuration
│       └── history.pkl         # Training history
│
├── output/                     # Generated Audio
│   ├── short/                  # Short audio samples (1-2 seconds)
│   └── long/                   # Long music pieces (10+ seconds)
│
├── notebooks/                  # Jupyter Notebooks
├── jobs/                       # SLURM job scripts
└── docs/                       # Additional documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd wavegan

# Install dependencies
pip install -r requirements.txt
```

### 2. Super Quick Start (Recommended for Beginners)

Use the interactive quick start script:

```bash
# Interactive mode - choose from predefined options
python quick_start.py

# Or run directly with option number
python quick_start.py --option 1  # Quick test (5 minutes)
python quick_start.py --option 2  # Standard training (30 minutes)
python quick_start.py --option 3  # High quality (2+ hours)
```

### 3. Complete Pipeline (Advanced Users)

Run the entire pipeline with one command:

```bash
# Quick test with sample data (5 minutes)
python scripts/run_complete_pipeline.py --dataset sample --epochs 10 --num_samples 5

# Better quality with more training (30 minutes)
python scripts/run_complete_pipeline.py --dataset sample --epochs 50 --batch_size 32 --num_samples 10
```

### 4. Step-by-Step Usage

#### Step 1: Download Dataset

```bash
# Download sample dataset (fast, for testing)
python scripts/download_dataset.py --dataset sample --output_dir ./datasets/sample

# Download speech commands dataset (better quality, larger)
python scripts/download_dataset.py --dataset sc09 --output_dir ./datasets/speech_commands
```

#### Step 2: Train Model

```bash
# Basic training
python train_wavegan.py ./datasets/sample/processed ./models --num-epochs 20

# Advanced training with custom parameters
python train_wavegan.py ./datasets/sample/processed ./models \
    --num-epochs 100 \
    --batch-size 64 \
    --model-size 128 \
    --learning-rate 1e-4
```

#### Step 3: Generate Audio

```bash
# Generate short samples (1 second each)
python scripts/generate_music.py ./models/[model_timestamp] \
    --num_samples 10 \
    --output_dir ./output/short

# Generate long music pieces (15 seconds each)
python scripts/generate_longer_music.py ./models/[model_timestamp] \
    --duration 15 \
    --num_pieces 3 \
    --add_reverb \
    --output_dir ./output/long
```

## 📊 Available Datasets

| Dataset  | Description          | Size   | Duration   | Use Case          |
| -------- | -------------------- | ------ | ---------- | ----------------- |
| `sample` | Synthetic sine waves | ~3MB   | 1s each    | Quick testing     |
| `sc09`   | Speech commands      | ~2GB   | 1s each    | Speech generation |
| `drums`  | Drum samples         | ~500MB | 1-5s each  | Rhythm generation |
| `piano`  | Piano notes          | ~1GB   | 2-10s each | Melody generation |

## ⚙️ Configuration Options

### Training Parameters

| Parameter         | Default | Description               |
| ----------------- | ------- | ------------------------- |
| `--num-epochs`    | 100     | Number of training epochs |
| `--batch-size`    | 64      | Training batch size       |
| `--model-size`    | 64      | Model capacity parameter  |
| `--learning-rate` | 1e-4    | Learning rate             |
| `--ngpus`         | 1       | Number of GPUs to use     |

### Generation Parameters

| Parameter       | Default | Description                       |
| --------------- | ------- | --------------------------------- |
| `--num_samples` | 10      | Number of samples to generate     |
| `--duration`    | 10.0    | Duration for long music (seconds) |
| `--sample_rate` | 16000   | Audio sample rate                 |
| `--add_reverb`  | False   | Add reverb effect                 |

## 🎯 Usage Examples

### Example 1: Quick Test

```bash
# 5-minute test run
python scripts/run_complete_pipeline.py \
    --dataset sample \
    --epochs 5 \
    --batch_size 16 \
    --num_samples 3
```

### Example 2: High Quality Training

```bash
# Download better dataset
python scripts/download_dataset.py --dataset sc09 --output_dir ./datasets/speech

# Train with more epochs
python train_wavegan.py ./datasets/speech/processed ./models \
    --num-epochs 200 \
    --batch-size 32 \
    --model-size 128

# Generate long music with effects
python scripts/generate_longer_music.py ./models/[timestamp] \
    --duration 30 \
    --num_pieces 5 \
    --add_reverb
```

### Example 3: Custom Dataset

```bash
# Prepare your own audio files in a directory
# Then train directly
python train_wavegan.py /path/to/your/audio/files ./models \
    --num-epochs 100 \
    --batch-size 64
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**

   ```bash
   # Reduce batch size
   python train_wavegan.py ... --batch-size 16
   ```

2. **Poor Audio Quality**

   ```bash
   # Increase training epochs
   python train_wavegan.py ... --num-epochs 200
   ```

3. **Slow Training**

   ```bash
   # Use GPU acceleration
   python train_wavegan.py ... --ngpus 1
   ```

4. **Audio Too Short**
   ```bash
   # Use the long music generator
   python scripts/generate_longer_music.py ... --duration 30
   ```

### System Requirements

- **Python**: 3.7+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ for large datasets)
- **Storage**: 5GB+ free space

## 📈 Training Tips

1. **Start Small**: Use `sample` dataset for initial testing
2. **Increase Gradually**: Move to larger datasets as you gain experience
3. **Monitor Progress**: Check generated samples during training
4. **Experiment**: Try different model sizes and learning rates
5. **Be Patient**: Good results require 50+ epochs

## 🎵 Output Files

After generation, you'll find:

- **Individual Samples**: `generated_XXXX.wav` files
- **Audio Collage**: Combined samples in one file
- **Metadata**: JSON files with generation parameters
- **Analysis**: Statistical information about generated audio

## 📚 Additional Resources

- [Original WaveGAN Paper](https://arxiv.org/abs/1802.04208)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Audio Processing with Librosa](https://librosa.org/)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

## 📄 License

This project is open source. Please check the original WaveGAN paper for academic usage guidelines.
