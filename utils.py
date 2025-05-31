import os
import librosa
import soundfile as sf
import numpy as np


def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    """
    Saves generated samples to disk

    Args:
        epoch_samples (np.array): Generated audio samples
        epoch (int): Current epoch number
        output_dir (str): Directory to save samples
        fs (int): Sample rate
    """
    sample_dir = os.path.join(output_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for idx, sample in enumerate(epoch_samples):
        sample_path = os.path.join(sample_dir, f'sample_{epoch}_{idx}.wav')

        # Handle different array shapes and ensure proper format
        if len(sample.shape) > 1:
            # If it's multi-dimensional, take the first channel or flatten
            if sample.shape[0] == 1:
                sample = sample[0]  # Remove channel dimension
            else:
                sample = sample.flatten()  # Flatten if necessary

        # Ensure it's 1D
        sample = np.squeeze(sample)

        # Normalize to prevent clipping
        max_amp = np.max(np.abs(sample))
        if max_amp > 0:
            sample = sample / max_amp * 0.95

        # Use soundfile for reliable writing
        try:
            sf.write(sample_path, sample.astype(np.float32), fs)
        except Exception as e:
            print(f"Warning: Failed to save {sample_path}: {e}")
            # Fallback to write_audio function
            write_audio(sample_path, sample.astype(np.float32), fs)


def write_audio(filepath, audio, sample_rate):
    """
    Write audio to file using the most compatible method available

    Args:
        filepath (str): Output file path
        audio (np.array): Audio data
        sample_rate (int): Sample rate
    """
    # Ensure audio is 1D and float32
    audio = np.squeeze(audio).astype(np.float32)

    try:
        # Try soundfile first (most reliable)
        sf.write(filepath, audio, sample_rate)
    except ImportError:
        try:
            # Fallback to librosa if available
            if hasattr(librosa, 'output') and hasattr(librosa.output, 'write_wav'):
                librosa.output.write_wav(filepath, audio, sample_rate)
            else:
                # Use newer librosa + soundfile combination
                sf.write(filepath, audio, sample_rate)
        except (ImportError, AttributeError):
            # Last resort: use scipy
            from scipy.io import wavfile
            # Convert to 16-bit integer
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(filepath, sample_rate, audio_int)
