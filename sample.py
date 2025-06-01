import logging
import librosa
import pescador
import os
import numpy as np


LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def file_sample_generator(filepath, window_length=16384, fs=16000):
    """
    Audio sample generator
    """
    try:
        audio_data, _ = librosa.load(filepath, sr=fs)

        # Clip amplitude
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 1:
            audio_data /= max_amp
    except Exception as e:
        LOGGER.error('Could not load {}: {}'.format(filepath, str(e)))
        raise StopIteration()

    audio_len = len(audio_data)

    # Pad audio to at least a single frame
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
        audio_len = len(audio_data)

    while True:
        if audio_len == window_length:
            # If we only have a single frame's worth of audio, just yield the whole audio
            sample = audio_data
        else:
            # Sample a random window from the audio file
            start_idx = np.random.randint(0, audio_len - window_length)
            end_idx = start_idx + window_length
            sample = audio_data[start_idx:end_idx]

        sample = sample.astype('float32')
        assert not np.any(np.isnan(sample))

        yield {'X': sample}


def create_batch_generator(audio_filepath_list, batch_size):
    streamers = []
    for audio_filepath in audio_filepath_list:
        s = pescador.Streamer(file_sample_generator, audio_filepath)
        streamers.append(s)

    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen


def get_all_audio_filepaths(audio_dir):
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]


def create_data_split(audio_filepath_list, valid_ratio, test_ratio,
                      train_batch_size, valid_size, test_size):
    num_files = len(audio_filepath_list)
    
    # Ensure we have at least 1 file for each split
    min_valid = max(1, int(np.ceil(num_files * valid_ratio)))
    min_test = max(1, int(np.ceil(num_files * test_ratio)))
    
    # Ensure we don't try to use more files than we have
    if min_valid + min_test >= num_files:
        # Adjust to ensure at least 1 training sample
        min_valid = max(1, min(min_valid, int(num_files * 0.2)))
        min_test = max(1, min(min_test, int(num_files * 0.1)))
    
    num_valid = min_valid
    num_test = min_test
    num_train = num_files - num_valid - num_test
    
    # Ensure we have at least 1 sample for each split
    assert num_valid > 0, f"No validation samples (total files: {num_files}, valid_ratio: {valid_ratio})"
    assert num_test > 0, f"No test samples (total files: {num_files}, test_ratio: {test_ratio})"
    assert num_train > 0, f"No training samples (total files: {num_files}, after valid: {num_valid}, test: {num_test})"

    valid_files = audio_filepath_list[:num_valid]
    test_files = audio_filepath_list[num_valid:num_valid + num_test]
    train_files = audio_filepath_list[num_valid + num_test:]

    train_gen = create_batch_generator(train_files, train_batch_size)
    valid_data = next(iter(create_batch_generator(valid_files, valid_size)))
    test_data = next(iter(create_batch_generator(test_files, test_size)))

    return train_gen, valid_data, test_data
