import numpy as np
import librosa

def evaluate_onset_detection(actual_onsets, estimated_onsets, sr, audio, window=0.05):
    """
    Evaluate an onset detection algorithm using the F-measure metric.

    Parameters:
        actual_onsets (array): Array of actual onset times in seconds.
        estimated_onsets (array): Array of estimated onset times in seconds.
        sr (int): Sample rate of the audio signal.
        audio (array): Audio signal as a one-dimensional numpy array.
        window (float): Window size in seconds.

    Returns:
        f_measure (float): F-measure value.
        precision (float): Precision value.
        recall (float): Recall value.
    """

    # Convert onset times to sample indices
    actual_onsets_idx = np.round(actual_onsets * sr).astype(int)
    estimated_onsets_idx = np.round(estimated_onsets * sr).astype(int)

    # Create windowed regions around actual onsets
    actual_onsets_regions = np.zeros((len(actual_onsets_idx), int(window * sr)))
    for i, onset_idx in enumerate(actual_onsets_idx):
        actual_onsets_regions[i, :] = audio[onset_idx:onset_idx+actual_onsets_regions.shape[1]]

    # Find estimated onsets within windowed regions
    tp = 0
    for onset_idx in estimated_onsets_idx:
        if np.any(np.abs(onset_idx - actual_onsets_idx) <= int(window * sr)):
            tp += 1

    # Count false positives and false negatives
    fp = len(estimated_onsets_idx) - tp
    fn = len(actual_onsets_idx) - tp

    # Calculate precision, recall, and F-measure
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * precision * recall / (precision + recall)

    return f_measure, precision, recall


def main():
    # Load audio signal
    audio, sr = librosa.load('audio.wav', sr=44100)

    # Load ground truth onsets
    actual_onsets = np.loadtxt('onsets.txt')

    # Load estimated onsets
    estimated_onsets = np.loadtxt('estimated_onsets.txt')

    # Evaluate onset detection algorithm
    f_measure, precision, recall = evaluate_onset_detection(actual_onsets, estimated_onsets, sr, audio)

    print('F-measure: {:.3f}'.format(f_measure))
    print('Precision: {:.3f}'.format(precision))
    print('Recall: {:.3f}'.format(recall))