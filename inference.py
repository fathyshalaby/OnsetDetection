import librosa as lb
import numpy as np

def inference(audio_file,model):
    audio, sr = lb.load(audio_file, sr=44100)
    # calculate the mel spectrogram
    mel_spectrogram = lb.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    # convert to log scale (dB). We'll use the peak power as reference.
    log_mel_spectrogram = lb.power_to_db(mel_spectrogram, ref=np.max)
    # reshape to input shape
    log_mel_spectrogram = log_mel_spectrogram.reshape(1, log_mel_spectrogram.shape[0], log_mel_spectrogram.shape[1], 1)
    # get the predicted onset times
    y_pred = model.predict(log_mel_spectrogram)
    y_pred_thresh = (y_pred > 0.5).astype(int)
    # convert to seconds
    estimated_onsets = librosa.frames_to_time(np.argwhere(y_pred_thresh == 1)[:, 1], sr=sr, hop_length=512)
    return estimated_onsets

