sample_rate = 32000
#sr = sample_rate
n_fft = 1024
hop_length = 320
win_length = 1024
window = 'hann'
center = True
pad_mode = 'reflect'
is_mono = True

# Mel parameters (the same as librosa.feature.melspectrogram)
n_mels = 64
fmin = 20
fmax = 14000

# Power to db parameters (the same as default settings of librosa.power_to_db
ref = 1.0
amin = 1e-10
top_db = None