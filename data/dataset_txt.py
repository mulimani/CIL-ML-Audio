from torch.utils.data import Dataset
import torch
import torchaudio
from lib.utils import config


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if x.size(1) <= audio_length:
        return torch.cat((x, torch.zeros(1, audio_length - x.size(1))), dim=1)
    else:
        return x[:, 0: audio_length]


class BatchData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)


class BatchDataBin(Dataset):
    def __init__(self, images, labels, labels_bin):
        self.images = images
        self.labels = labels
        self.labels_bin = labels_bin

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        label_bin = torch.tensor(self.labels_bin[index])
        return image, label, label_bin

    def __len__(self):
        return len(self.images)


class MelData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        # Spectrogram parameters (the same as librosa.stft)
        self.sample_rate = config.sample_rate
        #sr = sample_rate
        n_fft = config.n_fft
        hop_length = config.hop_length
        win_length = config.win_length

        # Mel parameters (the same as librosa.feature.melspectrogram)
        n_mels = config.n_mels
        fmin = config.fmin
        fmax = config.fmax

        # Power to db parameters (the same as default settings of librosa.power_to_db
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length,
                                                            n_mels=n_mels,
                                                            f_min=fmin, f_max=fmax)

    def __getitem__(self, index):
        audio_file = self.images[index]
        label = self.labels[index]
        y, sr = torchaudio.load(audio_file)
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)
        y = pad_or_truncate(y, self.sample_rate * 10)
        mbe = self.melspec(y)
        mbe = torch.transpose(mbe, 2, 1)
        mbe = torch.log(mbe + torch.finfo(torch.float32).eps)
        return mbe, label

    def __len__(self):
        return len(self.images)
