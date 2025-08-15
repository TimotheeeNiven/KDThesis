import os
import torch
import torchaudio
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchaudio.functional as F


mel_transform = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)
db_transform = AmplitudeToDB()
resize = T.Resize((32, 32))


class CustomSpeechCommands(Dataset):
    def __init__(self, subset='training', data_path='./data/speech_commands_v0.02'):
        self.data_path = data_path
        self.subset = subset

        def load_list(filename):
            with open(os.path.join(data_path, filename)) as f:
                return [os.path.normpath(os.path.join(data_path, line.strip())) for line in f]

        # List all .wav paths
        all_files = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".wav"):
                    all_files.append(os.path.join(root, file))

        if subset == "validation":
            self.files = load_list("validation_list.txt")
        elif subset == "testing":
            self.files = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self.files = [f for f in all_files if f not in excludes]
        else:
            raise ValueError(f"Unknown subset: {subset}")

        # Set of valid labels
        self.labels = sorted({os.path.basename(os.path.dirname(f)) for f in self.files})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = os.path.basename(os.path.dirname(path))
        waveform, sr = torchaudio.load(path)

        # Resample if needed
        if sr != 16000:
            waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)

        return waveform, label


def label_to_index(label, labels):
    return torch.tensor(labels.index(label))


def collate_fn(batch, labels):
    tensors, targets = [], []
    for waveform, label in batch:
        spec = mel_transform(waveform)
        spec_db = db_transform(spec)
        spec_db = spec_db.expand(3, -1, -1)
        spec_db = resize(spec_db)
        tensors.append(spec_db)
        targets.append(label_to_index(label, labels))

    return torch.stack(tensors), torch.tensor(targets)


def get_speechcommands_dataloaders(batch_size=64, num_workers=4):
    data_path = './data/speech_commands_v0.02'

    train_set = CustomSpeechCommands('training', data_path)
    val_set = CustomSpeechCommands('validation', data_path)

    labels = train_set.labels  # shared label list across all splits

    def make_loader(ds):
        return DataLoader(ds,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=lambda x: collate_fn(x, labels),
                          num_workers=num_workers)

    return make_loader(train_set), make_loader(val_set), labels

