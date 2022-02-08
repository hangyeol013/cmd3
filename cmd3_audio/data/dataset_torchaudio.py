import os
import os.path as osp
from typing import Any, Dict, List, Tuple
from PIL import Image

import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from pathlib import Path

class CMD3Dataset(Dataset):
    
    """
    CMD3Audio dataset.
    :param data_path: path to the file (`.csv`) with audio paths and labels.
    :param video_path_prefix: path to the directories containing audio.
    """

    def __init__(
        self,
        data_path,
        audio_path_prefix,
        mel_spectrogram, 
        sample_rate,
        num_samples,
        device
    ):
        super(CMD3Dataset, self).__init__()
        self.data_path = data_path
        self.audio_path_prefix = audio_path_prefix
        self.device = device
        self.transforms = mel_spectrogram.to(self.device)
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples
        

        self.audio_paths = self._get_audiopaths()
        self.audio_sample = self._get_sample(self.audio_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        audio_sample, sample_label = self.audio_sample[index]
        path_index = index // (len(self.audio_sample)+1)
        audio_paths, _ = self.audio_paths[path_index]
        audio_id = audio_paths.split(osp.sep)[-1]
        outputs = {"label": sample_label, "id": audio_id}

        outputs["audio"] = audio_sample

        return outputs

    def __len__(self) -> int:
        return len(self.audio_sample)

    def _get_audiopaths(self) -> List[Tuple[str, int]]:
        """Get paths to all directories containing audio. """
        with open(self.data_path) as f:
            lines = f.read().splitlines()
        audio_paths = []
        for line in lines:
            audio_filename, label = line.split(".mkv")
            # audio_path = '/home/vic-kang/cmd3/cmd3_audio/dataset/' + audio_filename + '.wav'
            audio_path = '/cmd3_audio/dataset/' + audio_filename + '.wav'
            audio_paths.append((audio_path, int(label)))

        print("num_audio: ", len(audio_paths))

        return audio_paths

    def _get_sample(self, audio_paths):
        
        samples = []
        for path in audio_paths:

            print("path: ", path)
            audio_path, label = path
            signal, sr = torchaudio.load(audio_path)
            signal = signal.to(self.device)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
            signal = self.transforms(signal)

            # for i in range(signal.shape[0]):
            #     samples.append((signal[i], int(label)))

            print(signal.shape)

        return samples

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    


if __name__ == "__main__":

    DATA_PATH = "cmd3_audio/dataset/set_splits/train_split.csv"
    AUDIO_PATH_PREFIX = "cmd3_audio/dataset/audio"
    SAMPLE_RATE = 8000
    NUM_SAMPLES = 8000

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using Device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 960,
        win_length=200,
        hop_length = 80,
        n_mels = 64
    )

    usd = CMD3Dataset(DATA_PATH,
                      AUDIO_PATH_PREFIX,
                      mel_spectrogram,
                      SAMPLE_RATE,
                      NUM_SAMPLES,
                      device)

    print(f"Tehre are {len(usd)} samples in the dataset.")
    signal = usd[0]