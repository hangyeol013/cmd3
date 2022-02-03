import os
import os.path as osp
from typing import Any, Dict, List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from models import vggish_input, vggish_params
from pathlib import Path

class CMD3Dataset(Dataset):
    
    """
    CMD3Audio dataset.
    :param data_path: path to the file (`.csv`) with audio paths and labels.
    :param video_path_prefix: path to the directories containing audio.
    """

    def __init__(
        self,
        data_path: str,
        audio_path_prefix: str,
    ):
        super(CMD3Dataset, self).__init__()
        self.data_path = data_path
        self.audio_path_prefix = audio_path_prefix

        self.audio_paths = self._get_audiopaths()
        self.audio_sample = self._get_sample(self.audio_paths)

    def _get_audiopaths(self) -> List[Tuple[str, int]]:
        """Get paths to all directories containing audio. """
        with open(self.data_path) as f:
            lines = f.read().splitlines()
        audio_paths = []
        for line in lines:
            audio_filename, label = line.split(".mkv")
            audio_path = '/home/vic-kang/cmd3/cmd3_audio/dataset/' + audio_filename + '.wav'
            audio_paths.append((audio_path, int(label)))

        print("num_audio: ", len(audio_paths))

        return audio_paths

    def _get_sample(self, audio_paths):
        
        samples = []
        for path in audio_paths:

            print("path: ", path)
            audio_path, label = path
            sample = vggish_input.wavfile_to_examples(audio_path)

            for i in range(sample.shape[0]):
                samples.append((sample[i], int(label)))

            print(len(samples))

        return samples

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
