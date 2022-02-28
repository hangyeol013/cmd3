import os
import torch
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset



class CMD3Dataset(Dataset):
    
    """
    CMD3Audio dataset.
    :param data_path: path to the file (`.csv`) with audio paths and labels.
    :param video_path_prefix: path to the directories containing audio.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        audio_path_prefix: str,
    ):
        super(CMD3Dataset, self).__init__()
        self.root_path = root_path
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
            audio_path = self.root_path + '/dataset/' + audio_filename + '.wav'
            audio_paths.append((audio_path, int(label)))

        return audio_paths

    def _get_sample(self, audio_paths):
        
        samples = []
        for path in audio_paths:

            # print("path: ", path)
            audio_path, label = path
            save_path = audio_path.replace("audio", "audio_samples")[:28]
            file_name = audio_path.replace("audio", "audio_samples")[28:].replace('wav', 'pt')
            save_file_name = os.path.join(save_path, file_name)
            sample = torch.load(save_file_name)

            for i in range(sample.shape[0]):
                audio_time = round(i*0.96, 2)
                samples.append((sample[i], int(label), audio_path, audio_time))

        return samples
        

    def __getitem__(self, index: int) -> Dict[str, Any]:
        audio_sample, sample_label, audio_path, time_frame = self.audio_sample[index]
        outputs = {"label": sample_label, "id": audio_path, "time_frame": time_frame}

        outputs["audio"] = audio_sample

        return outputs

    def __len__(self) -> int:
        print("audio num: ", len(self.audio_paths))
        print("sample num: ", len(self.audio_sample))
        return len(self.audio_sample)


if __name__ == "__main__":

    DATA_PATH = "./dataset/set_splits/train_split2.csv"
    AUDIO_PATH_PREFIX = "./dataset/audio"


    usd = CMD3Dataset(DATA_PATH,
                      AUDIO_PATH_PREFIX
                      )

    print(f"There are {len(usd)} samples in the dataset.")
    signal = usd[0]