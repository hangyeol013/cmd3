import os.path as osp
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset

from models import vggish_input


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
            audio_path = self.root_path + '/' + audio_filename + '.wav'
            audio_paths.append((audio_path, int(label)))

        print("num_audio: ", len(audio_paths))

        return audio_paths

    def _get_sample(self, audio_paths):
        
        samples = []
        for path in audio_paths:

            print("path: ", path)
            audio_path, label = path
            sample = vggish_input.wavfile_to_examples(audio_path)
            # sample_path = audio_path.split('dataset/')[-1]

            for i in range(sample.shape[0]):
                samples.append((sample[i], int(label), audio_path))

        return samples
        

    def __getitem__(self, index: int) -> Dict[str, Any]:
        audio_sample, sample_label, audio_path = self.audio_sample[index]
        # path_index = index // (len(self.audio_sample) // len(self.audio_paths))
        # audio_path, _ = self.audio_paths[path_index]
        # audio_id = audio_path.split(osp.sep)[-1]
        outputs = {"label": sample_label, "id": audio_path}

        outputs["audio"] = audio_sample

        return outputs

    def __len__(self) -> int:
        print(f"There are {len(self.audio_sample)} samples in the dataset.")
        return len(self.audio_sample)


if __name__ == "__main__":

    DATA_PATH = "cmd3_audio/dataset/set_splits/train_split2.csv"
    AUDIO_PATH_PREFIX = "cmd3_audio/dataset/audio"


    usd = CMD3Dataset(DATA_PATH,
                      AUDIO_PATH_PREFIX
                      )

    print(f"There are {len(usd)} samples in the dataset.")
    signal = usd[0]