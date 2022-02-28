import os
import os.path as osp
from typing import Callable, Any, Dict, List, Tuple
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def load_npy(file_path: str) -> np.array:
    """Load a `.npy` file."""
    with open(file_path, "rb") as f:
        data = np.load(f)

    return data


class CMD3Dataset(Dataset):
    """CMD8 dataset.

    :param data_path: path to the file (`.csv`) with video paths and labels.
    :param video_path_prefix: path to the directories containing frames.
    :param feature_path_prefix: path to the directories containing features.
    :param modalities: modalities to load (raw, depth and/or layer).
    :param n_samples: number of frames which composes a sample.
    :param clip_duration: total number of frames in one sub clip.
    :param clip_sampling: type of sampling (random or uniform).
    :param transform: transformations to apply to one sample.
    """

    def __init__(
        self,
        data_path: str,
        video_path_prefix: str,
        feature_path_prefix: str,
        modalities: List[str],
        n_samples: int,
        clip_duration: int,
        clip_sampling: str,
        transform: Callable = None,
    ):
        super(CMD3Dataset, self).__init__()
        self.data_path = data_path
        self.video_path_prefix = video_path_prefix
        self.feature_path_prefix = feature_path_prefix
        self.modalities = modalities
        self.n_samples = n_samples
        self.clip_duration = clip_duration
        self.clip_sampling = clip_sampling
        self.transform = transform

        self.video_paths = self._get_videopaths()
        self.sample_paths = self._get_samplepaths()

    def _get_videopaths(self) -> List[Tuple[str, int]]:
        """Get paths to all directories containing videos' frames."""
        with open(self.data_path) as f:
            lines = f.read().splitlines()

        video_paths = []
        for line in lines:
            video_filename, label = line.split(".mkv")
            video_path = osp.join(self.video_path_prefix, video_filename[6:])
            video_paths.append((video_path, int(label)))
            # for shot_index in os.listdir(video_path):
            #     video_shot_path = osp.join(video_path, shot_index)
            #     video_paths.append((video_shot_path, int(label)))

        return video_paths

    @staticmethod
    def _sample_uniformclips(
        last_index: int, duration: int
    ) -> List[Tuple[int, int]]:
        """Sample all non-overlapping clips of size `duration`."""
        sampled_clips = []
        previous_index = 0
        for current_index in range(duration, last_index, duration):
            sampled_clips.append((previous_index, current_index))
            previous_index = current_index

        return sampled_clips

    def _get_samplepaths(self) -> List[Tuple[List[str], int]]:
        """Get paths to each frame constituing a sample."""
        sample_paths = []
        for video_path, label in self.video_paths:
            frame_list = np.array(sorted(os.listdir(video_path)))
            max_frames = len(frame_list)

            if self.clip_sampling == "random":
                sample_indices = self._sample_randomclips(
                    max_frames, self.clip_duration
                )

            elif self.clip_sampling == "uniform":
                sample_indices = self._sample_uniformclips(
                    max_frames, self.clip_duration
                )

            for start_index, end_index in sample_indices:
                indices = np.linspace(
                    start_index, end_index - 1, self.n_samples, dtype=int
                )
                frame_paths = [
                    osp.join(video_path, frame_filename)
                    for frame_filename in frame_list[indices]
                ]
                sample_paths.append((frame_paths, label))

        return sample_paths

    def _load_raw(self, sample_framepaths: List[str]) -> torch.Tensor:
        """Load and pre-process raw frames."""
        frames = torch.stack(
            [
                ToTensor()(Image.open(frame_path))
                for frame_path in sample_framepaths
            ]
        ).permute(1, 0, 2, 3)

        if self.transform is not None:
            frames = self.transform(frames)

        return frames

    def _load_depth(self, sample_framepaths: List[str]) -> torch.Tensor:
        """Load and pre-process depth maps."""
        depth_maps = []
        for frame_path in sample_framepaths:
            split_framepath = frame_path.split(osp.sep)
            path_prefix = self.feature_path_prefix
            path_sufix = osp.join(*split_framepath[-4:])[:-4] + ".npy"
            depth_path = osp.join(path_prefix, "depth_maps", path_sufix)
            depth_maps.append(torch.from_numpy(load_npy(depth_path)) * 255)

        depth_maps = torch.stack(depth_maps)
        if self.transform is not None:
            depth_maps = self.transform(depth_maps)

        return depth_maps

    def _load_layer(self, sample_framepaths: List[str]) -> torch.Tensor:
        """Load and pre-process layer maps."""
        layer_maps = []
        for frame_path in sample_framepaths:
            split_framepath = frame_path.split(osp.sep)
            path_prefix = self.feature_path_prefix
            path_sufix = osp.join(*split_framepath[-4:])[:-4] + ".npy"
            layer_path = osp.join(path_prefix, "layer_maps", path_sufix)
            raw_layer_map = load_npy(layer_path)
            max_cluster = raw_layer_map.max()
            layer_maps.append(
                torch.from_numpy(raw_layer_map).float() / max_cluster * 255
            )

        layer_maps = torch.stack(layer_maps)
        if self.transform is not None:
            layer_maps = self.transform(layer_maps)

        return layer_maps

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_framepaths, sample_label = self.sample_paths[index]
        # clip_id = sample_framepaths[0].split(osp.sep)[-3]
        clip_id = sample_framepaths[0].split('/image')[0]
        outputs = {"label": sample_label, "id": clip_id}

        for current_modality in self.modalities:
            if current_modality == "raw":
                frames = self._load_raw(sample_framepaths)
                outputs["video"] = frames
            elif current_modality == "depth":
                depth_maps = self._load_depth(sample_framepaths)
                outputs["depth"] = depth_maps
            elif current_modality == "layer":
                layer_maps = self._load_layer(sample_framepaths)
                outputs["layer"] = layer_maps

        return outputs

    def __len__(self) -> int:
        print("video nums: ", len(self.video_paths))
        print("sample nums: ", len(self.sample_paths))
        return len(self.sample_paths)
