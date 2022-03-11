import os.path as osp
from typing import List

from pytorch_lightning import LightningDataModule
from pytorchvideo.transforms import Normalize, RandomShortSideScale
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomHorizontalFlip,
    RandomCrop,
    Resize,
)

from .dataset import CMD3Dataset


class CMD3DataModule(LightningDataModule):
    """Load train, val and test data loaders.

    :param root_dir: path to the root directory of the dataset.
    :param frame_size: frame resizing size.
    :param n_samples: number of frames which composes a sample.
    :param clip_duration: total number of frames in one sub clip.
    :param batch_size: how many samples per batch to load.
    :param num_workers: how many subprocesses to use for data loading.
    """

    def __init__(
        self,
        root_dir: str,
        modalities: List[str],
        frame_size: int,
        n_samples: int,
        clip_duration: int,
        batch_size: int,
        num_workers: int,
        augmentation: bool,
        normalize: bool,
    ):
        super(CMD3DataModule, self).__init__()

        self.root_dir = root_dir
        self.frame_dir = osp.join(self.root_dir, "dataset/video_jpg")
        self.feature_dir = osp.join(self.root_dir, "custom_features/cmd3_video")
        self.label_dir = osp.join(self.root_dir, "dataset/set_splits")

        self.modalities = modalities
        self.frame_size = frame_size
        self.n_samples = n_samples
        self.clip_duration = clip_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.normalize = normalize

    def _video_transform(self, mode: str) -> Compose:
        """Video transformations to apply."""
        transform_list = []

        if mode != "train":
            transform_list.append(Resize((self.frame_size, self.frame_size)))
        elif not self.augmentation:
            transform_list.append(Resize((self.frame_size, self.frame_size)))

        # transform_list.append(Lambda(lambda x: x / 255.0))
        if self.normalize:
            transform_list.append(
                Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
            )

        if mode == "train":
            if self.augmentation:
                transform_list.extend(
                    [
                        RandomShortSideScale(min_size=260, max_size=320),
                        RandomCrop(self.frame_size),
                        RandomHorizontalFlip(p=0.5),
                    ]
                )
            # else:
            #     transform_list.append(RandomHorizontalFlip(p=0.5))

        return Compose(transform_list)

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = CMD3Dataset(
            data_path=osp.join(self.label_dir, "train_split.csv"),
            video_path_prefix=self.frame_dir,
            feature_path_prefix=self.feature_dir,
            modalities=self.modalities,
            n_samples=self.n_samples,
            clip_duration=self.clip_duration,
            clip_sampling="uniform",
            transform=self._video_transform("train"),
        )

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Load validation set loader."""
        self.val_set = CMD3Dataset(
            data_path=osp.join(self.label_dir, "val_split.csv"),
            video_path_prefix=self.frame_dir,
            feature_path_prefix=self.feature_dir,
            modalities=self.modalities,
            n_samples=self.n_samples,
            clip_duration=self.clip_duration,
            clip_sampling="uniform",
            transform=self._video_transform("val"),
        )

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = CMD3Dataset(
            data_path=osp.join(self.label_dir, "test_split.csv"),
            video_path_prefix=self.frame_dir,
            feature_path_prefix=self.feature_dir,
            modalities=self.modalities,
            n_samples=self.n_samples,
            clip_duration=self.clip_duration,
            clip_sampling="uniform",
            transform=self._video_transform("test"),
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
