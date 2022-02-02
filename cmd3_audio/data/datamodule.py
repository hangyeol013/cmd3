import os.path as osp
from typing import List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


from data.dataset import CMD3Dataset



class CMD3DataModule(LightningDataModule):

    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int,
    ):
        super(CMD3DataModule, self).__init__()

        self.root_dir = root_dir
        self.audio_dir = osp.join(self.root_dir, 'audio')
        self.label_dir = osp.join(self.root_dir, "set_splits")

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = CMD3Dataset(
            data_path=osp.join(self.label_dir, "train_split.csv"),
            audio_path_prefix=self.audio_dir,
        )

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Load validation set loader."""
        self.val_set = CMD3Dataset(
            data_path=osp.join(self.label_dir, "val_split.csv"),
            audio_path_prefix=self.audio_dir,
        )

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = CMD3Dataset(
            data_path=osp.join(self.label_dir, "test_split.csv"),
            audio_path_prefix=self.audio_dir,
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )