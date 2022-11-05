import os
from collections import ChainMap
from typing import Optional, Type, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader

from datasets.dataset_interface import DatasetInterface


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        dataset_args: Dict[str, Any],
        batch_size: int,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dataloader_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self.dataset_cls = dataset_cls
        self.dataset_args = dataset_args
        self.batch_size = batch_size
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.dataloader_args = ChainMap(dataloader_args or {}, {'shuffle': True, 'num_workers': 8, 'pin_memory': True})

        # those attributes will be set during setup
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        self.dataset_cls(**ChainMap({'train': True}, self.dataset_args))
        self.dataset_cls(**ChainMap({'train': False}, self.dataset_args))

    def setup(self, stage: Optional[str] = None, validation_split_ratio: float = 0.1):
        # even though these datasets are identical, we have to create it once for the training data and once for
        # the validation data because of the different transformations
        train_data: DatasetInterface = self.dataset_cls(
            **ChainMap({'train': True}, self.dataset_args), transform=self._train_transforms
        )
        val_data: DatasetInterface = self.dataset_cls(
            **ChainMap({'train': True}, self.dataset_args), transform=self._val_transforms
        )
        train_indices, val_indices = train_test_split(
            list(range(len(train_data))),
            test_size=validation_split_ratio,
            random_state=int(os.environ['PL_GLOBAL_SEED']),
            shuffle=True,
            stratify=train_data.targets)
        self.train_data = Subset(train_data, train_indices)
        self.val_data = Subset(val_data, val_indices)

        self.test_data = self.dataset_cls(
            **ChainMap({'train': False}, self.dataset_args), transform=self._test_transforms
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, batch_size=self.batch_size, **self.dataloader_args)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data, batch_size=self.batch_size, **ChainMap({'shuffle': False}, self.dataloader_args)
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, **ChainMap({'shuffle': False}, self.dataloader_args)
        )

    def num_classes(self) -> int:
        return len(self.dataset_cls(**ChainMap({'train': False}, self.dataset_args)).classes)
