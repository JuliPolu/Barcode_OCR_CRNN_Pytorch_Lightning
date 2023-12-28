from typing import Optional
import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
from constants import DATA_PATH
from config import DataConfig
from transforms import get_transforms
from dataset import BarCodeDataset


class OCRDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config
        self._train_transforms = get_transforms(
            width=config.width,
            height=config.height,
            vocab=config.vocab,
            text_size=config.text_size,
        )
        self._valid_transforms = get_transforms(
            width=config.width,
            height=config.height,
            vocab=config.vocab,
            text_size=config.text_size,
            augmentations=False,
        )

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.train_sampler: Optional[RandomSampler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        df_train = read_df(self._config.data_path, 'train')
        df_valid = read_df(self._config.data_path, 'valid')

        self.train_dataset = BarCodeDataset(
            df=df_train,
            data_folder=self._config.data_path,
            transforms=self._train_transforms,
        )
        self.valid_dataset = BarCodeDataset(
            df=df_valid,
            data_folder=self._config.data_path,
            transforms=self._valid_transforms,
        )

        if self._config.num_iterations != -1:
            self.train_sampler = RandomSampler(
                data_source=self.train_dataset,
                num_samples=self._config.num_iterations * self._config.batch_size,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            sampler=self.train_sampler,
            shuffle=False if self.train_sampler else True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.normpath(os.path.join(data_path, f'df_{mode}.csv')))
