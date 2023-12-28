import os
from typing import Union, Optional, TypeAlias, Tuple, List
import pandas as pd
import numpy as np
import albumentations as albu
import cv2
from torch.utils.data import Dataset


TRANSFORM_TYPE: TypeAlias = Union[albu.BasicTransform, albu.BaseCompose]


class BarCodeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.transforms = transforms
        self.datafolder = data_folder
        self.df = df
        self.codes = self.cache_labels()
        self.crops = self.cache_images()
  
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, int]:
        text = self.codes[idx]
        image = self.crops[idx]

        data = {
            'image': image,
            'text': text,
            'text_length': len(text),
        }

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['text'], data['text_length']

    def __len__(self) -> int:
        return len(self.crops)

    def cache_images(self) -> List[np.ndarray]:
        crops = []
        for i in range(len(self.df)):
            image_path = os.path.join(self.datafolder, self.df['filename'][i])
            crop = cv2.imread(image_path)
            if crop is not None:
                crop = crop[..., ::-1]
                crops.append(crop)
            else:
                print(f"Failed to load image: {image_path}")
        return crops

    def cache_labels(self) -> List[str]:
        codes = []
        for i in range(len(self.df)):
            codes.append(str(self.df['code'][i]))
        return codes
