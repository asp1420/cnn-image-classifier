import os
import torch
import numpy as np

from PIL import Image
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torch import Tensor
from albumentations import Compose


class RetinaDataset(Dataset):
    ALLOWED_FORMATS = ['.jpg', '.png', '.jpeg']

    def __init__(self, root: str, transform: Compose=None) -> None:
        self.transform = transform
        self.images = list()
        for root, _, files in os.walk(root):
            for file_name in files:
                _, ext = os.path.splitext(file_name)
                if ext not in self.ALLOWED_FORMATS:
                    continue
                self.images.append(join(root, file_name))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_name = self.images[index]
        label = int(image_name.split('/')[-2])
        image = Image.open(image_name).convert('RGB')
        image = np.array(image)
        label = torch.as_tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = to_tensor(image)
        return image, label

    def __len__(self) -> int:
        return len(self.images)
