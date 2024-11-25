import albumentations as aug

from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose


def train_transform(size: int) -> Compose:
    transform = aug.Compose([
        aug.OneOf([
            aug.RandomBrightnessContrast(brightness_limit=0.0, p=1.0),
            aug.RandomGamma(p=1.0),
        ]),
        aug.OneOf([
            aug.Sharpen(p=1.0),
            aug.GaussianBlur(p=1.0),
        ]),
        aug.Rotate(limit=5),
        aug.HorizontalFlip(),
        aug.Normalize(),
        aug.Resize(width=size, height=size),
        ToTensorV2()
    ])
    return transform


def valid_transform(size: int) -> Compose:
    transform = aug.Compose([
        aug.Normalize(),
        aug.Resize(width=size, height=size),
        ToTensorV2()
    ])
    return transform


def test_transform(size: int) -> Compose:
    transform = valid_transform(size=size)
    return transform
