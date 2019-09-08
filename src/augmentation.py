from albumentations import *


def train_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(),
        Normalize()
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Normalize()
    ], p=1)
