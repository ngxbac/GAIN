import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class IP102Dataset(Dataset):
    def __init__(self, data_txt, root=None, transform=None):
        with open(data_txt, 'r') as f:
            lines = f.readlines()
            images = [line.split(" ")[0] for line in lines]
            targets = [int(line.split(" ")[1]) for line in lines]

        self.images = images
        self.targets = targets
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        target = self.targets[idx]

        image = os.path.join(self.root, image_name)
        image = load_image(image)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'images': image,
            'targets': target,
            'image_names': image_name
        }


class FlowerDataset(Dataset):
    def __init__(self, csv_file, root=None, transform=None):
        df = pd.read_csv(csv_file)

        self.images = df['file'].values
        self.targets = df['label'].values
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        target = self.targets[idx]

        image = os.path.join(self.root, image_file)
        image = load_image(image)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_name = image_file.split("/")[-1]
        return {
            'images': image,
            'targets': target,
            'image_names': image_name
        }
