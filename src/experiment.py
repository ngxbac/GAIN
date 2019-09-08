from collections import OrderedDict
import torch
import torch.nn as nn
import random
from catalyst.dl.experiment import ConfigExperiment
from dataset import *
from augmentation import train_aug, valid_aug
from torchvision.datasets import ImageFolder


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        return model_

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        image_size = kwargs.get("image_size", 320)
        train_data_txt = kwargs.get('train_data_txt', None)
        valid_data_txt = kwargs.get('valid_data_txt', None)
        root = kwargs.get('root', None)

        if train_data_txt:
            transform = train_aug(image_size)
            train_set = IP102Dataset(
                data_txt=train_data_txt,
                transform=transform,
                root=root
            )
            datasets["train"] = train_set

        if valid_data_txt:
            transform = valid_aug(image_size)
            valid_set = IP102Dataset(
                data_txt=valid_data_txt,
                transform=transform,
                root=root
            )
            datasets["valid"] = valid_set

        flower_train = kwargs.get('flower_train', None)
        flower_valid = kwargs.get('flower_valid', None)
        flower_root = kwargs.get('flower_root', None)

        if flower_train:
            transform = train_aug(image_size)
            train_set = FlowerDataset(
                csv_file=flower_train,
                transform=transform,
                root=flower_root
            )
            datasets["train"] = train_set

        if flower_valid:
            transform = valid_aug(image_size)
            valid_set = FlowerDataset(
                csv_file=flower_valid,
                transform=transform,
                root=flower_root
            )
            datasets["valid"] = valid_set

        return datasets


