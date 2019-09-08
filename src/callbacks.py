from catalyst.dl.core import Callback, RunnerState
from catalyst.dl.utils.criterion import accuracy
from catalyst.dl.callbacks.logging import TxtMetricsFormatter
from catalyst.contrib.criterion import IoULoss, BCEIoULoss
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List
import wandb


class GAINCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_cls_key: str = "logits",
        output_am_key: str = "logits_am",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
    ):
        self.input_key = input_key
        self.output_cls_key = output_cls_key
        self.output_am_key = output_am_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs_cls = state.output[self.output_cls_key]
        outputs_am = state.output[self.output_am_key]
        input = state.input[self.input_key]
        loss = criterion(outputs_cls, input) * 0.8
        loss_am = F.softmax(outputs_am)
        loss_am, _ = loss_am.max(dim=1)
        loss_am = loss_am.sum() / loss_am.size(0)
        loss += loss_am * 0.2
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class GAINMaskCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        input_mask: str = "masks",
        output_cls_key: str = "logits",
        output_am_key: str = "logits_am",
        output_soft_mask_key: str = "soft_mask",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
    ):
        self.input_key = input_key
        self.input_mask = input_mask
        self.output_cls_key = output_cls_key
        self.output_am_key = output_am_key
        self.output_soft_mask_key = output_soft_mask_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.soft_mask_criterion = nn.BCELoss()

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs_cls = state.output[self.output_cls_key]
        outputs_am = state.output[self.output_am_key]
        output_soft_mask = state.output[self.output_soft_mask_key]
        input = state.input[self.input_key]
        input_mask = state.input[self.input_mask]
        loss = criterion(outputs_cls, input) * 0.8
        loss_am = F.softmax(outputs_am)
        loss_am, _ = loss_am.max(dim=1)
        loss_am = loss_am.sum() / loss_am.size(0)
        loss_mask = self.soft_mask_criterion(output_soft_mask, input_mask)
        loss += loss_am * 0.1
        loss += loss_mask * 0.1
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class GAINSaveHeatmapCallback(Callback):
    def __init__(
        self,
        heatmap_key: str = 'heatmap',
        image_name_key: str = 'image_names',
        image_key: str = 'images',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        outdir: str = './heatmaps/'
    ):
        self.heatmap_key = heatmap_key
        self.image_name_key = image_name_key
        self.image_key = image_key
        self.mean = mean
        self.std = std
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            outdir = os.path.join(self.outdir, f"epoch{state.epoch}")
            os.makedirs(outdir, exist_ok=True)
            images = state.input[self.image_key]
            heatmaps = state.output[self.heatmap_key]
            image_names = state.input[self.image_name_key]

            # rand_wandb_images = np.random.randint(0, len(image_names), 2)

            for i, (image, ac, image_name) in enumerate(zip(images, heatmaps, image_names)):
                ac = ac.data.cpu().numpy()[0]
                heat_map = self._combine_heatmap_with_image(
                    image=image,
                    heatmap=ac
                )
                cv2.imwrite(f"{outdir}/{image_name}", heat_map)

                # if i in rand_wandb_images:
                #     wandb.log({"examples": [wandb.Image(heat_map, caption=image_name)]})

                # mask = mask.detach().cpu().numpy() * 255
                # mask = mask[0]
                # cv2.imwrite(f"{outdir}/{image_name}_mask.jpg", mask)

    def denorm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def _combine_heatmap_with_image(self, image, heatmap):
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))

        scaled_image = self.denorm(image) * 255
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))

        cam = heatmap + np.float32(scaled_image)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        heat_map = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
        return heat_map


class GCAMSaveHeatmapCallback(Callback):
    def __init__(
        self,
        # feedforward_key: str = 'feedforward',
        # backward_key: str = 'backward',
        head_map_key: str = 'heatmap',
        image_name_key: str = 'image_names',
        image_key: str = 'images',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        outdir: str = './heatmaps/'
    ):
        # self.feedforward_key = feedforward_key
        # self.backward_key = backward_key
        self.head_map_key = head_map_key
        self.image_name_key = image_name_key
        self.image_key = image_key
        self.mean = mean
        self.std = std
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            outdir = os.path.join(self.outdir, f"epoch{state.epoch}")
            os.makedirs(outdir, exist_ok=True)
            images = state.input[self.image_key]
            heatmaps = state.output[self.head_map_key]
            image_names = state.input[self.image_name_key]

            for image, heatmap, image_name in zip(images, heatmaps, image_names):
                # backward = backward.unsqueeze(0)
                # weight = backward.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
                # heatmap = F.relu((weight * forward).sum(dim=1)).squeeze(0)
                # heatmap = cv2.resize(heatmap.data.cpu().numpy(), images.size()[2:])
                heatmap = heatmap.data.cpu().numpy()[0]

                heat_map = self._combine_heatmap_with_image(
                    image=image,
                    heatmap=heatmap
                )
                cv2.imwrite(f"{outdir}/{image_name}", heat_map)

    def denorm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def _combine_heatmap_with_image(self, image, heatmap):
        # import pdb
        # pdb.set_trace()
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))

        scaled_image = self.denorm(image) * 255
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))

        cam = heatmap + np.float32(scaled_image)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        heat_map = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
        return heat_map

