import pytorch_lightning as pl
import torch 
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
from pathlib import Path

from lib.training.dataset import TrainDataset
from lib.models.unet import UNet

from lib.training.loss import DCandCELoss
from lib.training.metrics import DiceCoefficient

from lib.inference.utils import write_nii, interpolate, makedirs
from lib.inference.dataset import TestDataset, patch_generator
from lib.patch import reconstruct_from_patches, sample_patch_by_sliwin
from lib.utils import read_image


def collate_fn(x):
    return x[0]

class SegmentModule(pl.LightningModule):
    def __init__(self, model_config=None, training_config=None, data_config=None, test_config=None):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.test_config = test_config
        self.model = self.configure_model()
        
        self.save_hyperparameters("model_config", "training_config", "data_config")

    def setup(self, stage):
        if stage == 'fit':
            self.loss = self.configure_loss()
            self.metric_fn = DiceCoefficient()
        elif stage == 'test':
            makedirs(self.test_config["result_dir"])

    def train_dataloader(self):
        train_ds = TrainDataset(
            self.data_config["train_excel"], 
            self.data_config["image_header"], 
            self.data_config["mask_header"], 
            self.data_config["n_patches_per_image"], 
            self.data_config["batch_size"], 
            self.data_config["patch_size"], 
            self.data_config["selected_labels"], 
            self.data_config["inten_norm"], 
            fg_ratio=self.data_config["fg_ratio"], 
            augment=True)
        train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)
        return train_loader

    def val_dataloader(self):
        val_ds = TrainDataset(
            self.data_config["val_excel"], 
            self.data_config["image_header"], 
            self.data_config["mask_header"], 
            self.data_config["n_patches_per_image"], 
            self.data_config["batch_size"], 
            self.data_config["patch_size"], 
            self.data_config["selected_labels"],
            self.data_config["inten_norm"],
            fg_ratio=self.data_config["fg_ratio"], 
            augment=False)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)
        return val_loader

    def test_dataloader(self):
        test_ds = TestDataset(self.test_config["test_excel"], self.test_config["patch_size"], self.test_config["normalization"])
        return DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)

    def configure_model(self):
        n_classes = len(self.data_config["selected_labels"]) + 1
        if self.model_config["model"] == "unet":
            return UNet(self.model_config["n_channels"], n_classes, self.model_config["norm"], self.model_config["nonlin"], deep_supervise=True, dropout=self.model_config["dropout"])

    def configure_loss(self):
        if self.training_config["loss_function"]["loss"] == "dc_and_ce":
            return DCandCELoss(self.training_config["loss_function"]["weights"])

    def configure_optimizers(self):
        if self.training_config["optimizer"] == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.training_config["lr"], momentum=0.95, nesterov=True)
        elif self.training_config["optimizer"] == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.training_config["lr"])

        lr_schedule = self.training_config["lr_schedule"]
        if lr_schedule["scheduler"] == "poly":
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / lr_schedule["k"]) ** 0.9)
        elif lr_schedule["scheduler"] == "plateau":
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=lr_schedule["patience"]),
                "reduce_on_plateau": True,
                "monitor": "val_loss"
            }
        else:
            lr_scheduler = None

        optim_config = optimizer if lr_scheduler is None else ([optimizer], [lr_scheduler])
        return optim_config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batchidx):
        data, target = batch

        logits = self.model(data)
        
        loss = 0.25 * self.loss(logits[0], target) + 0.5 * self.loss(logits[1], target) + self.loss(logits[2], target)

        metrics = self.metric_fn(logits[2], target)

        result = {"loss": loss}
        result.update({"%s_%d" % (str(self.metric_fn), i): metrics[i] for i in range(len(metrics))})

        self.log_dict(result, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batchidx):
        data, target = batch

        logits = self.model(data)
        
        loss = 0.25 * self.loss(logits[0], target) + 0.5 * self.loss(logits[1], target) + self.loss(logits[2], target)

        metrics = self.metric_fn(logits[2], target)

        result = {"val_loss": loss}
        result.update({"val_%s_%d" % (str(self.metric_fn), i): metrics[i] for i in range(len(metrics))})

        self.log_dict(result)
        return loss
    

    def test_step(self, batch, batchidx):
        # if batch["patch_indices"] is None:
        #     predictions = self.test_on_whole(batch["image"], batch["org_size"])
        # else:
        prediction = self.test_on_patch(batch["image"], batch["patch_indices"], batch["org_size"], batch["fine_patch_size"])

        np.save(os.path.join(self.test_config["result_dir"], Path(batch["path"]).stem), prediction)

    def test_on_whole(self, image, org_size):
        image_batch = np.repeat(image[np.newaxis, ...], self.test_config['batch_size'], axis=0)
        image_batch = np.moveaxis(image_batch, -1, 1)
        image_batch = torch.from_numpy(image_batch).to(device=self.device, dtype=torch.float)

        all_logits = self.model(image_batch)
    
        all_preds = [F.softmax(logits, axis=1) for logits in all_logits]

        predictions = [preds[self.test_config['batch_size'] // 2].detach().cpu().numpy() for preds in all_preds]

        predictions = interpolate(preditions, org_size)

        return predictions

    def test_on_patch(self, image, patch_indices, org_size, fine_patch_size):
        fine_patch_indices = sample_patch_by_sliwin(np.array(org_size), fine_patch_size, fine_patch_size // 2)

        patch_gen = patch_generator(image, patch_indices, self.test_config["patch_size"], self.test_config["batch_size"], self.device)

        all_patch_logits = [self.model(image_batch)[2] for image_batch in patch_gen]
        all_patch_logits = torch.cat(all_patch_logits, axis=0)
        restore_patch = partial(self.restore_patch, fine_patch_size=fine_patch_size, fine_patch_indices=fine_patch_indices, org_size=org_size)
        reconstruction = restore_patch(all_patch_logits)
        reconstruction = np.moveaxis(reconstruction[np.newaxis, ...], -1, 1)
        return reconstruction

    def restore_patch(self, patch_logits, fine_patch_size, fine_patch_indices, org_size):
        patch_logits = F.interpolate(patch_logits, size=fine_patch_size.tolist(), mode='bilinear')
        patch_logits = torch.movedim(patch_logits, 1, -1).detach().cpu().numpy()
        reconstruction = reconstruct_from_patches(patch_logits, fine_patch_indices, org_size)
        reconstruction = F.softmax(torch.from_numpy(reconstruction), dim=-1).numpy()
        return reconstruction