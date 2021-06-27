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
import SimpleITK as sitk

from lib.training.dataset import TrainDataset
from lib.models.unet import UNet
from lib.models.unet_plus_plus import UNetPlusPlus
from lib.models.acunet import ACUNet
from lib.models.canny_net import Net

from lib.training.loss import MultiHeadDiceLoss, MultiHeadCELoss, MultiHeadDCandCELoss
from lib.training.metrics import MultiHeadDiceCoefficient

from lib.inference.utils import write_nii, makedirs, postprocess
from lib.inference.dataset import TestDataset, patch_generator
from lib.evaluation.calculate_metrics import measure_overlap, measure_surface, bootstrap_confidence_interval
from lib.evaluation.utils import metric_logging
from lib.patch import reconstruct_from_patches, sample_patch_by_sliwin
from lib.utils import read_image, whd_2_hwd


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
            self.metric_fn = MultiHeadDiceCoefficient(n_heads=len(self.data_config["selected_labels"]))

    def train_dataloader(self):
        train_ds = TrainDataset(
            self.data_config["train_excel"], 
            self.data_config["image_headers"], 
            self.data_config["mask_headers"], 
            self.data_config["n_patches_per_image"], 
            self.data_config["batch_size"], 
            self.data_config["patch_size"], 
            self.data_config["selected_labels"], 
            self.data_config["inten_norm"], 
            fg_ratio=self.data_config["fg_ratio"], 
            augment=self.data_config["augment"])
        train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)
        return train_loader

    def val_dataloader(self):
        val_ds = TrainDataset(
            self.data_config["val_excel"], 
            self.data_config["image_headers"], 
            self.data_config["mask_headers"], 
            self.data_config["n_patches_per_image"], 
            self.data_config["batch_size"], 
            self.data_config["patch_size"], 
            self.data_config["selected_labels"],
            self.data_config["inten_norm"],
            fg_ratio=self.data_config["fg_ratio"], 
            augment=None)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)
        return val_loader

    def test_dataloader(self):
        test_ds = TestDataset(
            self.test_config["test_excel"], 
            self.test_config["image_header"], 
            self.test_config["mask_headers"], 
            self.test_config["selected_labels"], 
            self.test_config["patch_size"], 
            self.test_config["pixval_replacement"], 
            self.test_config["new_spacing"], 
            self.test_config["normalization"])
        return DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)

    def configure_model(self):
        n_classes = [sum(lbs) + 1 for lbs in self.data_config["selected_labels"]]
        if self.model_config["model"] == "unet":
            return UNet(self.model_config["n_channels"], n_classes, self.model_config["norm"], self.model_config["nonlin"], self.model_config["dropout"])
        elif self.model_config["model"] == "unet_plus_plus":
            return UNetPlusPlus(self.model_config["n_channels"], n_classes, self.model_config["norm"], self.model_config["nonlin"], self.model_config["dropout"])
        elif self.model_config["model"] == "acunet":
            return ACUNet(self.model_config["heights"], self.model_config["n_channels"], n_classes, self.model_config["norm"], self.model_config["nonlin"], self.model_config["dropout"])

    def configure_loss(self):
        loss_config =  self.training_config["loss_function"]
        if loss_config["loss"] == "dice":
            return MultiHeadDiceLoss(loss_config["weights"])
        elif loss_config["loss"] == "ce":
            return MultiHeadCELoss(loss_config["weights"])
        elif loss_config["loss"] == "dc_and_ce":
            return MultiHeadDCandCELoss(loss_config["weights"])

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
        data, targets = batch

        logits = self.model(data)

        loss = self.loss(logits, targets)
        
        metrics = self.metric_fn(logits, targets)

        result = {"loss": loss}
        for i in range(len(metrics)):
            result.update({"%s%d_%d" % (str(self.metric_fn), i, j): metrics[i][j] for j in range(len(metrics[i]))})

        self.log_dict(result, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batchidx):
        data, targets = batch

        logits = self.model(data)

        loss = self.loss(logits, targets)
        
        metrics = self.metric_fn(logits, targets)

        result = {"val_loss": loss}
        for i in range(len(metrics)):
            result.update({"val_%s%d_%d" % (str(self.metric_fn), i, j): metrics[i][j] for j in range(len(metrics[i]))})

        self.log_dict(result)
        return loss

    def test_step(self, batch, batchidx):
        predictions = self.predict_on_batch(batch["sitk_image"], batch["sitk_raw_image"])

        log = {}
        m_overlap_metrics = []
        m_surface_metrics = []
        for i, (reference, prediction) in enumerate(zip(batch["masks"], predictions)):
            overlap_metrics = measure_overlap(reference, prediction, self.test_config["overlap_metrics"], len(self.test_config["selected_labels"][i]))
            surface_metrics = measure_surface(reference, prediction, self.test_config["surface_metrics"], len(self.test_config["selected_labels"][i]), whd_2_hwd(batch["sitk_raw_image"].GetSpacing()))
            
            metric_logging(log, self.test_config["overlap_metrics"], self.test_config["mask_headers"][i], self.test_config["selected_labels"][i], overlap_metrics)
            metric_logging(log, self.test_config["surface_metrics"], self.test_config["mask_headers"][i], self.test_config["selected_labels"][i], surface_metrics)
            
            m_overlap_metrics.append(overlap_metrics)
            m_surface_metrics.append(surface_metrics)

        self.log_dict(log, on_step=True, on_epoch=True)

        return m_overlap_metrics, m_surface_metrics

    def test_epoch_end(self, outputs):
        outputs = np.asarray(outputs)
        outputs = np.moveaxis(outputs, 0, -1)
        all_overlap_metrics, all_surface_metrics = outputs

        log = {}
        overlap_ci_names = ["ci_" + m for m in self.test_config["overlap_metrics"]]
        surface_ci_names = ["ci_" + m for m in self.test_config["surface_metrics"]]

        for i, (mask_overlap_metrics, mask_surface_metrics) in enumerate(zip(all_overlap_metrics, all_surface_metrics)):
            mask_overlap_intervals = [[bootstrap_confidence_interval(m) for m in label_metrics] for label_metrics in mask_overlap_metrics]
            metric_logging(log, overlap_ci_names, self.test_config["mask_headers"][i], self.test_config["selected_labels"][i], mask_overlap_intervals)

            mask_surface_intervals = [[bootstrap_confidence_interval(m) for m in label_metrics] for label_metrics in mask_surface_metrics]
            metric_logging(log, surface_ci_names, self.test_config["mask_headers"][i], self.test_config["selected_labels"][i], mask_surface_intervals)
        
        self.log_dict(log)


    def predict_on_batch(self, sitk_image, sitk_raw_image):
        size, spacing = whd_2_hwd(sitk_image.GetSize()), whd_2_hwd(sitk_image.GetSpacing())
        raw_size, raw_spacing = whd_2_hwd(sitk_raw_image.GetSize()), whd_2_hwd(sitk_raw_image.GetSpacing())

        if len(self.test_config["patch_size"]) == 2:
            spacing, raw_spacing = spacing[:-1], raw_spacing[:-1]
            size, raw_size = size[:-1], raw_size[:-1]

        raw_patch_size = np.ceil(np.multiply(self.test_config["patch_size"], np.divide(spacing, raw_spacing))).astype(np.int32)

        raw_patch_indices = sample_patch_by_sliwin(np.array(raw_size), raw_patch_size, raw_patch_size // 2)
        patch_indices = sample_patch_by_sliwin(np.array(size), self.test_config["patch_size"], np.array(self.test_config["patch_size"]) // 2)

        image = np.moveaxis(sitk.GetArrayFromImage(sitk_image), 0, -1)

        patch_gen = patch_generator(image, patch_indices, self.test_config["patch_size"], self.test_config["batch_size"], self.device)

        all_patch_logits = [self.model(image_batch) for image_batch in patch_gen]
        all_patch_logits = [torch.cat([logits[i] for logits in all_patch_logits], axis=0) for i in range(len(self.model.n_classes))]

        restore_patch = partial(self.restore_patch, raw_patch_size=raw_patch_size, raw_patch_indices=raw_patch_indices, raw_size=raw_size)
        reconstructions = np.asarray([restore_patch(patch_logits) for patch_logits in all_patch_logits])
        
        predictions = [postprocess(recon, len(self.test_config["selected_labels"][i]), self.test_config["n_largest_components"][i]) for i, recon in enumerate(reconstructions)]
        return predictions

    def restore_patch(self, patch_logits, raw_patch_size, raw_patch_indices, raw_size):
        patch_logits = F.interpolate(patch_logits, size=raw_patch_size.tolist(), mode='bilinear')
        patch_logits = torch.movedim(patch_logits, 1, -1).detach().cpu().numpy()
        reconstruction = reconstruct_from_patches(patch_logits, raw_patch_indices, raw_size)
        reconstruction = F.softmax(torch.from_numpy(reconstruction), dim=-1).numpy()
        return reconstruction