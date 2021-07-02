import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
from pathlib import Path
from scipy.ndimage.morphology import distance_transform_edt
import SimpleITK as sitk

from lib.training.dataset import TrainDataset
from lib.models.mlp import MLP
from lib.models.canny_net import CannyNet
from lib.models.unet import UNet

from lib.training.loss import DCandCELoss
from lib.training.metrics import DiceCoefficient

from lib.evaluation.utils import metric_logging

from lib.inference.utils import write_nii, interpolate, makedirs, postprocess
from lib.inference.dataset import TestDataset, patch_generator
from lib.evaluation.calculate_metrics import measure_overlap, measure_surface, bootstrap_confidence_interval
from lib.patch import reconstruct_from_patches, sample_patch_by_sliwin
from lib.point import sample_uncertain_points, point_sample, calculate_uncertainty, get_uncertain_point_coords_on_grid
from lib.utils import read_image, whd_2_hwd


def collate_fn(x):
    return x[0]

class RenderModule(pl.LightningModule):
    def __init__(self, model_config=None, training_config=None, data_config=None, test_config=None):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.test_config = test_config

        self.segment_model = self.configure_model()
        self.in_features = [-3, -2, -1]


        n_classes = len(self.data_config["selected_labels"]) + 1
        self.local_render = MLP([226, 256, 256, n_classes], self.model_config["dropout"])
        self.global_render = MLP([514, 256, 256, n_classes], self.model_config["dropout"])

        self.save_hyperparameters("model_config", "training_config", "data_config")
    
    def configure_model(self):
        n_classes = len(self.data_config["selected_labels"]) + 1
        if self.model_config["model"] == "unet":
            return UNet(self.model_config["n_channels"], n_classes, self.model_config["norm"], self.model_config["nonlin"], self.model_config["dropout"])

    def setup(self, stage):
        if stage == 'fit':
            self.point_loss = nn.CrossEntropyLoss()
            self.seg_loss = DCandCELoss(self.training_config["loss_weights"])
            self.metric_fn = DiceCoefficient()

    def train_dataloader(self):
        train_ds = TrainDataset(
            self.data_config["train_excel"], 
            self.data_config["image_header"], 
            self.data_config["mask_header"],
            self.data_config["dmap_header"],
            self.data_config["n_patches_per_image"], 
            self.data_config["batch_size"], 
            self.data_config["patch_size"], 
            self.data_config["selected_labels"], 
            self.data_config["inten_norm"], 
            fg_ratio=self.data_config["fg_ratio"], 
            augment=True)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=1, collate_fn=collate_fn, drop_last=False)
        return train_loader

    def val_dataloader(self):
        val_ds = TrainDataset(
            self.data_config["val_excel"], 
            self.data_config["image_header"], 
            self.data_config["mask_header"], 
            self.data_config["dmap_header"],
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
        test_ds = TestDataset(
            self.test_config["test_excel"], 
            self.test_config["image_header"], 
            self.test_config["mask_header"], 
            self.test_config["selected_labels"], 
            self.test_config["patch_size"], 
            self.test_config["pixval_replacement"], 
            self.test_config["new_spacing"], 
            self.test_config["normalization"])
        return DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, drop_last=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.training_config["lr"])

        lr_scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=self.training_config["plateau_patience"]),
            "reduce_on_plateau": True,
            "monitor": "val_loss"
        }

        return ([optimizer], [lr_scheduler])

    def forward(self, x):
        return self.render_model(x)

    def training_step(self, batch, batchidx):
        data, target, dmap, cmap = batch

        seg_logits, seg_features = self.segment_model(data)

        with torch.no_grad():
            point_coords = sample_uncertain_points(seg_logits, dmap, self.data_config["num_points"], self.data_config["oversample_factor"], self.data_config["importance_sample_ratio"])

        _point_sample = partial(point_sample, point_coords=point_coords, align_corners=False)

        local_features = torch.cat([_point_sample(input=seg_features[i]) for i in self.in_features], dim=1)

        position_encoding = _point_sample(cmap)

        local_logits = self.local_render(torch.cat([position_encoding, local_features], dim=1))

        global_features = F.adaptive_avg_pool2d(seg_features[0], 1)
        B, C, *_ = global_features.shape
        global_features = global_features.view(B, C, 1).expand(B, C, self.data_config["num_points"])

        global_logits = self.global_render(torch.cat([position_encoding, global_features], dim=1))

        point_logits = global_logits + local_logits

        point_targets = _point_sample(input=target.to(torch.float), mode="nearest").to(torch.long)
        point_targets = torch.argmax(point_targets, dim=1)

        loss = self.point_loss(point_logits, point_targets) + self.seg_loss(seg_logits, target)
        metrics = self.metric_fn(seg_logits, target)
        
        result = {"loss": loss}
        result.update({"%s_%d" % (str(self.metric_fn), i): metrics[i] for i in range(len(metrics))})
        
        self.log_dict(result, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batchidx):
        data, target, dmap, cmap = batch

        seg_logits, seg_features = self.segment_model(data)

        with torch.no_grad():
            point_coords = sample_uncertain_points(seg_logits, dmap, self.data_config["num_points"], self.data_config["oversample_factor"], self.data_config["importance_sample_ratio"])

        _point_sample = partial(point_sample, point_coords=point_coords, align_corners=False)
        
        local_features = torch.cat([_point_sample(input=seg_features[i]) for i in self.in_features], dim=1)

        position_encoding = _point_sample(cmap)

        local_logits = self.local_render(torch.cat([position_encoding, local_features], dim=1))

        global_features = F.adaptive_avg_pool2d(seg_features[0], 1)
        B, C, *_ = global_features.shape
        global_features = global_features.view(B, C, 1).expand(B, C, self.data_config["num_points"])

        global_logits = self.global_render(torch.cat([position_encoding, global_features], dim=1))

        point_logits = global_logits + local_logits

        point_targets = _point_sample(input=target.to(torch.float), mode="nearest").to(torch.long)
        point_targets = torch.argmax(point_targets, dim=1)

        loss = self.point_loss(point_logits, point_targets) + self.seg_loss(seg_logits, target)
        metrics = self.metric_fn(seg_logits, target)
        
        result = {"val_loss": loss}
        result.update({"val_%s_%d" % (str(self.metric_fn), i): metrics[i] for i in range(len(metrics))})
        
        self.log_dict(result, on_epoch=True)
        return loss

    def test_step(self, batch, batchidx):
        prediction = self.predict_on_batch(batch["sitk_image"], batch["sitk_raw_image"])

        overlap_metrics = measure_overlap(batch["mask"], prediction, self.test_config["overlap_metrics"], len(self.test_config["selected_labels"]))
        surface_metrics = measure_surface(batch["mask"], prediction, self.test_config["surface_metrics"], len(self.test_config["selected_labels"]), whd_2_hwd(batch["sitk_raw_image"].GetSpacing()))
        
        log = {}
        metric_logging(log, self.test_config["overlap_metrics"], self.test_config["mask_header"], self.test_config["selected_labels"], overlap_metrics)
        metric_logging(log, self.test_config["surface_metrics"], self.test_config["mask_header"], self.test_config["selected_labels"], surface_metrics)
        
        self.log_dict(log, on_step=True, on_epoch=True)

        return overlap_metrics, surface_metrics

    def test_epoch_end(self, outputs):
        outputs = np.asarray(outputs)
        outputs = np.moveaxis(outputs, 0, -1)
        all_overlap_metrics, all_surface_metrics = outputs

        log = {}
        overlap_ci_names = ["ci_" + m for m in self.test_config["overlap_metrics"]]
        surface_ci_names = ["ci_" + m for m in self.test_config["surface_metrics"]]
        
        mask_overlap_intervals = [[bootstrap_confidence_interval(m) for m in label_metrics] for label_metrics in all_overlap_metrics]
        metric_logging(log, overlap_ci_names, self.test_config["mask_header"], self.test_config["selected_labels"], mask_overlap_intervals)

        mask_surface_intervals = [[bootstrap_confidence_interval(m) for m in label_metrics] for label_metrics in all_surface_metrics]
        metric_logging(log, surface_ci_names, self.test_config["mask_header"], self.test_config["selected_labels"], mask_surface_intervals)
    
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

        all_patch_logits = torch.cat([self.segment_model(image_batch)[0] for image_batch, _ in patch_gen], axis=0)

        reconstruction = self.restore_patch(all_patch_logits, raw_patch_size=raw_patch_size, raw_patch_indices=raw_patch_indices, raw_size=raw_size)
        
        prediction = postprocess(reconstruction, len(self.test_config["selected_labels"]), self.test_config["n_largest_components"])
        return prediction

    def restore_patch(self, patch_logits, raw_patch_size, raw_patch_indices, raw_size):
        patch_logits = F.interpolate(patch_logits, size=raw_patch_size.tolist(), mode='bilinear')
        patch_logits = torch.movedim(patch_logits, 1, -1).detach().cpu().numpy()
        reconstruction = reconstruct_from_patches(patch_logits, raw_patch_indices, raw_size)
        reconstruction = F.softmax(torch.from_numpy(reconstruction), dim=-1).numpy()
        return reconstruction

    # def test_step(self, batch, batchidx):
    #     predictions = self.test_on_patch(batch["image"], batch["patch_indices"], batch["org_size"], batch["fine_patch_size"])

    #     # plt.imshow(predictions[0][1])
    #     # plt.show()
    #     # np.save(os.path.join(self.test_config["result_dir"], Path(batch["path"]).stem), predictions)
    
    # def render(self, patch_logits, patch_features, cmap_batch, n_samples):
    #     uncertainty_map = self.calc_umap(patch_logits)
    #     point_indices, point_coords = get_uncertain_point_coords_on_grid(uncertainty_map, n_samples)

    #     _point_sample = partial(point_sample, point_coords=point_coords, align_corners=False)


    #     B, C, H, W = patch_logits.shape
    #     unfold_logits = F.unfold(patch_logits, kernel_size=3, dilation=1, stride=1, padding=1)
    #     unfold_logits = unfold_logits.view(B, C * 3 * 3, H, W)

    #     raw_logits = _point_sample(unfold_logits)
    #     local_features = [_point_sample(input=patch_features[i]) for i in self.in_features]
    #     local_features = torch.cat([raw_logits] + local_features, dim=1)

    #     point_coords_ = _point_sample(cmap_batch)
    #     point_features = self.point_render(point_coords_)

    #     local_logits = self.local_render(torch.cat([point_features, local_features], dim=1))

    #     global_features = F.adaptive_avg_pool2d(patch_features[0], 1)
    #     B, C, *_ = global_features.shape
    #     global_features = global_features.view(B, C, 1).expand(B, C, n_samples)

    #     global_logits = self.global_render(torch.cat([point_features, global_features], dim=1))

    #     point_logits = self.sdf(torch.cat([global_logits, local_logits], dim=1))
    #     return point_logits, point_indices

    # def calc_umap(self, patch_logits):
    #     ###################
    #     # dmaps = []
    #     # for i in range(patch_logits.shape[0]):
    #     #     patch_preds = torch.argmax(patch_logits[i], 0)
    #     #     dmap = np.zeros((patch_logits.shape[2], patch_logits.shape[3]))
    #     #     for i in range(patch_logits.shape[1]):
    #     #         dmap -= distance_transform_edt(patch_preds.detach().cpu().numpy() == i)
    #     #     dmaps.append(dmap)
    #     # dmaps = np.stack(dmaps, axis=0)

    #     # uncertainty_map = torch.from_numpy(dmaps).to(patch_logits.device)
    #     ###################
    #     uncertainty_map = calculate_uncertainty(patch_logits)
    #     ###################
    #     return uncertainty_map

    # def test_on_patch(self, image, patch_indices, org_size, fine_patch_size):
    #     fine_patch_indices = sample_patch_by_sliwin(np.array(org_size), fine_patch_size, fine_patch_size // 2)

    #     patch_gen = patch_generator(image, patch_indices, self.test_config["patch_size"], self.test_config["batch_size"], self.device)

    #     all_patch_logits = []
    #     for image_batch, cmap_batch in patch_gen:
    #         patch_logits, patch_features = self.segment_model(image_batch)

    
    #         point_logits, point_indices = self.render(patch_logits, patch_features, cmap_batch, 112 * 112)

    #         N, C, H, W = patch_logits.shape
    #         rendered_pos = torch.ones(N, H, W, device=patch_logits.device)

    #         rendered_pos = rendered_pos.reshape(N, H * W).scatter(1, point_indices, 2).view(N, 1, H, W)

    #         pred = F.softmax(patch_logits, dim=1).detach().cpu().numpy()
    #         pos = rendered_pos.detach().cpu().numpy()
    #         plt.subplot(121)
    #         plt.imshow(pred[0, 1])
    #         plt.subplot(122)
    #         plt.imshow(pos[0, 0])
    #         plt.show()

    #         # rendered_logits = torch.zeros_like(patch_logits, device=patch_logits.device)
    #         # rendered_pos = torch.ones_like(patch_logits, device=patch_logits.device)
    #         # N, C, H, W = patch_logits.shape
    #         # point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    #         # rendered_logits = rendered_logits.reshape(N, C, H * W).scatter(2, point_indices, point_logits).view(N, C, H, W)
    #         # rendered_pos = rendered_pos.reshape(N, C, H * W).scatter(2, point_indices, 2).view(N, C, H, W)

    #         # patch_logits = (patch_logits + rendered_logits) / rendered_pos

    #         # point_logits, point_indices = self.render(patch_logits, patch_features, cmap_batch, 112 * 112)

    #         # rendered_logits = torch.zeros_like(patch_logits, device=patch_logits.device)
    #         # rendered_pos = torch.ones_like(patch_logits, device=patch_logits.device)
    #         # N, C, H, W = patch_logits.shape
    #         # point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    #         # rendered_logits = rendered_logits.reshape(N, C, H * W).scatter(2, point_indices, point_logits).view(N, C, H, W)
    #         # rendered_pos = rendered_pos.reshape(N, C, H * W).scatter(2, point_indices, 2).view(N, C, H, W)

    #         # patch_logits = (patch_logits + rendered_logits) / rendered_pos
    #         patch_logits = F.interpolate(patch_logits, size=fine_patch_size.tolist(), mode='bilinear')
    #         all_patch_logits.append(patch_logits)


       
    #     all_patch_logits = torch.cat(all_patch_logits, dim=0)

    #     restore_patch = partial(self.restore_patch, fine_patch_indices=fine_patch_indices, org_size=org_size)
    #     reconstruction = restore_patch(all_patch_logits)
    #     reconstruction = reconstruction[np.newaxis, ...]
    #     reconstruction = np.moveaxis(reconstruction, -1, 1)
    #     return reconstruction

    # def restore_patch(self, all_patch_logits, fine_patch_indices, org_size):
    #     patch_logits = torch.movedim(all_patch_logits, 1, -1).detach().cpu().numpy()
    #     reconstruction = reconstruct_from_patches(patch_logits, fine_patch_indices, org_size)
    #     reconstruction = F.softmax(torch.from_numpy(reconstruction), dim=-1).numpy()
    #     return reconstruction