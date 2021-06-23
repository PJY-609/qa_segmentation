import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import math

from lib.utils import read_image, read_mask, intensity_normalization
from lib.patch import extract_patches
from lib.training.augment import augment

class TrainDataset(Dataset):
    def __init__(self, excel_path, image_header, mask_header, n_patches_per_image, batch_size, patch_size, selected_labels, normalization, augment, fg_ratio=0.33):
        df = pd.read_excel(excel_path, engine="openpyxl")
        self.image_paths = df[image_header].values
        self.mask_paths = df[mask_header].values
        self.n_patches_per_image = n_patches_per_image
        self.batch_size = batch_size
        self.patch_size = np.array(patch_size)
        self.selected_labels = selected_labels
        self.normalization = normalization
        self.augment = augment
        self.fg_ratio = fg_ratio

    def __len__(self):
        n_images = len(self.image_paths)
        steps_per_image = math.ceil(self.n_patches_per_image / self.batch_size)
        steps_per_epoch = n_images * steps_per_image
        return steps_per_epoch

    def __getitem__(self, index):
        steps_per_image = math.ceil(self.n_patches_per_image / self.batch_size)
        data_idx = math.floor(index / steps_per_image)

        image = read_image(self.image_paths[data_idx])
        mask = read_mask(self.mask_paths[data_idx], self.selected_labels)

        images = intensity_normalization(image, self.normalization)

        sampled_pts = np.repeat([(image.ndim - 1) * [0]], self.batch_size, axis=0)
        if np.less(self.patch_size, image.shape[:-1]).all():
            sampled_pts = self.sample_points(mask)

        image_batch = extract_patches(image, sampled_pts, self.patch_size)
        mask_batch = extract_patches(mask, sampled_pts, self.patch_size)
        
        if self.augment:
            image_batch, mask_batch = augment(image_batch, mask_batch)

        # one hot
        mask_batch = np.squeeze(mask_batch, -1)
        mask_batch = np.eye(len(self.selected_labels) + 1)[mask_batch].astype(np.int)  

        # channel first
        image_batch = np.moveaxis(image_batch, -1, 1)
        mask_batch = np.moveaxis(mask_batch, -1, 1) 

        image_batch = torch.from_numpy(image_batch).float()
        mask_batch = torch.from_numpy(mask_batch).long()
        return image_batch, mask_batch

    def sample_points(self, mask):
        selected_label = np.random.choice(len(self.selected_labels)) + 1
        sampled_mask = mask == selected_label
        sampled_mask = np.squeeze(sampled_mask, -1)

        n_fg_pts = 1 if self.batch_size == 2 else int(self.batch_size * self.fg_ratio)
        n_rd_pts = self.batch_size - n_fg_pts

        all_fg_pts = np.argwhere(sampled_mask == 1)
        sampled_fg_pts = all_fg_pts[np.random.choice(len(all_fg_pts), size=n_fg_pts)]

        # randomly sample pixels
        sampled_rd_pts = [np.random.randint(s[1], s[0] - s[1], size=n_rd_pts) for s in zip(sampled_mask.shape, self.patch_size / 2)]
        sampled_rd_pts = np.array(sampled_rd_pts).T

        sampled_pts = np.concatenate([sampled_fg_pts, sampled_rd_pts])
        np.random.shuffle(sampled_pts)
        
        # top left corner
        sampled_pts = np.round(sampled_pts - (self.patch_size / 2)).astype(np.int) 
        return sampled_pts
