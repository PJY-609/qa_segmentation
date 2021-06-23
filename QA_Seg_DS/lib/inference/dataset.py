import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import math

from lib.utils import read_image, read_mask, intensity_normalization, get_info_reader
from lib.patch import extract_patches, sample_patch_by_sliwin

def patch_generator(image, patch_indices, patch_size, batch_size, device):
    steps = math.ceil(len(patch_indices) / batch_size)
    for i in range(steps):
        patch_idxs = patch_indices[i * batch_size:(i + 1) * batch_size]
        image_batch = extract_patches(image, patch_idxs, patch_size, "symmetric")
        image_batch = np.moveaxis(image_batch, -1, 1)
        image_batch = torch.from_numpy(image_batch).to(device=device, dtype=torch.float)
        yield image_batch


class TestDataset(Dataset):
    def __init__(self, excel_path, patch_size, normalization):
        df = pd.read_excel(excel_path, engine="openpyxl")
        self.image_paths = df["image"].values
        self.org_image_paths = df["org_image"].values
        self.patch_size = np.array(patch_size)
        self.patch_overlap = self.patch_size // 2
        self.normalization = normalization

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = read_image(self.image_paths[index])
        image = intensity_normalization(image, self.normalization)
        
        patch_indices = sample_patch_by_sliwin(image.shape[:-1], self.patch_size, self.patch_overlap)
        margins = self.patch_size - np.array(image.shape[:-1])
        if (margins > 0).all():
            patch_indices = np.unsqueeze(np.zeros_like(self.patch_size) - margins // 2, axis=0)

        org_reader = get_info_reader(self.org_image_paths[index])
        new_reader = get_info_reader(self.image_paths[index])

        org_size = (org_reader.GetSize()[1], org_reader.GetSize()[0])
        org_spacing = (org_reader.GetSpacing()[1], org_reader.GetSpacing()[0])
        new_spacing = (new_reader.GetSpacing()[1], new_reader.GetSpacing()[0])

        fine_patch_size = np.ceil(np.multiply(self.patch_size, np.divide(new_spacing, org_spacing))).astype(np.int32)

        ret = {
            "image": image,
            "patch_indices": patch_indices,
            "path": self.image_paths[index],
            "org_path": self.org_image_paths[index],
            "org_size": org_size,
            "fine_patch_size": fine_patch_size
        }
        return ret
