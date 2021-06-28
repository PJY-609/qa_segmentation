import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import math
import SimpleITK as sitk

from lib.utils import read_image, read_mask, intensity_normalization
from lib.utils import get_info_reader, replace_pixval, resample
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
    def __init__(self, excel_path, image_header, mask_header, selected_labels, patch_size, pixval_replacement, new_spacing, normalization):
        df = pd.read_excel(excel_path, engine="openpyxl")
        self.image_paths = df[image_header].values
        self.mask_paths = df[mask_header].values
        self.selected_labels = selected_labels
        self.patch_size = np.array(patch_size)
        self.pixval_replacement = pixval_replacement
        self.new_spacing = new_spacing
        self.normalization = normalization

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        raw_image = sitk.ReadImage(self.image_paths[index])
        
        if len(raw_image.GetSize()) == 2:
            raw_image = sitk.JoinSeries(raw_image)

        image = sitk.GetArrayFromImage(raw_image)
        image = replace_pixval(image, self.pixval_replacement)
        image = intensity_normalization(image, self.normalization)

        image = sitk.GetImageFromArray(image)
        image.CopyInformation(raw_image)
        
        image = resample(image, self.new_spacing, sitk.sitkBSpline)

        mask = read_mask(self.mask_paths[index], self.selected_labels)

        ret = {
            "sitk_image": image,
            "sitk_raw_image": raw_image,
            "mask": mask,
            "path": self.image_paths[index]
        }
        return ret

