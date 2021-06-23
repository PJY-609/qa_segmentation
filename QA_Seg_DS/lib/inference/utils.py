import torch.nn.functional as F
import torch
import SimpleITK as sitk
import os

def write_nii(image, dest_path):
    image = sitk.GetImageFromArray(image)
    sitk.WriteImage(image, dest_path)


def interpolate(x, size, mode="bilinear"):
	x = torch.from_numpy(x)
	x = torch.movedim(x, -1, 1)
	x = F.interpolate(x, size, mode=mode)
	x = torch.movedim(x, 1, -1)
	x = x.numpy()
	return x

def makedirs(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)