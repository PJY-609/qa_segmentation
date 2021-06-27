import numpy as np
import matplotlib.pyplot as plt

def sample_patch_by_sliwin(image_size, patch_size, overlap):
    margins = np.subtract(patch_size, image_size)
    if (margins > 0).all():
        return np.unsqueeze(np.zeros_like(patch_size) - margins // 2, axis=0)

    step = patch_size - overlap
    start = np.zeros_like(patch_size)
    stop = image_size - step

    patch_indices = np.mgrid[tuple(slice(start[i], stop[i], step[i]) for i in range(len(image_size)))]
    patch_indices = patch_indices.reshape(len(image_size), -1).T
    return patch_indices


def pad_image(data, patch_indices, patch_size, pad_mode):
    data_size = data.shape[:-1] # no channel axis

    fixed_indices = patch_indices

    is_underflow = patch_indices < 0
    is_overflow = (patch_indices + patch_size) > (data_size * np.ones_like(patch_indices))

    if np.any(is_underflow) or np.any(is_overflow):
        underflow = is_underflow * np.abs(patch_indices)
        overflow = is_overflow * np.abs(patch_indices + patch_size - data_size)
        
        lower_padding = np.max(underflow, 0)
        upper_padding = np.max(overflow, 0)
        padding = np.stack([lower_padding, upper_padding], axis=1)
        padding = padding.tolist() + [[0, 0]]
        
        data = np.pad(data, padding, pad_mode)
        fixed_indices = patch_indices + lower_padding
    return data, fixed_indices


def extract_patches(data, patch_indices, patch_size, pad_mode="symmetric"):
    data_ndim = data.ndim - 1 # no channel axis

    data, fixed_indices = pad_image(data, patch_indices, patch_size, pad_mode)

    patches = []
    for idx in fixed_indices:
        patch = data[tuple(slice(idx[i], idx[i] + patch_size[i]) for i in range(data_ndim))]
        patches.append(patch)
    patches = np.asarray(patches)
    return patches


def trim_patch(patch, patch_index, output_size):
    patch_size = patch.shape[:-1]

    is_underflow = patch_index < 0
    is_overflow = (patch_index + patch_size) >= output_size

    if np.any(is_underflow) or np.any(is_overflow):
        lower_bound = np.asarray(is_underflow * np.abs(patch_index), dtype=np.int)
        overflow = (patch_index + patch_size) - output_size
        upper_bound = np.asarray(patch_size - (is_overflow * overflow), dtype=np.int)
        
        patch = patch[tuple(slice(l, u) for l, u in zip(lower_bound, upper_bound))]
        
    patch_index[is_underflow] = 0
    return patch, patch_index


def reconstruct_from_patches(patches, patch_indices, output_size):
    size = output_size + (patches.shape[-1],) # plus channel axis
    reconstruction, count = np.zeros(size), np.zeros(size)

    output_size = np.asarray(output_size).astype(np.int)
    patch_indices = patch_indices.astype(np.int)

    for patch, index in zip(patches, patch_indices):
        patch, index = trim_patch(patch, index, output_size)
        
        PATCH = tuple(slice(index[i], index[i] + patch.shape[i]) for i in range(len(output_size)))
        reconstruction[PATCH] += patch
        count[PATCH] += 1
        
    reconstruction /= (count + 1e-8) 
    return reconstruction
