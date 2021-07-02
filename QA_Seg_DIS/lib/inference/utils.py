import torch.nn.functional as F
import torch
import numpy as np
import os
import cv2
import heapq

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


def select_n_largest_components(pred, n):
	_, labels, stats, centroids = cv2.connectedComponentsWithStats(pred.astype(np.uint8))
	areas = stats[1:, cv2.CC_STAT_AREA]

	n_largest_idxs = heapq.nlargest(n, range(len(areas)), key=lambda x: areas[x])
	n_largest_idxs = np.asarray(n_largest_idxs) + 1
	mask = np.isin(labels, n_largest_idxs)

	pred = np.zeros_like(labels)
	pred[mask] = 1
	return pred

def postprocess(prediction, n_labels, n_largest):
	prediction = np.argmax(prediction, axis=-1)

	result = np.zeros_like(prediction)
	for i in range(n_labels):
		label = prediction == (i + 1)
		label = select_n_largest_components(label, n_largest[i])
		result += label

	result = result[..., np.newaxis]
	return result