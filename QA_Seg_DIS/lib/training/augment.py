import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from functools import partial

def val_augs():
    return iaa.SomeOf((0, 1),
        [
            iaa.OneOf([
                iaa.Multiply((0.8, 1.2)),
                iaa.Add((-0.15, 0.15))]),
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.1))]),
            iaa.OneOf([
                iaa.GaussianBlur((0.0, 3.0)),
                iaa.AverageBlur(k=(3, 7))])
        ], random_order=True)

def geo_augs():        
    return iaa.SomeOf((0, 4),
        [   
           iaa.Crop(percent=(0.0, 0.2)),
           iaa.Fliplr(0.5),
           iaa.Flipud(0.5),
           iaa.Affine(scale=(0.8, 2.0), mode="symmetric"),
           iaa.Affine(translate_percent=(-0.4, 0.4), mode="symmetric"),
           # iaa.Affine(rotate=(-180, 180), mode="symmetric"),
           # iaa.Affine(shear=(-25, 25), mode="symmetric")
        ], random_order=True)


def augment(image_batch, mask_batch, dmap_batch, cmap_batch):
    geo_seq = geo_augs().to_deterministic()
    val_seq = val_augs().to_deterministic()
    
    image_aug_batch = geo_seq(images=image_batch)
    image_aug_batch = val_seq(images=image_aug_batch)

    geo_augment = partial(geo_seq, images=image_batch)
    _, mask_aug_batch, cmap_aug_batch = geo_augment(segmentation_maps=mask_batch, heatmaps=cmap_batch)
    _, dmap_aug_batch = geo_augment(heatmaps=dmap_batch)

    return image_aug_batch, mask_aug_batch, dmap_aug_batch, cmap_aug_batch

    
