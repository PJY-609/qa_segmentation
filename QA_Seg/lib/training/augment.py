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
           iaa.Affine(rotate=(-180, 180), mode="symmetric"),
           iaa.Affine(shear=(-25, 25), mode="symmetric")
        ], random_order=True)


def augment(image_batches, mask_batches, image_augments):
    geo_seq = geo_augs().to_deterministic()
    val_seq = val_augs().to_deterministic()
    
    image_aug_batches = []
    for image_batch, augment in zip(image_batches, image_augments):
      image_aug_batch = image_batch.astype(np.float16)
      if "geo" in augment:
        image_aug_batch = geo_seq(images=image_aug_batch)
      elif "val" in augment:
        image_aug_batch = val_seq(images=image_aug_batch)
      image_aug_batches.append(image_aug_batch)

    augment_mask = partial(geo_seq, images=image_batch)
    mask_aug_batches = [augment_mask(segmentation_maps=m)[1] for m in mask_batches]

    return image_aug_batches, mask_aug_batches

    
