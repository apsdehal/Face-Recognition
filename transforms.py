from skimage.transform import warp, AffineTransform
from torchvision import transforms
from constants import IMG_SIZE
from PIL import Image

import numpy as np


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        img_data = np.array(img)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation,
                             translation=translation)
        img_data1 = warp(img_data, af.inverse)
        img1 = Image.fromarray(np.uint8(img_data1 * 255))
        return img1


train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.6, 0.6, 0.6, 0.2),
    RandomAffineTransform(rotation_range=(-0.2, 0.2),
                          translation_range=(-0.1, 0.1),
                          scale_range=(0.8, 1.2),
                          shear_range=(-0.1, 0.1)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
