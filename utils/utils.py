import os
import re
from typing import Tuple, Mapping, Dict, Hashable

import monai
import numpy as np
import torch
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import get_track_meta
from monai.transforms import MapTransform, RandomizableTransform, InvertibleTransform
from monai.utils import GridSamplePadMode, convert_to_tensor

import interpol
from interpol import grid_pull
from interpol.api import affine_grid
import nibabel as nib


def save_img(image, affine, filename):
    nifti_img = nib.Nifti1Image(image, affine)
    # if os.path.exists(filename):
    #     raise OSError("File already exists! Killing job")
    # else:
    nib.save(nifti_img, filename)


class Jeff(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma: randomly select sigma value.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        prob: probability of Gaussian smooth.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = monai.transforms.RandGaussianSmooth.backend

    def __init__(
        self,
        keys: KeysCollection,
        target_shape=(512, 512),
        prob: float = 0.1,
        blur_factor_min: float = 1.5,
        blur_factor_max: float = 4,
        do_blur: bool = False,
        allow_missing_keys: bool = False,
        device=None,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.do_blur = do_blur
        self.device = device
        self.target_shape = target_shape
        self.blur_factor_min = blur_factor_min
        self.blur_factor_max = blur_factor_max

    def gaussian_blur(self, img: NdarrayOrTensor):
        # Pick uniformly sampled multiplicative factor between 0.8 and 1.0
        alpha_factor = np.random.uniform(low=0.8, high=1.0)
        # Pick uniformly sampled multiplicative factor between blur_factor_min and blur_factor_max
        blur_factor = np.random.uniform(low=self.blur_factor_min, high=self.blur_factor_max)
        resampling_adjustment_factor = 0  # np.random.uniform(low=0.5, high=1.0) * 2 * np.log(10) / (2 * np.pi)
        sigma_blur = (
            2 * np.log(10) * alpha_factor * blur_factor / (2 * np.pi)
            - resampling_adjustment_factor
        )
        # Intelligent way of removing channel
        sigma_sequence = np.array([0] * len(img[0, ...].shape)) + sigma_blur

        gs = monai.transforms.GaussianSmooth(sigma=sigma_sequence.tolist())
        img = gs(img)

        return img, sigma_blur

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        for key in self.key_iterator(d):
            # Log original shape and qform
            d[key].meta["original_shape"] = d[key].meta["spatial_shape"]
            # # Check for anisotropy and assign values
            if self.do_blur:
                # Blur + resize
                d[key], sigma_blur = self.gaussian_blur(d[key])
                d[key].meta["sigma_blur"] = sigma_blur

        return d


class CustomResized(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma: randomly select sigma value.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        prob: probability of Gaussian smooth.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = monai.transforms.RandGaussianSmooth.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Tuple[float, float] = (0.25, 1.5),
        target_shape=(512, 512),
        approx: str = "erf",
        prob: float = 0.1,
        mode: int = 1,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        allow_missing_keys: bool = False,
        device=None,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_smooth = monai.transforms.RandGaussianSmooth(
            sigma_x=sigma, sigma_y=sigma, sigma_z=sigma, approx=approx, prob=1.0
        )
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        self.target_shape = target_shape

    def resize(
        self, img: NdarrayOrTensor,
    ) -> (NdarrayOrTensor, NdarrayOrTensor):
        # TODO Fix this: Check if key or data structure is best!
        img = convert_to_tensor(img, track_meta=get_track_meta())

        # Resize
        img.data = interpol.resize(img.data[0, ...], shape=self.target_shape, bound=self.padding_mode,
                                   interpolation=1)[None, ...]

        return img

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma
        # self.rand_smooth.randomize(None)
        for key in self.key_iterator(d):
            # Log original shape and qform
            d[key].meta["original_shape"] = d[key].meta["spatial_shape"]
            # Resize
            d[key] = self.resize(d[key])

        return d
