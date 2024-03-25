import monai
import numpy as np
from PIL import Image
from monai.transforms import (Compose,
                              LoadImaged,
                              ToDeviced,
                              SpatialCropd,
                              RandSpatialCropSamplesd, SqueezeDimd, AddChanneld, ScaleIntensityd, CopyItemsd
                              )
from utils.utils import RandBlurd, CustomResized
import os

test_transforms = Compose([
    LoadImaged(keys=["image"]),
    ToDeviced(keys=["image"], device="cuda:0"),
    AddChanneld(keys=["image"]),
    SpatialCropd(keys=["image"], roi_start=(50, 0, 0), roi_end=(160, 23423, 23434)),
    RandSpatialCropSamplesd(keys=["image"], roi_size=(1, -1, -1), num_samples=1,
                            random_center=True,
                            random_size=False),
    SqueezeDimd(keys=["image"], dim=0),
    CustomResized(keys=["image"], target_shape=(512, 512)),
    CopyItemsd(keys=["image"], times=1),
    RandBlurd(keys=["image"], blur_factor=2, do_blur=True, prob=1.0),
    ScaleIntensityd(
        keys=["image", "image_0"], minv=0.0, maxv=1.0)
    ]
)

test_image_dict = [
    {"image": "/data/Data-ProBono/normal.nii.gz"},
    {"image": "/data/Data-ProBono/subsamp2.nii.gz"},
    {"image": "/data/Data-ProBono/512.nii.gz"},

]

# Dataloader
training_dataset = monai.data.Dataset(
    data=test_image_dict,
    transform=test_transforms,
)

train_loader = monai.data.DataLoader(
    training_dataset,
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
)

for batch in train_loader:
    print(batch["image"].shape)

    # Print min and max
    print(batch["image"].min(), batch["image"].max())

    # Save image
    Image.fromarray((batch["image"].squeeze().cpu().numpy() * 255).astype(np.uint8)).save(f"/data/Data-ProBono/{os.path.basename(batch['image_meta_dict']['filename_or_obj'][0]).split('.')[0]}_slice_blur.png")
    Image.fromarray((batch["image_0"].squeeze().cpu().numpy() * 255).astype(np.uint8)).save(f"/data/Data-ProBono/{os.path.basename(batch['image_meta_dict']['filename_or_obj'][0]).split('.')[0]}_slice_original.png")
