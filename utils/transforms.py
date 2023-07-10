import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    train_transforms = A.Compose(
        [
            A.SmallestMaxSize(max_size=512),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                               rotate_limit=360, p=0.5),
            A.RandomCrop(height=512, width=512),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                       b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.MultiplicativeNoise(
                multiplier=[0.5, 2], per_channel=True, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HueSaturationValue(hue_shift_limit=0.2,
                                 sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.SmallestMaxSize(max_size=512),
            A.CenterCrop(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transforms, test_transforms
