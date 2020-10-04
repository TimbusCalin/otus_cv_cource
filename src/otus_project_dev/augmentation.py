import albumentations as albu
from albumentations.pytorch import ToTensor


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def post_transforms(CLASSES):
    return [
        albu.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
        ToTensor(num_classes=len(CLASSES))
    ]


def hard_transforms():
    result = [
        albu.Cutout(max_h_size=15, max_w_size=25, p=0.5),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
    ]
    return result


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result
