import cv2
import numpy as np
import collections

from catalyst import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SegmentationDataset(Dataset):
    def __init__(
        self,
        classes,
        images,
        masks=None,
        transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.n_classes = len(classes)
        self.classes = classes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)
        result = {"image": image}

        if self.masks is not None:
            mask = utils.imread(str(self.masks[idx]))
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            one_hot = np.zeros((mask.shape[0], mask.shape[1], self.n_classes))
            for i, unique_value in enumerate(self.classes):
                one_hot[:, :, i][mask == unique_value] = 1
            result["mask"] = one_hot

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result


def get_loaders(
    images,
    masks,
    classes,
    random_state,
    valid_size=0.2,
    batch_size=32,
    num_workers=4,
    train_transforms_fn = None,
    valid_transforms_fn = None,):

    indices = np.arange(len(images))

    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
      indices, test_size=valid_size, random_state=random_state, shuffle=True
    )

    np_images = np.array(images)
    np_masks = np.array(masks)

    # Creates our train dataset
    train_dataset = SegmentationDataset(
      classes = classes,
      images = np_images[train_indices].tolist(),
      masks = np_masks[train_indices].tolist(),
      transforms = train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
      classes = classes,
      images = np_images[valid_indices].tolist(),
      masks = np_masks[valid_indices].tolist(),
      transforms = valid_transforms_fn
    )

    # Catalyst uses torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )

    valid_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      drop_last=True,
    )

    # And expect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders