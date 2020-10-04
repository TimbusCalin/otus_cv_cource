from matplotlib import pyplot as plt
import numpy as np


def draw_sample(image, mask, classes, is_tensor=False):

    gray_mask = []
    if is_tensor:
        mask = mask.permute(1, 2, 0)
        image = image.permute(1, 2, 0)

    for line in mask:
        for pix in line:
            gray_mask.append(classes[np.argmax(pix)])

    gray_mask = np.reshape(gray_mask, (mask.shape[0], mask.shape[1]))

    # print("classes: {}".format(np.unique(gray_mask)))

    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image")

    plt.subplot(1, 2, 2)
    # plt.imshow(gray_mask, cmap='gray', vmin=0, vmax=255)
    plt.imshow(gray_mask)
    plt.title(f"Mask")
