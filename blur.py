# Blurs parts of images
from PIL import Image, ImageFilter
import numpy as np


def blurimage(img, mask):
    # TODO: Horribly inefficient, update to be less aweful (masked arrays?)
    fullblur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=3)))
    img = np.asarray(img)
    partialblur = np.zeros(img.shape)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y]:
                partialblur[x][y] = img[x][y]
            else:
                partialblur[x][y] = fullblur[x][y]
    return Image.fromarray(np.uint8(partialblur))
