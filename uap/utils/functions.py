import numpy as np
import skimage.io as sio
from scipy.misc import imread
def img_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = sio.imread(img_path)
    img = np.resize(img, (size, size))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img

