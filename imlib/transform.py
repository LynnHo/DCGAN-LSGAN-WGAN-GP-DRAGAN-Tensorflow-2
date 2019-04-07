import numpy as np
import skimage.color as color
import skimage.transform as transform


rgb2gray = color.rgb2gray
gray2rgb = color.gray2rgb

imresize = transform.resize
imrescale = transform.rescale


def immerge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    """Merge images to an image with (n_rows * h) * (n_cols * w).

    Parameters
    ----------
    images : numpy.array or object which can be converted to numpy.array
        Images in shape of N * H * W(* C=1 or 3).

    """
    images = np.array(images)
    n = images.shape[0]
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1),
             w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img
