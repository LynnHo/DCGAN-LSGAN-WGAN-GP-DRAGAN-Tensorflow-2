import functools
import random

import tensorflow as tf


def center_crop(image, size):
    # for image of shape [batch, height, width, channels] or [height, width, channels]
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    offset_height = (tf.shape(image)[-3] - size[0]) // 2
    offset_width = (tf.shape(image)[-2] - size[1]) // 2
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, size[0], size[1])


def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
    """Color jitter.

    Examples
    --------
    >>> color_jitter(img, 25, 0.2, 0.2, 0.1)

    """
    tforms = []
    if brightness > 0:
        tforms.append(functools.partial(tf.image.random_brightness, max_delta=brightness))
    if contrast > 0:
        tforms.append(functools.partial(tf.image.random_contrast, lower=max(0, 1 - contrast), upper=1 + contrast))
    if saturation > 0:
        tforms.append(functools.partial(tf.image.random_saturation, lower=max(0, 1 - saturation), upper=1 + saturation))
    if hue > 0:
        tforms.append(functools.partial(tf.image.random_hue, max_delta=hue))

    random.shuffle(tforms)
    for tform in tforms:
        image = tform(image)

    return image


def random_grayscale(image, p=0.1):
    return tf.cond(pred=tf.random.uniform(shape=(), dtype=tf.float32) < p,
                   true_fn=lambda: tf.image.adjust_saturation(image, 0),
                   false_fn=lambda: image)
