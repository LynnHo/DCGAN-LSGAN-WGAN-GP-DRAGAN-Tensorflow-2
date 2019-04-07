from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import scipy
import numpy as np
import tensorflow as tf

from collections import OrderedDict


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)


def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """ return a Session with simple config """

    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)


def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), \
        '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens


def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)


def summary(tensor_collection, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    """
    usage:

    1. summary(tensor)

    2. summary([tensor_a, tensor_b])

    3. summary({tensor_a: 'a', tensor_b: 'b})
    """

    def _summary(tensor, name, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
        """ Attach a lot of summaries to a Tensor. """

        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        with tf.name_scope('summary_' + name):
            summaries = []
            if len(tensor._shape) == 0:
                summaries.append(tf.summary.scalar(name, tensor))
            else:
                if 'mean' in summary_type:
                    mean = tf.reduce_mean(tensor)
                    summaries.append(tf.summary.scalar(name + '/mean', mean))
                if 'stddev' in summary_type:
                    mean = tf.reduce_mean(tensor)
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                    summaries.append(tf.summary.scalar(name + '/stddev', stddev))
                if 'max' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
                if 'min' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
                if 'sparsity' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
                if 'histogram' in summary_type:
                    summaries.append(tf.summary.histogram(name, tensor))
            return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]
    with tf.name_scope('summaries'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)


def counter(scope='counter'):
    with tf.variable_scope(scope):
        counter = tf.Variable(0, dtype=tf.int32, name='counter')
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt


def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False


def memory_data_batch(memory_data_dict, batch_size, preprocess_fns={}, shuffle=True, num_threads=16,
                      min_after_dequeue=5000, allow_smaller_final_batch=False, scope=None):
    """
    memory_data_dict:
        for example
        {'img': img_ndarray, 'point': point_ndarray} or
        {'img': img_tensor, 'point': point_tensor}
        the value of each item of `memory_data_dict` is in shape of (N, ...)

    preprocess_fns:
        for example
        {'img': img_preprocess_fn, 'point': point_preprocess_fn}
    """

    with tf.name_scope(scope, 'memory_data_batch'):
        fields = []
        tensor_dict = OrderedDict()
        for k in memory_data_dict:
            fields.append(k)
            tensor_dict[k] = tf.convert_to_tensor(memory_data_dict[k])  # the same dtype of the input data
        data_num = tensor_dict[k].get_shape().as_list()[0]

        # slice to single example, and since it's memory data, the `capacity` is set as data_num
        data_values = tf.train.slice_input_producer(list(tensor_dict.values()), shuffle=shuffle, capacity=data_num)
        data_keys = list(tensor_dict.keys())
        data_dict = {}
        for k, v in zip(data_keys, data_values):
            if k in preprocess_fns:
                data_dict[k] = preprocess_fns[k](v)
            else:
                data_dict[k] = v

        # batch datas
        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            data_batch = tf.train.shuffle_batch(data_dict,
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue,
                                                num_threads=num_threads,
                                                allow_smaller_final_batch=allow_smaller_final_batch)
        else:
            data_batch = tf.train.batch(data_dict,
                                        batch_size=batch_size,
                                        allow_smaller_final_batch=allow_smaller_final_batch)

        return data_batch, data_num, fields


class MemoryData:

    def __init__(self, memory_data_dict, batch_size, preprocess_fns={}, shuffle=True, num_threads=16,
                 min_after_dequeue=5000, allow_smaller_final_batch=False, scope=None):
        """
        memory_data_dict:
            for example
            {'img': img_ndarray, 'point': point_ndarray} or
            {'img': img_tensor, 'point': point_tensor}
            the value of each item of `memory_data_dict` is in shape of (N, ...)

        preprocess_fns:
            for example
            {'img': img_preprocess_fn, 'point': point_preprocess_fn}
        """

        self.graph = tf.Graph()  # declare ops in a separated graph
        with self.graph.as_default():
            # @TODO
            # There are some strange errors if the gpu device is the
            # same with the main graph, but cpu device is ok. I don't know why...
            with tf.device('/cpu:0'):
                self._batch_ops, self._data_num, self._fields = memory_data_batch(memory_data_dict, batch_size, preprocess_fns, shuffle, num_threads,
                                                                                  min_after_dequeue, allow_smaller_final_batch, scope)

        print(' [*] MemoryData: create session!')
        self.sess = session(graph=self.graph)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def __len__(self):
        return self._data_num

    def batch(self, fields=None):
        batch_data = self.sess.run(self._batch_ops)
        if fields is None:
            fields = self._fields
        if isinstance(fields, (list, tuple)):
            return [batch_data[field] for field in fields]
        else:
            return batch_data[fields]

    def fields(self):
        return self._fields

    def __del__(self):
        print(' [*] MemoryData: stop threads and close session!')
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


def disk_image_batch(image_paths, batch_size, shape, preprocess_fn=None, shuffle=True, num_threads=16,
                     min_after_dequeue=100, allow_smaller_final_batch=False, scope=None):
    """
    This function is suitable for bmp, jpg, png and gif files

    image_paths: string list or 1-D tensor, each of which is an iamge path
    preprocess_fn: single image preprocessing function
    """

    with tf.name_scope(scope, 'disk_image_batch'):
        data_num = len(image_paths)

        # dequeue a single image path and read the image bytes; enqueue the whole file list
        _, img = tf.WholeFileReader().read(tf.train.string_input_producer(image_paths, shuffle=shuffle, capacity=data_num))
        img = tf.image.decode_image(img)

        # preprocessing
        img.set_shape(shape)
        if preprocess_fn is not None:
            img = preprocess_fn(img)

        # batch datas
        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            img_batch = tf.train.shuffle_batch([img],
                                               batch_size=batch_size,
                                               capacity=capacity,
                                               min_after_dequeue=min_after_dequeue,
                                               num_threads=num_threads,
                                               allow_smaller_final_batch=allow_smaller_final_batch)
        else:
            img_batch = tf.train.batch([img],
                                       batch_size=batch_size,
                                       allow_smaller_final_batch=allow_smaller_final_batch)

        return img_batch, data_num


class DiskImageData:

    def __init__(self, image_paths, batch_size, shape, preprocess_fn=None, shuffle=True, num_threads=16,
                 min_after_dequeue=100, allow_smaller_final_batch=False, scope=None):
        """
        This function is suitable for bmp, jpg, png and gif files

        image_paths: string list or 1-D tensor, each of which is an iamge path
        preprocess_fn: single image preprocessing function
        """

        self.graph = tf.Graph()  # declare ops in a separated graph
        with self.graph.as_default():
            # @TODO
            # There are some strange errors if the gpu device is the
            # same with the main graph, but cpu device is ok. I don't know why...
            with tf.device('/cpu:0'):
                self._batch_ops, self._data_num = disk_image_batch(image_paths, batch_size, shape, preprocess_fn, shuffle, num_threads,
                                                                   min_after_dequeue, allow_smaller_final_batch, scope)

        print(' [*] DiskImageData: create session!')
        self.sess = session(graph=self.graph)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def __len__(self):
        return self._data_num

    def batch(self):
        return self.sess.run(self._batch_ops)

    def __del__(self):
        print(' [*] DiskImageData: stop threads and close session!')
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def imwrite(image, path):
    """ save an [-1.0, 1.0] image """

    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))


def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """

    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img
