from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import gzip
import struct
import subprocess
import numpy as np


def unzip_gz(file_name):
    unzip_name = file_name.replace('.gz', '')
    gz_file = gzip.GzipFile(file_name)
    open(unzip_name, 'w+').write(gz_file.read())
    gz_file.close()


def mnist_load(data_dir, dataset='train'):
    """
    modified from https://gist.github.com/akesling/5358964

    return:
    1. [-1.0, 1.0] float64 images of shape (N * H * W)
    2. int labels of shape (N,)
    3. # of datas
    """

    if dataset is 'train':
        fname_img = os.path.join(data_dir, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    elif dataset is 'test':
        fname_img = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    if not os.path.exists(fname_img):
        unzip_gz(fname_img + '.gz')
    if not os.path.exists(fname_lbl):
        unzip_gz(fname_lbl + '.gz')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        struct.unpack('>II', flbl.read(8))
        lbls = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack('>IIII', fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), rows, cols) / 127.5 - 1

    return imgs, lbls, len(lbls)


def mnist_download(download_dir):
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = url_base + file_name
        save_path = os.path.join(download_dir, file_name)
        cmd = ['curl', url, '-o', save_path]
        print('Downloading ', file_name)
        if not os.path.exists(save_path):
            subprocess.call(cmd)
        else:
            print('%s exists, skip!' % file_name)
