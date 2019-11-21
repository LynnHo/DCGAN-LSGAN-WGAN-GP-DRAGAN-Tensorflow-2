import tensorflow as tf

from tf2lib.data import *
from tf2lib.image import *
from tf2lib.ops import *
from tf2lib.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
