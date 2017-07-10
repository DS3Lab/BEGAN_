import os
from PIL import Image
from glob import glob
import tensorflow as tf

def get_loader(batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
 
    queue=tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 3])
#    queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return queue
