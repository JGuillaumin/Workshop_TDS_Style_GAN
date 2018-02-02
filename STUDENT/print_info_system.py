import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import scipy
import keras
import skimage
import sklearn
from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_available_devices():
  local_device_protos = device_lib.list_local_devices()
  return [(x.device_type, x.name) for x in local_device_protos]


if __name__ == "__main__":
  print("tensorflow : {}".format(tf.VERSION))
  print("numpy : {}".format(np.__version__))
  print("scipy : {}".format(scipy.__version__))
  print("keras : {}".format(keras.__version__))
  print("skimage : {}".format(skimage.__version__))
  print("sklearn : {}".format(sklearn.__version__))

  print("=========================================")
  pprint(get_available_devices())
  print("=========================================")
