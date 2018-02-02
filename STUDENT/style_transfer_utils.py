from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from keras import constraints
from keras import regularizers
import numpy as np
from scipy.misc import imresize


def resize_image(img, target_size=(256, 256)):

  h, w, _ = img.shape
  short_edge = min([h,w])

  # maximal center crop
  yy = int((h - short_edge) / 2.)
  xx = int((w - short_edge) / 2.)
  img = img[yy: yy + short_edge, xx: xx + short_edge]

  # resize
  img = imresize(img, size=target_size, interp='bicubic')

  return img


def preprocess_image(img):

  # convert to float
  if img.dtype != np.float32:
    img = img.astype(np.float32)

  # RGB -> BGR
  img = img[..., ::-1]

  # subtract means (from ImageNet dataset)
  img[..., 0] -= 103.939
  img[..., 1] -= 116.779
  img[..., 2] -= 123.68

  # add a new dim for a fake 'batch' dim
  img = np.expand_dims(img, axis=0)
  return img


def deprocess_image(img):

    if len(img.shape)>3:
      img = img[0]
    # add the mean (BGR format)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    # BGR to RGB
    img = img[:, :, ::-1]

    img = np.clip(img, 0, 255).astype('uint8')
    return img


class TrainableImageLayer(Layer):
  def __init__(self, input_shape=None, input_values=None, mean=0.0, stddev=50.0, seed=None, **kwargs):

    self.mean = mean
    self.stddev = stddev
    self.seed = seed

    self.trainable = True
    self.built = True
    self.sparse = False
    self.dtype = K.floatx()

    if input_shape and input_values:
      raise ValueError('Only provide the input_shape OR '
                       'input_values argument to '
                       'TrainableInputLayer, not both at the same time.')

    if input_values is not None:
      # If input_values is set
      if len(input_values.shape) == 3:
        shape = (1,) + tuple(input_values.shape)
      elif len(input_values.shape) == 4:
        shape = tuple(input_values.shape)
      else:
        raise ValueError('Incorrect rank for input_values')
    else:
      shape = (1,) + tuple(input_shape)

    self.shape = shape

    if input_shape:
      # if input_shape is set, initilize the variable with normal random values
      initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)
    else:
      initializer = initializers.Constant(value=np.expand_dims(input_values, axis=0))

    self.initializer = initializers.get(initializer)
    self.constraint = constraints.get(None)
    self.regularizer = regularizers.get(None)

    super(TrainableImageLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(shape=self.shape,
                                   dtype=self.dtype,
                                   initializer=self.initializer,
                                   name='kernel',
                                   regularizer=self.regularizer,
                                   constraint=self.constraint)
    super(TrainableImageLayer, self).build(input_shape)

  def call(self, inputs, **kwargs):
    return self.kernel

  def compute_output_shape(self, input_shape):
    return tuple(self.shape)

  def get_config(self):
    config = {'shape': self.shape}
    base_config = super(TrainableImageLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
