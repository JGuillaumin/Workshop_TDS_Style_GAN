import numpy as np
from scipy.misc import imresize, imread
import threading
import os
import random
from glob import glob


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
# only with resize (and optionally cropping) !
class BatchGenerator(object):
  def __init__(self, directory,
               target_size=(128, 128),
               batch_size=32,
               shuffle=True,
               seed=None,
               color_mode='rgb',
               pre_processing='tf'):
    """

    :param directory:
    :param target_size:
    :param reshape_mode:
    :param batch_size:
    :param shuffle:
    :param seed:
    :param color_mode:
    """

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

    self.directory = directory
    self.target_size = tuple(target_size)

    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed

    if color_mode not in {'rgb', 'bgr'}:
      raise ValueError('Invalid color mode:', color_mode, '; expected "rgb" or "bgr".')
    self.color_mode = color_mode

    if pre_processing not in {"unit", 'tf'}:
      raise ValueError('Invalid pre-processing mode: ', pre_processing, '; expected "unit" or "tf"')
    self.pre_processing = pre_processing

    # + (3,) : for RGB channels
    self.image_shape = self.target_size + (3,)

    self.samples = _count_valid_files_in_directory(directory, white_list_formats)
    print("Found {} images".format(self.samples))

    self.filenames = _list_valid_filenames_in_directory(directory, white_list_formats)

    self._batch_index = 0
    self._total_batches_seen = 0
    self._lock = threading.Lock()
    self._index_generator = self._flow_index()

  def _flow_index(self):

    # Ensure self.batch_index is 0.
    self.reset()
    while 1:
      if self.seed is not None:
        np.random.seed(self.seed + self._total_batches_seen)
      if self._batch_index == 0:
        index_array = np.arange(self.samples)
        if self.shuffle:
          index_array = np.random.permutation(self.samples)

      current_index = (self._batch_index * self.batch_size) % self.samples
      if self.samples > current_index + self.batch_size:
        current_batch_size = self.batch_size
        self._batch_index += 1
      else:
        self._batch_index = 0
        continue
      self._total_batches_seen += 1
      yield (index_array[current_index: current_index + self.batch_size], current_index)

  def reset(self):
    self.batch_index = 0

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    """For python 2.x.
    # Returns
        The next batch.
    """
    with self._lock:
      index_array, current_index = next(self._index_generator)
    # The transformation of images is not under thread lock
    # so it can be done in parallel

    batch_imgs = np.zeros((self.batch_size,) + self.image_shape, dtype=np.float32)

    # build batch of image data
    for i, j in enumerate(index_array):
      fname = self.filenames[j]
      batch_imgs[i] = load_img(os.path.join(self.directory, fname),
                               color_mode=self.color_mode,
                               target_size=self.target_size)

    # minimal pre-processing
    # images : scale lr images to [0,1]
    if self.pre_processing == "unit":
      batch_imgs = np.divide(batch_imgs, 255.)
    else: # self.pre_processing == "tf":
      batch_imgs /= 127.5
      batch_imgs -= 1.

    return batch_imgs


def load_img(path, color_mode='rgb', target_size=(128,128)):
  img = imread(path, mode='RGB')
  if color_mode == 'bgr':
    img = img[...,::-1]

  img = imresize(img, size=target_size, interp='bicubic')

  return img


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def _count_valid_files_in_directory(directory, white_list_formats):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


# inspired from  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def _list_valid_filenames_in_directory(directory, white_list_formats):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.
    # Returns
        filenames: the path of valid files in `directory`
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    filenames = []
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return filenames