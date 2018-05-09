import numpy
import math

class DataSet(object):

  def __init__(self,
               images,
               labels=None,
               fileNames=None,
               one_hot=False,
               dtype=numpy.float32,
               reshape=False,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    Original: https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    """
    #seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed()
    #dtype = dtypes.as_dtype(dtype).base_dtype
    if labels is not None:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    if fileNames is not None:
      assert images.shape[0] == fileNames.shape[0], (
          'images.shape: %s fileNames.shape: %s' % (images.shape, fileNames.shape))

    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2] * images.shape[3])
    if dtype == numpy.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      #map(lambda x: x * (1.0 / 255.0), images)
      for i in range(len(images)):
        images[i] *= 1.0 / 255.0
      #images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels if labels is not None else None
    self._fileNames = fileNames if fileNames is not None else None
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def fileNames(self):
    return self._fileNames

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def split_to_batches(self, batch_size):
    images_splitted = numpy.array_split(self.images, math.ceil(len(self.images) / batch_size))
    if self.labels is not None:
        labels_splitted = numpy.array_split(self.labels, math.ceil(len(self.labels) / batch_size))
        return zip(images_splitted, labels_splitted)
    else:
        return zip(images_splitted, [None] * len(images_splitted))



  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      if self.labels is not None:
        self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      if self.labels is not None:
        labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        if self.labels is not None:
            self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      if self.labels is not None:
        labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0) if self.labels is not None else None
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end] if self.labels is not None else None