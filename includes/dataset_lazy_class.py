import numpy
from includes.dataset_class import DataSet
import time

class DataSetLazy(DataSet):

    def __init__(self,
               dataHandler,
               indiceArray,
               labels=None,
               dataSetParams = None,
               one_hot=False,
               dtype=numpy.float32,
               reshape=False,
               seed=None):

        numpy.random.seed()

        # if labels is not None:
        #   assert len(indiceArray) == labels.shape[0], (
        #       'images.shape: %s labels.shape: %s' % (len(indiceArray), labels.shape))
        self._num_examples = len(indiceArray)
        self._imagesHandler = dataHandler
        self._indicesArray = indiceArray
        self._imagesStartPos = dataSetParams['startPosition']
        self._imagesItemBytesSize = dataSetParams['itemBytesSize']
        self._imageRows = dataSetParams['imageHeight']
        self._imageCols = dataSetParams['imageWidth']
        self._imageChannels = dataSetParams['imageChannels']
        self._dtype = dtype
        self._reshape = reshape
        self._images = None
        self._labels = labels if labels is not None else None
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        if self._images is None:
            self._images = self.readLazy(self._indicesArray)
        return self._images



    @property
    def labels(self):
        return self._labels[self._indicesArray]

    @property
    def num_examples(self):
        return self._num_examples

    # @property
    # def epochs_completed(self):
    #     return self._epochs_completed

    def convertType(self, dataset):
        if self._dtype == numpy.float32:
            dataset = dataset.astype(numpy.float32)
            dataset = numpy.multiply(dataset, 1.0 / 255.0)
        return dataset
    def reshapeFromSource(self, dataset):
        dataset = dataset.reshape(-1, self._imageRows, self._imageCols, self._imageChannels)
        return dataset

    def reshapeDataset(self, dataset):
        pass



    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            numpy.random.shuffle(self._indicesArray)
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_part = self._indicesArray[start:].copy()
            # Shuffle the data
            if shuffle:
                numpy.random.shuffle(self._indicesArray)
            # Start next epoch
            self._index_in_epoch = batch_size - len(rest_part)
            images_rest_part = self.readLazy(rest_part)
            newPartIndices = self._indicesArray[0:self._index_in_epoch]
            images_new_part = self.readLazy(newPartIndices)
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), \
                   numpy.concatenate((self._labels[rest_part],
                                      self._labels[newPartIndices]), axis=0) if self.labels is not None else None
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.readLazy(self._indicesArray[start:end]), self._labels[self._indicesArray[start:end]] if self.labels is not None else None


    def readLazy(self, indices):
        #move handler to first element
        bytestring = b''
        #dbg.start_timer('lazyRead')
        for index in indices:
            self._imagesHandler.seek(self._imagesStartPos + index * self._imagesItemBytesSize, 0)
            bytestring += self._imagesHandler.read(self._imagesItemBytesSize)
        output = numpy.frombuffer(bytestring, dtype=numpy.uint8)
        #dbg.measure_time('lazyRead', 'Read {} elements'.format(len(indices)))
        return self.reshapeFromSource(self.convertType(output))