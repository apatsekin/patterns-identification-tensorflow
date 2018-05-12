import tensorflow as tf
# from operator import mul
# from functools import reduce
# import includes.dataset_helper as dsf
import numpy
# import includes.debug as dbg
# import collections
# import os
# import math
from includes.experiment_class import Experiment
from sklearn.feature_extraction.image import PatchExtractor
from sklearn import preprocessing
import collections


class RawInput(Experiment):
    """
    Just use raw input as a feature layer
    """

    def _define_placeholders(self):
        pass

    def _build_network(self):
        pass

    def _train_batch(self, iteration):
        pass

    def train(self):
        pass

    def _setup_embedding_vector(self, index, vectors):
        pass

    def _extract_features(self, images):
        return images.reshape(images.shape[0], -1)

    def load_best_check_point(self):
        # no checkpoints in kmeans
        pass
