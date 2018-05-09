import tensorflow as tf
#from operator import mul
#from functools import reduce
#import includes.dataset_helper as dsf
import numpy
#import includes.debug as dbg
#import collections
#import os
#import math
from includes.experiment_class import Experiment
from sklearn.feature_extraction.image import PatchExtractor
from sklearn import preprocessing
import collections


KmeansOp = collections.namedtuple('KmeansOp', ['inputPlaceholder', 'allScores', 'clusteringScores', 'init', 'trainOp', 'cluster_centers_initialized'])

class KmeansLearning(Experiment):
    """
    Feature-learning model using unsupervised vector quantization (K-Means) to learn centroids.
    Paper: https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf
    """
    def _build_network(self):
        '''Generate graph for K-means feature learning'''
        #convert the whole dataset two 3x3=9 array like (1534,510,510,9)
        #with tf.device("/cpu:0"):
        flatPatchSize = self.params['receptive_field'] **2
        patches = self.graph['extractPatches'] = tf.extract_image_patches(images=self.input_tensor, ksizes=[1, self.params['receptive_field'], self.params['receptive_field'], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                 padding='SAME')
        #put them all in one batch
        patchesPerImageAx = patches.get_shape().as_list()[1]
        self.graph['extractPatches'] = patches = tf.reshape(patches,[-1, flatPatchSize], name="ReshapePatchesToOneArray")
        #shuffle it
        #patches = tf.random_shuffle(patches, name="ShufflePatches")
        #slice
        #self.graph['rndPatches'] = patches = patches[:10000,:]

        #kmeans Input
        kmeansInput = tf.placeholder(tf.float32, shape=(None, flatPatchSize), name="KmeansInput")
        #set input placeholder
        #kmeansInput = KmeansOp(kmeansInput, None, None, None, None)
        #set other kmeans graph vars
        self._kmeans_gen_graph(kmeansInput, patchesPerImageAx = patchesPerImageAx, vectorSize = flatPatchSize, num_clusters= self.params['centroids'])

    def _img_preprocessing(self, input):
        if self.params['img_preprocess']:
            #input = preprocessing.normalize(input)
            input = preprocessing.scale(input, axis=1)
            #input = self.whiten2(input)
        return input

    def _kmeans_gen_graph(self, kmeansInput, patchesPerImageAx, vectorSize,  num_clusters = 1000):
        '''Using K-Means graph generator from TF contrib library'''
        kmeans = tf.contrib.factorization.KMeans(kmeansInput, num_clusters)
        (all_scores, _, clustering_scores, cluster_centers_initialized, init, train_op) = kmeans.training_graph()

        self.kMeans = KmeansOp(kmeansInput, all_scores, clustering_scores, init, train_op, cluster_centers_initialized)
        #computing means of distances to centroids for each input vector
        with tf.name_scope('io_tensors'):
            self.tboard['preprocessedInput'] = tf.summary.image('preprocessedInput', tf.reshape(kmeansInput, [-1, self.params['receptive_field'], self.params['receptive_field'],1]), 6)
        self.graph['cluster_score_means'] = cluster_score_means = tf.reduce_mean(tf.squeeze(self.kMeans.allScores), axis=1, keepdims=True, name="GetDistMeans")
        #setting max(0, mean - x)
        self.graph['clusterScoresRaw'] = clustScores = tf.nn.relu(cluster_score_means - self.kMeans.allScores, name="ConvertInputToVector")
        self.graph['clustScores'] = clustScores = tf.reshape(clustScores, [-1, patchesPerImageAx, patchesPerImageAx, num_clusters], name="SetShapeOfVectors")
        #clust_quarters = tf.split(clustScores, 2,  axis=2)
        #split into four big patches
        clust_quarters = tf.extract_image_patches(images=clustScores, ksizes=[1, int(patchesPerImageAx / 2) , int(patchesPerImageAx / 2), 1],
                                                 strides=[1, int(patchesPerImageAx / 2), int(patchesPerImageAx / 2), 1], rates=[1, 1, 1, 1],
                                                 padding='VALID', name="SplitIntoFourPatches")
        clust_quarters_size = clust_quarters.get_shape().as_list()
        #shape [Batch, 2, 2 - for four big tiles, (patchesPerImageAx / 2)^2 - vectors in each tile, num_clusters - k-centroids
        self.graph['clust_quarters'] = clust_quarters = tf.reshape(clust_quarters, [-1, clust_quarters_size[1], clust_quarters_size[2], pow(int(patchesPerImageAx / 2), 2), num_clusters], name="ReshapeQuarters")
        clust_quarters = tf.reduce_sum(clust_quarters, axis=3, name="PoolToFourVectors")
        self.graph['features_layer'] = tf.reshape(clust_quarters,[-1, num_clusters*4],name="StackFourVectors")

        pass
    def _hook_post_tf_init(self):
        if self.trainingPatches is None:
            raise NameError("trainingPatches for KMeans is not set")
        self.session.run(self.kMeans.init, {self.kMeans.inputPlaceholder: self.trainingPatches})
        cluster_centers_initialized = self.session.run(self.kMeans.cluster_centers_initialized, {self.kMeans.inputPlaceholder: self.trainingPatches})
        if not cluster_centers_initialized:
            raise Exception("clusters centers init failed!")
        super()._hook_post_tf_init()

    def _train_batch(self, iteration):
        if iteration % 100 == 0:
            print("[training] running {} iteration..".format(iteration))
        self.session.run(self.kMeans.trainOp, {self.kMeans.inputPlaceholder: self.trainingPatches})

    def train(self):
        patch_extractor = PatchExtractor((self.params['receptive_field'], self.params['receptive_field']), max_patches=self.params['training_patches_num'])
        self.trainingPatches = patch_extractor.transform(self.dataset.train.images).reshape(-1, self.params['receptive_field'] **2)
        self.trainingPatches = self._img_preprocessing(self.trainingPatches)
        super().train()



    def _extract_features(self, images):
        #sizeVector = numpy.prod(self.graph['features_layer'].get_shape().as_list()[1:])

        self.inferencePatches, tboard_summary_output = self.session.run((self.graph['extractPatches'],self.tboard['input_images']),
                                            {self.input_tensor: images})
        self.inferencePatches = self._img_preprocessing(self.inferencePatches)
        print("inference patches: {}, {}".format(self.inferencePatches.shape[0], self.inferencePatches.shape[1]))
        self.tboard['train_writer'].add_summary(tboard_summary_output, 100000)
        # obtain kmeans scores for the patches
        #test = numpy.asarray(self.session.run(self.graph['clustQuarters'], {self.kMeans.inputPlaceholder: inferencePatches}))
        output_vectors, tboard_summary_output = self.session.run(
            (self.graph['features_layer'],self.tboard['preprocessedInput']),
            {self.kMeans.inputPlaceholder: self.inferencePatches})
        self.tboard['train_writer'].add_summary(tboard_summary_output, 100000)

        #do we need a reshape to flatten conv output?
        return numpy.asarray(output_vectors)

    def load_best_check_point(self):
        #no checkpoints in kmeans
        pass



    def whiten2(self, flat_x):
        # Calculate principal components
        sigma = numpy.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = numpy.linalg.svd(sigma)
        principal_components = numpy.dot(numpy.dot(u, numpy.diag(1. / numpy.sqrt(s + 10e-7))), u.T)

        # Apply ZCA whitening
        return numpy.dot(flat_x, principal_components)