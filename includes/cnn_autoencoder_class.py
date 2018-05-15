import tensorflow as tf
import math
import includes.dataset_helper as dsf
import includes.clustering as clust
from operator import mul
from functools import reduce
from includes.cnn_vectors_class import CnnVects
import os
import numpy
import time

class CnnEncoder(CnnVects):
    def _set_default_params(self):
        super()._set_default_params()
        self._default_params = {**self._default_params,
                              **{'ae_same_weights': False,
                                 'supervised_training': False}}

    def _deconv_layer(self, input_tensor, patch_size, stride, lastReconstLayer = False, max_pool=False):
        if self.graph['conv_shapes']:
            output_shape, pool_indices = self.graph['conv_shapes'].pop()
        else:
            raise Exception("No more conv shapes left for deconv.")
        if self.params['ae_same_weights']:
            if self.graph['conv_weights']:
                output_weight = self.graph['conv_weights'].pop()
            else:
                raise Exception("No more conv weights left for deconv.")
        if max_pool:
            #if pool_indices is not None:
            input_tensor = self._unpool_2d(input_tensor, pool_indices, stride = [1, 2, 2, 1], required_output_shape=output_shape)

        if self.params['ae_same_weights']:
            weights = output_weight
        else:
            weights = tf.Variable(
                tf.truncated_normal([patch_size, patch_size, output_shape[3], input_tensor.get_shape().as_list()[3]],
                                    stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, tf.float32, [output_shape[3]]))
        output_de_convolved = tf.nn.conv2d_transpose(input_tensor, weights,
                                                   tf.stack([tf.shape(self.input_tensor)[0], output_shape[1], output_shape[2], output_shape[3]]),
                                                   stride, padding='SAME')
        #don't use batchNorm & dropout on the last layer (picture reconstruction)
        if not lastReconstLayer:
            if self.params['batch_normalization']:
                output_de_convolved = self._batch_normalize(output_de_convolved, bias, convolutional = True)
        else:
            #no batch norm on last layer
            output_de_convolved = output_de_convolved + bias
        output_activated = self._activation_func(output_de_convolved)

        if not lastReconstLayer:
            #dropout only on interal layers
            output_activated = tf.nn.dropout(output_activated, self.dropout_prob_keep_conv, self._compatible_convolutional_noise_shape(output_activated))
        return output_activated

    def _set_deconv_layers(self, input_layer):
        for i, layer_params in enumerate(reversed(self.params['conv_layers'])):
            with tf.variable_scope("DeConvLayerSet{}".format(i)):
                last_reconst_layer = i == (len(self.params['conv_layers']) - 1)
                input_layer = self._deconv_layer(input_layer, layer_params.wSize, [1, layer_params.strd, layer_params.strd, 1], lastReconstLayer= last_reconst_layer, max_pool=layer_params.mxpl)
        return input_layer

    def _set_de_dense_layers(self, input_layer):
        '''Set  dense layers for decoder symmetric to encoder'''
        if not self.params['dense_layers']:
            return input_layer
        #skipping last FC layer, we don't need two middle layers
        for i, layer_params in enumerate(reversed(self.params['dense_layers'][:-1])):
            with tf.variable_scope("DeFC_LayerSet{}".format(i)):
                input_layer = self._layer_dense(input_layer, layer_params)
        #now we need to make an FC layer which is transformed into Conv for deconv input
        conv_shape, pool_indices = self.graph['conv_shapes'].pop()
        straight_shape = self._get_tensor_flat_size(conv_shape)
        skip_dropout = False
        if not self.params['conv_layers']:
            skip_dropout = True
        input_layer = self._layer_dense(input_layer, straight_shape, skip_dropout = skip_dropout)
        input_layer = tf.reshape(input_layer, shape=[-1, conv_shape[1], conv_shape[2], conv_shape[3]  ])
        return input_layer


    def _build_network(self):
        with tf.variable_scope("RotateImage"):
            input_image = self._rotate_image(self.input_tensor)
        with tf.variable_scope("conv_layers"):
            encoded = self._set_conv_layers(input_image)
        with tf.variable_scope("FC_Layers"):
            encoded = self._set_dense_layers(encoded)
        self.graph['features_layer'] = tf.reshape(encoded, [-1, self._get_tensor_flat_size(encoded)],name='features_layer')
        self._save_config(self.params['run_config'][1], {'encodedShape': self.graph['features_layer'].get_shape().as_list()}, append=True)

        # decode
        with tf.variable_scope("DeFC_Layers"):
            decoded = self._set_de_dense_layers(encoded)
        with tf.variable_scope("DeConvLayers"):
            self.graph['outputLayer'] = decoded = self._set_deconv_layers(decoded)
        with tf.name_scope('io_tensors'):
            tf.summary.image('output', decoded, 6)
        #loss per image
        self.graph['loss'] = tf.reduce_mean(tf.reduce_sum(tf.square(decoded - input_image), [1,2,3]))
        tf.summary.scalar("Loss", self.graph['loss'])
        with tf.variable_scope("Optimizer"):
            self.graph['train_step'] = tf.train.AdamOptimizer(self._learning_rate).minimize(self.graph['loss'])
        if self.params['batch_normalization']:
            self.graph['bnParams'] = tf.group(*self.bnParams)

    def train(self):
        #calc average image
        self.graph['meanImg'] = numpy.mean(self.dataset.train.images, axis=0)
        self.graph['testMeanImg'] = numpy.mean(self.dataset.test.images, axis=0)

        super().train()


    def _unpool_2d(self,
                  pool,
                  ind,
                  stride=[1, 2, 2, 1],
                  scope='unpool_2d',
                  required_output_shape=None):
      """
      2D unpooling operation for autoencoders, fully-convolutional nets, etc.
      Supports pooling with argmax and without. Set ind = None to disable pooled position preservation.
      Paper: https://arxiv.org/abs/1505.04366
      Original & discussion: https://github.com/tensorflow/tensorflow/issues/2169
      Unpooling layer after max_pool_with_argmax.
           Args:
               pool:        max pooled output tensor
               ind:         argmax indices from tf.nn.max_pool_with_argmax()
               stride:      stride is the same as for the pool
           Return:
               unpooled layer (tf tensor)
      """

      with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        if ind is None:
            pool_ = pool
            for i in range(3, 1, -1):
                shape = tf.shape(pool_)
                for j in range(stride[i - 1] - 1):
                    pool_ = tf.concat([pool_, tf.zeros(shape)], i)
            ret = tf.reshape(pool_, output_shape)
        else:
            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                              shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        #if uppooling shape should be odd, like 4x4 -> 7x7, cut it down to that shape
        if required_output_shape is not None and set_output_shape[1] > required_output_shape[1]:
            ret = ret[:,:required_output_shape[1],:required_output_shape[2],:]
        return ret








