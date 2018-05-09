import tensorflow as tf
import includes.dataset_helper as dsf
import numpy
import includes.debug as dbg
import os
import math
from includes.experiment_class import Experiment


class Cnn(Experiment):
    def _set_default_params(self):
        super()._set_default_params()
        self._default_params = {**self._default_params,
                                **{'is_maxpool_with_indices': True,
                                   'activation_func': 'relu',
                                   'supervised_training': True,
                                   'normalize_image': False,
                                   'max_learning_rate': 0.02,
                                   'min_learning_rate': 0.0001,
                                   'decay_speed': 1600,
                                   'rotate_images': False,
                                   'batch_normalization': True}}

    def _define_placeholders(self):
        super()._define_placeholders()
        if self.params['supervised_training']:
            self._labels_tensor = tf.placeholder(tf.float32, [None, self.dataset.params['numClasses']],
                                                 name="LabelsTensor")
        self._learning_rate = tf.placeholder(tf.float32, name="LearningRate")
        if self.params['batch_normalization']:
            # batch norm
            self.bnTestFlag = tf.placeholder(tf.bool, name="BatchNormTestFlag")
            self.bnIteration = tf.placeholder(tf.int32, name="BatchNormIteration")
            self.bnParams = []
        # dropout
        self.dropout_prob_keep = tf.placeholder(tf.float32, name="DropoutProb")
        self.dropout_prob_keep_conv = tf.placeholder(tf.float32, name="DropoutProbConv")

    def _define_properties(self):
        super()._define_properties()
        self.graph['conv_shapes'], self.graph['conv_weights'] = [], []
        # store min loss
        self._best_loss = math.inf

    def _load_dataset(self, path):
        self.dataset = dsf.read_data_sets(path, one_hot=True, reshape=False, shuffle=True,
                                          validation_size=self.params['test_size'],
                                          k_fold=self.params['k_fold'])

    def _activation_func(self, x, function=None):
        if function is None:
            function = self.params['activation_func']
        with tf.variable_scope(function):
            if function == 'sigmoid':
                return tf.nn.sigmoid(x)
            elif function == 'relu':
                return tf.nn.relu(x)
            elif function == 'leakRelu':
                return tf.nn.leaky_relu(x, alpha=0.2)
            else:
                raise Exception("Unknown activation funtion: {}".format(function))

    def _batch_normalize(self, input_tensor, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, self.bnIteration)
        if convolutional:
            mean, variance = tf.nn.moments(input_tensor, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(input_tensor, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        self.bnParams.append(update_moving_averages)
        mean_tensor = tf.cond(self.bnTestFlag, lambda: exp_moving_avg.average(mean), lambda: mean)
        variance_tensor = tf.cond(self.bnTestFlag, lambda: exp_moving_avg.average(variance), lambda: variance)
        output_tensor = tf.nn.batch_normalization(input_tensor, mean_tensor, variance_tensor, offset, None, 1e-5)
        return output_tensor

    # dropout thing
    # A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
    def _compatible_convolutional_noise_shape(self, input_tensor):
        noise_shape = tf.shape(input_tensor)
        noise_shape = noise_shape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noise_shape

    def _network_layer_set(self, input_tensor, patch_size, stride, conv_channels, max_pool):
        weights = tf.get_variable("weights",
                                  shape=(patch_size, patch_size, input_tensor.get_shape().as_list()[3], conv_channels),
                                  initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("weights", weights)
        bias = tf.get_variable("bias", shape=(conv_channels), initializer=tf.constant_initializer(0.0))
        with tf.variable_scope("Convolution"):
            output_convolved = tf.nn.conv2d(input_tensor, weights, stride,
                                            padding='SAME')  # update slide window size from params

        self.graph['conv_weights'].append(weights)
        if self.params['batch_normalization']:
            with tf.variable_scope("BatchNorm"):
                output_convolved = self._batch_normalize(output_convolved, bias, convolutional=True)
        else:
            output_convolved = output_convolved + bias
        with tf.variable_scope("Activation"):
            output_activated = self._activation_func(output_convolved)
        # maxpooling
        argmax = None
        if max_pool:
            with tf.variable_scope("MaxPooling"):
                # max_pool with indices is not required in CNN and doesn't work on CPU
                if self.params['supervised_training'] or self.params['is_maxpool_with_indices'] is False:
                    output_activated = tf.nn.max_pool(output_activated, [1, 2, 2, 1], [1, 2, 2, 1],
                                                      padding='SAME')
                else:
                    output_activated, argmax = tf.nn.max_pool_with_argmax(output_activated, [1, 2, 2, 1], [1, 2, 2, 1],
                                                                          padding='SAME')
        # setting shape of input layer (which was convolved) + if maxpooled indices of output layer
        # so during decode we can 1) apply depool and then 2) deconv to input shape
        self.graph['conv_shapes'].append((input_tensor.get_shape().as_list(), argmax))
        with tf.variable_scope("Dropout"):
            return tf.nn.dropout(output_activated, self.dropout_prob_keep_conv,
                                 self._compatible_convolutional_noise_shape(output_activated))

    def _layer_dense(self, input_tensor, size, skip_dropout=False, is_logits=False):
        # getting number of neurons in last conv layer like 4*4*50 => l
        # to convert into FC layer
        if (len(
                input_tensor.get_shape().as_list()) == 4):  # input_layer is convolutional, let's save it's form before reshaping
            # don't need pooling indices here, since it's just for reshapeing back to matrix from vector
            self.graph['conv_shapes'].append((input_tensor.get_shape().as_list(), None))
        output_linear, bias = self._layer_dense_logits(input_tensor, size)  # tf.matmul(outputReshaped, weights)
        if (is_logits):
            return output_linear
        if self.params['batch_normalization']:
            with tf.variable_scope("BatchNorm"):
                output_linear = self._batch_normalize(output_linear, bias)
        else:
            output_linear = output_linear + bias
        output_activated = self._activation_func(output_linear)
        if not skip_dropout:
            output_activated = tf.nn.dropout(output_activated, self.dropout_prob_keep)
        return output_activated

    def _layer_dense_logits(self, input_tensor, output_num):
        '''FC layer without softmax, since TF has softmax+crossentropy C function'''
        shape = self._get_tensor_flat_size(input_tensor)
        if len(input_tensor.get_shape().as_list()) > 2:
            input_tensor = tf.reshape(input_tensor, shape=[-1, shape])
        weights = tf.Variable(tf.truncated_normal([shape, output_num], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, tf.float32, [output_num]))
        return tf.nn.xw_plus_b(input_tensor, weights, bias), bias

    def _set_conv_layers(self, input_layer):
        '''iteration to setup Convolutional layers to graph'''
        for i, layerParams in enumerate(self.params['conv_layers']):
            with tf.variable_scope("ConvLayerSet{}".format(i)):
                input_layer = self._network_layer_set(input_layer, layerParams.wSize,
                                                      stride=[1, layerParams.strd, layerParams.strd, 1],
                                                      conv_channels=layerParams.ch, max_pool=layerParams.mxpl)
        return input_layer

    # iteration to setup FullyConnected layers to graph

    def _set_dense_layers(self, input_layer):
        for i, layerParams in enumerate(self.params['dense_layers']):
            with tf.variable_scope("FC_Layer{}".format(i)):
                input_layer = self._layer_dense(input_layer, layerParams)
        return input_layer

    def _build_network(self):
        '''main method to describe overall graph structure'''
        with tf.variable_scope("ImageRotation"):
            input_image = self._rotate_image(self.input_tensor)
        iter_layer = self._set_conv_layers(input_image)
        iter_layer = self._set_dense_layers(iter_layer)
        # give a name to features layer in TF
        self.graph['features_layer'] = tf.identity(iter_layer, name="features_layer")
        # self._setup_embedding_vector()
        fc_layer2, _ = self._layer_dense_logits(iter_layer, self.dataset.params['numClasses'])
        sft_max_layer = tf.nn.softmax(fc_layer2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_layer2, labels=self._labels_tensor)
        # avg loss per image across batch
        self.graph['loss'] = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("Loss", self.graph['loss'])
        correct_prediction = tf.equal(tf.argmax(sft_max_layer, 1), tf.argmax(self._labels_tensor, 1))
        self.graph['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("Accuracy", self.graph['accuracy'])
        with tf.variable_scope("Optimizer"):
            self.graph['train_step'] = tf.train.AdamOptimizer(self._learning_rate).minimize(cross_entropy)
        if self.params['batch_normalization']:
            self.graph['bnParams'] = tf.group(*self.bnParams)

    def _gen_run_input(self, is_test=False):
        '''Generates input Ops for session.run: what should it calculate during train or test?'''
        runParams = {'loss': self.graph['loss']}
        if is_test:
            runParams['tboard_summary_output'] = self.tboard['merged_summary']
            runParams['features_layer'] = self.graph['features_layer']
        if 'accuracy' in self.graph:
            runParams['accuracy'] = self.graph['accuracy']
        return runParams

    def _test_run(self, iteration):
        print("Running test for {} iteration...".format(iteration))
        test_outputs = []
        # if no labels in dataset, test_labels_batch will be = None
        for test_input_batch, test_labels_batch in self.dataset.test.split_to_batches(self.params['training_batch_size']):
            if self.params['normalize_image']:
                test_input_batch = numpy.array([img - self.graph['testMeanImg'] for img in test_input_batch])

            test_output = self.session.run(self._gen_run_input(is_test=True),
                                           self._gen_run_params(test_input_batch,
                                                                iteration,
                                                                labels=test_labels_batch, is_test=True))
            self.tboard['train_writer'].add_summary(test_output['tboard_summary_output'], iteration)
            test_outputs.append(test_output)
        test_outputs = dsf.merge_arrays(test_outputs)
        # autoencoder doesn't have accuracy
        if 'accuracy' in test_outputs:
            test_outputs['accuracy'] = numpy.mean(test_outputs['accuracy'])
        test_outputs['loss'] = numpy.mean(test_outputs['loss'])
        # testVectors = numpy.concatenate(test_outputs['features_layer'], axis=0)

        if (iteration > 100 and test_outputs['loss'] < self._best_loss):
            self.saver.save(self.session, os.path.join(self._log_dir, "model.ckpt"), iteration)
            self._best_loss = test_outputs['loss']
            print("new loss achieved: {}".format(self._best_loss))
        return test_outputs

    def _gen_run_params(self, batch, iteration, labels=None, is_test=False, is_batch_norm_run=False):
        '''Paramters generator for session.run (training, testing, labels? batchnorm?)'''
        learning_rate_value = self.params['min_learning_rate'] + (self.params['max_learning_rate'] - self.params[
            'min_learning_rate']) * math.exp(-iteration / self.params['decay_speed'])
        run_params = {}
        run_params[self.input_tensor] = batch
        if not is_test:
            run_params[self._learning_rate] = learning_rate_value

        if self.params['supervised_training']:
            run_params[self._labels_tensor] = labels
        # DROPOUT
        if is_test or is_batch_norm_run:
            run_params[self.dropout_prob_keep_conv] = 1.0
            run_params[self.dropout_prob_keep] = 1.0
        else:
            run_params[self.dropout_prob_keep_conv] = self.params['dropout_prob_keep_conv']
            run_params[self.dropout_prob_keep] = self.params['dropout_prob_keep']
        # BATCH NORM
        if self.params['batch_normalization']:
            if is_test:
                run_params[self.bnTestFlag] = True
            else:
                run_params[self.bnTestFlag] = False
        if is_batch_norm_run:
            run_params[self.bnIteration] = iteration
        return run_params

    def _train_batch(self, iteration):
        input_batch, labels_batch = self.dataset.train.next_batch(self.params['training_batch_size'])
        # train only on difference from mean
        if self.params['normalize_image']:
            if self.graph['meanImg'] is None or self.graph['testMeanImg'] is None:
                raise Exception("mean image not set!")
                input_batch = numpy.array([img - self.graph['meanImg'] for img in input_batch])

        dbg.start_timer('_train_batch')
        self.session.run(self.graph['train_step'], self._gen_run_params(input_batch, iteration, labels=labels_batch))

        if self.params['batch_normalization']:
            self.session.run(self.graph['bnParams'],
                             self._gen_run_params(input_batch, iteration, labels=labels_batch, is_batch_norm_run=True))
        dbg.measure_time('_train_batch', 'Train {} batch'.format(self.params['training_batch_size']))

        if (iteration % 100 == 0):
            test_output = self._test_run(iteration)
            train_output = self.session.run(self._gen_run_input(),
                                            self._gen_run_params(input_batch, iteration, labels=labels_batch))
            print("{}: train loss: {}, test loss: {}, accuracy: {}".format(iteration, train_output['loss'],
                                                                           test_output['loss'],
                                                                           test_output[
                                                                               'accuracy'] if 'accuracy' in test_output else 'N/A'))

    def load_best_check_point(self):
        '''Restoring best checkpoint from logs dir. The latest one assuming that we save checkpoint only when better loss achieved.'''
        last_checkpoint = self.saver.last_checkpoints
        if (len(last_checkpoint) < 1):
            raise Exception("No saved model found (checkpoint)!")
        print("Restoring last checkpoint - {}".format(last_checkpoint[-1]))
        self.saver.restore(self.session, last_checkpoint[-1])
        print("Model restored..")
        self._save_model_builder()
