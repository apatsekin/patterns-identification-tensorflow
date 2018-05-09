import tensorflow as tf
from includes.cnn_autoencoder_class import CnnEncoder

class CnnClassifierEncoder(CnnEncoder):
    """
    Mix of autoencoder and classifier to create a feature learning pipeline
    """
    def _build_network(self):
        input_image = self._rotate_image(self.input_tensor)
        #encode
        encoded = self._set_conv_layers(input_image)
        encoded = self._set_dense_layers(encoded)
        self.graph['features_layer'] = tf.reshape(encoded, [-1, self._get_tensor_flat_size(encoded)],name='features_layer')
        classes_layer, _ = self._layer_dense_logits(encoded, self.dataset.params['numClasses'])
        sft_max_layer = tf.nn.softmax(classes_layer)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=classes_layer, labels=self._labels_tensor)

        self._save_config(self.params['run_config'][1], {'encodedShape': self.graph['features_layer'].get_shape().as_list()}, append=True)

        # decode
        decoded = self._set_de_dense_layers(encoded)
        self.graph['outputLayer'] = decoded = self._set_deconv_layers(decoded)
        with tf.name_scope('io_tensors'):
            tf.summary.image('output', decoded, 6)
        self.graph['loss'] = tf.reduce_mean(tf.reduce_sum(tf.square(decoded - input_image), [1,2,3])) + tf.reduce_sum(cross_entropy)
        tf.summary.scalar("Loss", self.graph['loss'])
        correct_prediction = tf.equal(tf.argmax(sft_max_layer, 1), tf.argmax(self._labels_tensor, 1))
        self.graph['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("Accuracy", self.graph['accuracy'])
        self.graph['train_step'] = tf.train.AdamOptimizer(self._learning_rate).minimize(self.graph['loss'])
        if self.params['batch_normalization']:
            self.graph['bnParams'] = tf.group(*self.bnParams)



