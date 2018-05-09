from includes.cnn_class import Cnn


class CnnVects(Cnn):
    """
    Extends Cnn class with some feature extraction methods
    """

    def _extract_features(self, images):
        run_params = {self.input_tensor: images,
                      self.dropout_prob_keep: 1,
                      self.dropout_prob_keep_conv: 1}
        if self.params['batch_normalization']:
            run_params[self.bnTestFlag] = True
        output_vectors = self.session.run(self.graph['features_layer'], run_params)
        # do we need a reshape to flatten conv output?
        return output_vectors
