from includes.cnn_vectors_class import CnnVects
from includes.cnn_autoencoder_class import CnnEncoder

from includes.kmeans_feature_learning_class import KmeansLearning
from includes.cnn_classifier_encoder_class import CnnClassifierEncoder
from includes.raw_input_class import RawInput

def create(type, args):
    if type == 'cnnClassifier':
        return CnnVects(**args)
    elif type == 'cnnAE':
        return CnnEncoder(**args)
    elif type == 'kmeans':
        return KmeansLearning(**args)
    elif type == 'cnnAEClsfr':
        return CnnClassifierEncoder(**args)
    elif type == 'rawInput':
        return RawInput(**args)
    else:
        raise Exception("Unknown feature learning class type!")



