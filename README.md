# Unsupervised clustering through feature learning using Tensorflow

## Description
The overall pipeline consists of two major steps: 
* Feature learning
* Clustering and analysis

All setups are configured through `.task` JSON files located in `tasks` directory. Results are stored in `results` folder,
TensorBoard logs and checkpoints stored in `logs` folder.

### Feature Learning
Three types of feature learning are provided: convolutional autoencoder, pre-trained CNN, vector quantization based on K-means.
#### Convolutional autoencoder
Default set of layers includes convolutional, batch normalization and max-pooling layers. Optionally dense layers can be
added in the middle of the model.
Available options:
* Random input rotation during training
* Max pooling with or without argmax (preservation of positions during decode). Argmax works on GPU only.
* Same weights for encoder and decoder
* Any number of dense layers

#### Pre-trained CNN
For instance, MNIST dataset is split into two: 1-5 digits and 6-0.
The CNN classifier is trained on the first one (1-5), than feature-learning part of the model
is evaluated on the second one (6-0). Feature-learning part - everything but the last Softmax layer.

#### Vector quantization
Feature learning described in [this paper](https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf). Based on k-means' centroids learning.

### Clustering and Quality Analysis
Output includes t-SNE embedding for three types of clustering (K-means, DBSCAN, HDBSCAN) and ground truth coloring.
Here goes the example of ground truth scatter plot for MNIST dataset (output from Quick start example):

![Convolutional autoencoder as a feature-learning method for MNIST. T-SNE.](https://user-images.githubusercontent.com/15197972/40034090-516fd6ee-57c9-11e8-9fac-ee1ecec64037.png)

3D PCA reduction using Tensorboard Embedding:

![Convolutional autoencoder as a feature-learning method for MNIST. PCA Tensorboard Projector.](https://user-images.githubusercontent.com/15197972/39952082-b1b0ae9c-555e-11e8-8b2f-a2858aed1091.png)


## Installation
Python 3 and the following libraries are required. Installation using pip:
```commandline
pip install tensorflow, matplotlib, scikit_image, hdbscan, imageio, numpy, scipy, Pillow, skimage, scikit_learn
```



## Quick start
To run convolutional autoencoder on MNIST dataset:
```commandline
python cnn_feature_cluster.py --task=examples/cnnAE
```

After training is finished, test results will be saved in `results` folder. The feature learning quality is evaluated using clustering
and V-measure entropy score. 

To start tensorboard:
```commandline
tensorboard --logdir=logs/train
``` 
Tensorboard supports input/output samples, computational graph, loss/accuracy plots, feature vector embeddings (projector) 
