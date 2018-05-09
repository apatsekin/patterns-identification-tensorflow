# Unsupervised clustering through feature learning using Tensorflow

## Quick start
To run convolutional autoencoder on MNIST dataset:
```commandline
python cnn_feature_cluster.py --load=examples/cnnAE
```

After training is finished, test results will be saved in `results` folder. The feature learning quality is evaluated using clustering
and V-measure entropy score. 

To start tensorboard:
```commandline
tensorboard --logdir=logs/train
``` 
Tensorboard supports input/output samples, computational graph, loss/accuracy plots, feature vector embeddings (projector) 
