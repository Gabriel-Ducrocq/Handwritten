# Handwritten

mlp.py contains the source code of a multi-layer neural network designed for a handrwitten-digit recognition task. I used the back-propagation algorithm and the classic sigmoid activation function.


I used the MNIST dataset, available at : http://yann.lecun.com/exdb/mnist/

train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz contain the 60,000 training images et their labels.
t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz contain the 10,000 test images and their labels.

The "weigths" and "biases" files contain the values of the weights and bias after about 10 hours training a MLP with an input layer of 784 neurons, 30 in the hidden layer and 10 in the output layer.

This MLP performs 95.37% of accuracy on the test set.
