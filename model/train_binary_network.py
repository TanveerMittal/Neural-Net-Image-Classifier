#-*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import cv2
import glob


#Architecture for network based off code for cifar problem
def cnn_architecture(input):
    #Treats dataset as input into model training
    network = input

    #Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='tanh')

    #Step 2: Max pooling
    network = max_pool_2d(network, 2)

    #Step 3: Convolution
    network = conv_2d(network, 64, 3, activation='tanh')

    #Step 4: Convolution
    network = conv_2d(network, 64, 3, activation='tanh')

    #Step 5: Max pooling
    network = max_pool_2d(network, 2)

    #Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='sigmoid')

    #Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)

    #Step 8: Fully-connected neural network with one output to make prediction
    network = fully_connected(network, 1, activation='softmax')

    #Tell tflearn how we want to train the network
    network = regression(network, loss='categorical_crossentropy', learning_rate=0.001)

    return network

#Load the data set
X = [cv2.imread(x) for x in glob.glob("./shoes/*")] + [cv2.imread(x) for x in glob.glob("./negatives/*")]
print("done reading images")
Y = [0 for x in glob.glob("./shoes/*")] + [1 for x in glob.glob("./negatives/*")]
print("done indexing outputs")

#Shuffle the data to reduce possible bias in training process
X, Y = shuffle(X, Y)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

#Blurring images so training better represents real life images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

#Create cnn with defined architecture
cnn = cnn_architecture(input_data(shape=[None, 1280, 720, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug))

#Wrap the network in a model object
model = tflearn.DNN(cnn, tensorboard_verbose=0, checkpoint_path='shoe-classifier.tfl.ckpt')

#Training
model.fit(X, Y, n_epoch=10, shuffle=True,
          show_metric=True, batch_size=64,
          snapshot_epoch=True,
          run_id='shoe-classifier')

#Save model when training is complete to a file
model.save("shoe-classifier.tfl")
print("Successfully trainined")
