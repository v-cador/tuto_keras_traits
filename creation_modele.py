import cv2
import sys, os, gc
import numpy as np
import random
import time as tm
import pickle
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
# from keras.metrics import binary_accuracy
# from keras.metrics import categorical_accuracy
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, TFOptimizer
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPoolicmdng2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import datetime
import matplotlib.pyplot as plt

import matplotlib
# import matplotlib.pyplot as plt
import warnings

K.set_image_dim_ordering('tf')

# ----------------------------------------------------------
# Neural Network model functions
# ----------------------------------------------------------

# fmeasure implementation for Keras 2.0
# fct de la precision (TP parmi les predits positif) et du recall (TP parmis les obs positifs)
def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # nb de vrais positifs
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))  # nb de predits positifs
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))  # nb de positifs observés

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


# Randomly permute the weights of a Neural Network
# ???
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)




# Compile Neural Network parameters
def compile_sequential_model(num_classes, epochs, slide_w=50, slide_H=50, grayscale=True, binary_classif=True, dropout=0.2, dense=512, lrate=0.001, nb_conv=32):
    print("   >> Compile sequential model")

    # Depth = 3 in RGB Mode else Depth = 1 in Grayscale mode
    if grayscale:
        picture_depth = 1
    else:
        picture_depth = 3

    # Create the sequential model (pictures = (3=GBR, X=80, Y=80)
    model = Sequential()

    # Create (3,3) convolutions and generate 32 pictures
    # model.add(Convolution2D(32, 3, 3, input_shape=(picture_depth, slide_w, slide_H), border_mode='same',
    #                         activation='relu', W_constraint=maxnorm(3)))
    model.add(Convolution2D(nb_conv, 3, 3, input_shape=(slide_w, slide_H, picture_depth), border_mode='same',
                            activation='relu', W_constraint=maxnorm(3)))
    # model.add(Convolution2D(10, 3, 3, input_shape=(slide_w, slide_H, picture_depth), border_mode='same',
    #                         activation='relu', W_constraint=maxnorm(3)))

    # Dropout apply to next layer input values
    model.add(Dropout(dropout))
    # model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))

    # Reduce the picture by taking the max value of 2x2 across the image
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Make a 1D vector (very large size = N)
    model.add(Flatten())

    # Fully connected layer (dense) N x 512 multiply output by fully connected layer (Wx+b)
    model.add(Dense(dense, activation='relu', W_constraint=maxnorm(3)))

    # Through away 50% of neurons to generalize
    model.add(Dropout(0.5))

    # Last layer fit to one results vector
    if binary_classif:
        # Binary class (0/1)
        model.add(Dense(1, activation='sigmoid'))
    else:
        # Multi class label (Class1,Class2,Class3,...)
        model.add(Dense(num_classes, activation='softmax'))

    # Specify backpropagation method (Stochastic Gradien Decent)
    # lrate = 0.001
    optim = SGD(lr=lrate, momentum=0.9, nesterov=True)

    # To try other optimizers
    # decay = lrate/epochs
    # optim = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # optim = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
    # optim = Adam(lr=lrate)

    # Compile model
    if binary_classif:
        # model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[fmeasure])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['categorical_accuracy'])

    # Print model decription
    print(model.summary())

    return model

