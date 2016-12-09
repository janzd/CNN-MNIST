from __future__ import division
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

import skimage.transform
import skimage.util
from skimage.io import imread

import matplotlib.pyplot as plt

from datetime import datetime

from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import keras.optimizers as opt
import keras.utils.np_utils as kutils

from preprocessing import preprocess_image

train = pd.read_csv("../data/train.csv").values
test  = pd.read_csv("../data/test.csv").values

nb_epoch = 100
batch_size = 128
img_height, img_width = 28, 28

trainX = train[:, 1:].reshape(train.shape[0], 1, img_height, img_width)
trainX = trainX.astype(float)
trainX /= 255.0
trainX = np.array([preprocess_image(x[0]).reshape(1, img_height, img_width) for x in trainX])

print trainX.shape

trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]

# Split the training data into training and validation data
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2)

print trainX.shape, valX.shape

inputs = Input(shape=(1,28,28))

x = ZeroPadding2D((2,2))(inputs)
x = Convolution2D(64, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = ZeroPadding2D((2, 2))(x)
x = Convolution2D(128, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)

x = ZeroPadding2D((2, 2))(x)
x = Convolution2D(256, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(256, 3, 3, init='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)
x = Dropout(0.2)(x)

x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(512, 3, 3, init='he_normal')(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(512, 3, 3, init='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)
#x = Dropout(0.3)(x)

#x = ZeroPadding2D((1, 1))(x) #
#x = Convolution2D(512, 3, 3, init='he_normal')(x) #
#x = Activation('relu')(x) #

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation="relu", init='he_normal')(x)
predictions = Dense(nb_classes, activation="softmax")(x)

model = Model(input=inputs, output=predictions)

# print the model
model.summary()
# set up the optimizer
adam = opt.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# compile the model with multiclass logloss (categorical cross-entropy) as the loss function
# and use classification accuracy as another metric to measure
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

def rotation_and_shear(img, rotation_angle=15, shear_angle=30):
    img = skimage.transform.rotate(img, np.random.randint(-rotation_angle, rotation_angle))
    tf_shift = skimage.transform.SimilarityTransform(translation=(-14, -14))
    tf_inv_shift = skimage.transform.SimilarityTransform(translation=(14, 14))
    tf_shear = skimage.transform.AffineTransform(shear=np.deg2rad(np.random.randint(-shear_angle, shear_angle)))
    img = skimage.transform.warp(img, (tf_shift + (tf_shear + tf_inv_shift)).inverse)
    return img

def shift_pixels(img, shift_range=1):
    tf_shift = skimage.transform.SimilarityTransform(translation=(np.random.randint(-shift_range, shift_range), np.random.randint(-shift_range, shift_range)))
    img = skimage.transform.warp(img, tf_shift)
    return img

def add_noise(img, gauss_var=0.02):
    #img = skimage.util.random_noise(img, mode='pepper', amount=0.1)
    img = skimage.util.random_noise(img, mode='gaussian', var=gauss_var)
    return img

def batch_generator(trainX, trainY, batch_size):
    while True:
        idxs = np.arange(0, trainY.shape[0])
        np.random.shuffle(idxs)
        for i in range(trainY.shape[0] // batch_size):
            batchX = [trainX[idx] for idx in idxs[i * batch_size : (i+1) * batch_size]]
            batchY = [trainY[idx] for idx in idxs[i * batch_size : (i+1) * batch_size]]
            batchX = [add_noise(rotation_and_shear(img[0])) for img in batchX]
            batchX = np.array(batchX)
            batchX = batchX.reshape(batchX.shape[0], 1, batchX.shape[1], batchX.shape[2])
            yield batchX, np.array(batchY)


# reduce the learning rate by factor of 0.5 if the validation loss does not get lower in 7 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0000001, verbose=1)

history = model.fit_generator(batch_generator(trainX, trainY, batch_size=batch_size), validation_data=(valX, valY), 
    samples_per_epoch=np.floor(trainY.shape[0] / batch_size) * batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[reduce_lr])

# preprocess the test data to feed it into the network
testX = test.reshape(test.shape[0], 1, img_height, img_width)
testX = testX.astype(float)
testX /= 255.0

testX = np.array([preprocess_image(x[0]).reshape(1, img_height, img_width) for x in testX])

# predict class probabilities on test data
test_pred_prob = model.predict(testX)

# select the classes with highest probabilities as class predictions
test_pred = np.argmax(test_pred_prob, axis=1)
print test_pred

# predict class probabilities on all training data (that includes both training subset and validation subset so it's kind of pointless)
train_pred = model.predict(trainX)
print("Train accuracy: " + str(np.sum(np.argmax(train_pred, axis=1) == np.where(trainY == 1)[1]).astype(float) / train_pred.shape[0]))

# save predictions
if not os.path.exists('../results/'):
    os.makedirs('../results/')
np.savetxt('../results/mnist-predictions%s.csv' % datetime.now().strftime('%Y-%m-%d_%H%M'), np.c_[range(1, len(test_pred) + 1), test_pred], delimiter = ',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Saved predictions to a CSV file.")

# serialize model to JSON
if not os.path.exists('../models/'):
    os.makedirs('../models/')
model_json = model.to_json()
filename = "../models/model%s.json" % datetime.now().strftime('%Y-%m-%d_%H%M')
with open(filename, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../models/model%s_weights.h5" % datetime.now().strftime('%Y-%m-%d_%H%M'))
print("Saved model to disk.")