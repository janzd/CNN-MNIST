import numpy as np
import pandas as pd
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import keras.optimizers as opt
import keras.utils.np_utils as kutils

train = pd.read_csv("../data/train.csv").values
test  = pd.read_csv("../data/test.csv").values

nb_epoch = 100

batch_size = 128
img_rows, img_cols = 28, 28

nb_filters_1 = 64
nb_filters_2 = 128
nb_filters_3 = 256
nb_filters_4 = 512
nb_filters_5 = 1024
filter_size_1 = 5
filter_size_2 = 3

trainX = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0

data_mean = np.mean(trainX)

trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]

inputs = Input(shape=(1,28,28))

x = ZeroPadding2D((2,2))(inputs)
x = Convolution2D(64, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = ZeroPadding2D((2, 2))(x)
x = Convolution2D(64, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)
#x = GaussianNoise(sigma=3)(x)

x = ZeroPadding2D((2, 2))(x)
x = Convolution2D(256, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = ZeroPadding2D((2, 2))(x)
x = Convolution2D(256, 5, 5, init='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)
x = Dropout(0.2)(x)

x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(512, 3, 3, init='he_normal')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = ZeroPadding2D((1, 1))(x)
x = Convolution2D(512, 3, 3, init='he_normal')(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)
#x = Dropout(0.2)(x)

#x = ZeroPadding2D((1, 1))(x) #
#x = Convolution2D(512, 3, 3, init='he_normal')(x) #
#x = Activation('relu')(x) #

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation="relu", init='he_normal')(x)
predictions = Dense(nb_classes, activation="softmax")(x)

model = Model(input=inputs, output=predictions)

model.summary()
adam = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#def batch_generator(trainX, trainY, batch_size):
#    while True:
#        for i in range(trainY.shape[0] / batch_size):
#        	batch = trainX[i * batch_size : (i+1) * batch_size]


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)

model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, verbose=1, callbacks=[reduce_lr])

testX = test.reshape(test.shape[0], 1, 28, 28)
testX = testX.astype(float)
testX /= 255.0

test_pred = model.predict(testX)
train_pred = model.predict(trainX)

print("Train accuracy: " + str(np.sum(np.argmax(train_pred, axis=1) == np.where(trainY == 1)[1]).astype(float) / train_pred.shape[0]))

np.savetxt('../results/mnist-vggnet.csv', np.c_[range(1,len(test_pred)+1),test_pred], delimiter = ',', header = 'ImageId,Label', comments = '', fmt='%d')
