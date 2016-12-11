# CNN for MNIST dataset

Classification of MNIST digits by means of convolutional neural networks with character shape normalization in the pre-processing phase and data augmentation by affine transformations and addition of random noise.

The code is written using Keras deep learning library.

Used in Digit Recognizer competition on Kaggle https://www.kaggle.com/c/digit-recognizer

## Network architecture

|Layer Type|Parameters|Input Size|Output Size|
|:--:|:--:|:--:|:--:|:--:|
|Input| - |28x28x1|-|-|
|Convolution (1)|64 filters, kernel 5x5, padding 2|28x28x1|28x28x64|
|ReLU| - |28x28x64|28x28x64|
|Convolution (2)|128 filters, kernel 5x5, padding 2|28x28x64|28x28x128|
|ReLU| - |28x28x128|28x28x128|
|MaxPooling (1)|stride 2|28x28x128|14x14x128|
|Convolution (3)|256 filters, kernel 5x5, padding 2|14x14x128|14x14x256|
|ReLU| - |14x14x256|14x14x256|
|Convolution (4)|256 filters, kernel 3x3, padding 1|14x14x256|14x14x256|
|ReLU| - |14x14x256|14x14x256|
|MaxPooling (2)|stride 2|14x14x256|7x7x256|
|Dropout|0.2| - | - |
|Convolution (5)|512 filters, kernel 3x3, padding 1|7x7x256|7x7x512|
|ReLU| - | - | - |
|Dropout|0.2| - | - |
|Convolution (6)|512 filters, kernel 3x3, padding 1|7x7x512|7x7x512|
|ReLU| - | - | - |
|MaxPooling (3)|stride 2|7x7x512|3x3x512|
|Dropout|0.5| - | - |
|Fully-connected (7)|2048|4608|2048|
|ReLU| - | - | - |
|Fully-connected (8)|#classes|2048|#classes|
|Softmax| - | - | - |
