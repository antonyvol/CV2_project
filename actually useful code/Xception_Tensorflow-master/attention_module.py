### This is the attention module v2.0 in Keras. This NN gets pre-extracted image features from the feature_extractor.py and calculates fixation maps that represent visual saliency.

import keras
import numpy as np
import scipy.stats as st
import scipy
from feature_extractor import extract_features
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D
from keras.layers.core import Activation
from keras.models import Sequential

from EltWiseProd import EltWiseProd

# Some constants
no_of_filters = 728 * 2
batch_size = 128
epochs = 10

# Load feature maps
train_features_np = np.load("/Users/Angelie/Google Drive/train_data_features.npy")
test_features_np = np.load("/Users/Angelie/Google Drive/test_data_features.npy")
val_features_np = np.load("/Users/Angelie/Google Drive/validation_data_features.npy")

#train_features = tf.convert_to_tensor(train_features_np)
test_features = tf.convert_to_tensor(test_features_np)
val_features = tf.convert_to_tensor(val_features_np)

# Load fixation maps
def load_labels(directory):
    def load_label(img):
        np_image = scipy.misc.imread(img)
        np_image = scipy.misc.imresize(np_image, (299, 299))
        np_image = np.expand_dims(np_image, axis=0).astype(np.float32)
        np_image /= 127.5
        np_image -= 1.
        return np_image
    listing = os.listdir(directory)
    labels = np.array([load_label(directory + img) for img in listing])
    #print(load_label(directory))
    #out_arr = np.concatenate(np.squeeze(np.asarray(labels))[4:], axis=2)
    return tf.convert_to_tensor(labels)

train_labels = load_labels('/Users/Angelie/Documents/Universitaet/IAS/SS19/CV2/Eye_fixation_project/cv2_data/train/fixations/')

val_labels = load_labels('/Users/Angelie/Documents/Universitaet/IAS/SS19/CV2/Eye_fixation_project/cv2_data/val/fixations/')

# Create 2D normal continuous distribution
def gaussian2d(dim, mean=0, std=1):
    x = np.linspace(-1, 1, dim+1)
    gaussian1d = np.diff(st.norm.cdf(x, loc=mean, scale=std))
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    return gaussian2d

# Network model
model = Sequential()
model.add(Conv2D(63, kernel_size=(3,3)))
model.add(Conv2D(1, kernel_size=(1,1)))
pierced = Activation('relu')
prior = gaussian2d(17,std=0.5)
weighted = EltWiseProd([pierced, prior])
model.add(weighted)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_features_np, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_features_np, val_labels))
score = model.evaluate(val_features_np, val_labels, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
#plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()