### This is the attention module v2.0 in Keras. This NN gets pre-extracted image features from the feature_extractor.py and calculates fixation maps that represent visual saliency.

import keras
import numpy as np
import scipy.stats as st
import scipy
from feature_extractor import extract_features
import tensorflow as tf

from keras.layers import Dense, Conv2D
from keras.layers.core import Activation
from keras.models import Sequential

from EltWiseProd import EltWiseProd

# Some constants
no_of_filters = 728 * 2


# Create 2D normal continuous distribution
def gaussian2d(dim, mean=0, std=1):
    x = np.linspace(-1, 1, dim+1)
    gaussian1d = np.diff(st.norm.cdf(x, loc=mean, scale=std))
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    return gaussian2d

feature_map = np.load("/Users/Angelie/Documents/Universitaet/IAS/SS19/CV2/Eye_fixation_project/validation_data_features.npy")

features = tf.convert_to_tensor(feature_map)

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
'''
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()'''
