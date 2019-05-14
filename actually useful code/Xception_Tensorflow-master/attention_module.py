import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import feature_extractor

# Create 2D normal continuous distribution
def gaussian2d(dim, mean=0, std=1):
    x = np.linspace(-1, 1, dim+1)
    gaussian1d = np.diff(st.norm.cdf(x, loc=mean, scale=std))
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    return gaussian2d


# For the feature tensor
dropout = tf.placeholder(tf.float32, shape=(None,19,19,no_of_filters))

# Saliency layer and "piercer"
saliency_conv = tf.layers.conv2d(dropout_placeholder, 64, (3,3), activation='relu')
pierce_conv = tf.layers.conv2d(saliency_conv, 1, (1,1), activation='relu')

# Calc prior & multiply with conv-nn output
prior = gaussian2d(17,std=0.5)
output = tf.math.multiply(pierce_conv,prior)


sess = tf.Session()
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    dropout = feature_extractor.extract_features()

