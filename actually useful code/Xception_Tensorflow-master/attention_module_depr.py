import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy
import feature_extractor


#train_data_path = '/Users/Angelie/Documents/Universitaet/IAS/SS19/CV2/Eye_fixation_project/cv2_data/train'
#val_data_path = '/Users/Angelie/Documents/Universitaet/IAS/SS19/CV2/Eye_fixation_project/cv2_data/val'
#test_data_path = '/Users/Angelie/Documents/Universitaet/IAS/SS19/CV2/Eye_fixation_project/cv2_data/test'

no_of_filters = 728 * 2

def create_label(img_path):
    img_array = scipy.misc.imread(img_path, flatten=True)
    img_array = scipy.misc.imresize(img_array, (19, 19))
    return img_array

# Create 2D normal continuous distribution
def gaussian2d(dim, mean=0, std=1):
    x = np.linspace(-1, 1, dim+1)
    gaussian1d = np.diff(st.norm.cdf(x, loc=mean, scale=std))
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    return gaussian2d


# For the feature tensor
dropout_placeholder = tf.placeholder(tf.float32, shape=(None,19,19, no_of_filters))

# Saliency layer and "piercer"
saliency_conv = tf.layers.conv2d(dropout_placeholder, 64, (3,3), activation='relu')
pierce_conv = tf.layers.conv2d(saliency_conv, 1, (1,1), activation='relu')

# Calc prior & multiply with conv-nn output
prior = gaussian2d(17,std=0.5)
pierced = tf.math.multiply(pierce_conv,prior)
logits = tf.layers.dense(pierced)

# Loss
labels = []
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
minimize_op = optimizer.minimize(loss)

# Statistics 
predictions = tf.argmax(logits, axis=1)
correct_prediction = tf.equal(labels, predictions)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess = tf.Session()
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    dropout = feature_extractor.extract_features()

