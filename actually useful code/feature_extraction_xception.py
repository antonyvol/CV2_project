import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
from skimage import transform, io
import h5py

# ------------ READ AND PREPROCESS (FOR XCEPTION) IMAGE BY IMAGE --------------
# ..\\cv2_data\\train\\images\\0001.jpg
def read_single_image(path):
	data = io.imread(path).astype('float32')
	data = np.swapaxes(data, -3, -2)
	# preprocessing for Xception
	data /= 255. 
	data -= 0.5
	data *= 2.
	data = transform.resize(data, (299, 299, 3))
	data = np.reshape(data, (1, 299, 299, 3))

	return data


# ------------------ ACTUALLY XCEPTION ----------------------

weights = h5py.File('xception_weights_tf_dim_ordering_tf_kernels.h5', 'r')

# Get the data
# data = weights['convolution2d_1'].values()
# print(data)


x_in = tf.placeholder(tf.float32, shape=(1, 299, 299, 3))

cur_weights = []
weights['convolution2d_1'].items()
print(cur_weights)
# block1_conv1_filters = tf.Variable(, shape=)
# block1_conv1 = tf.nn.conv2d(input=x_in, filter=block1_conv1_filters, strides=(1, 2, 2, 1), padding='VALID')




# -------------------------- code to test stuff in progress ----------------
# init_op = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	data = read_single_image('..\\cv2_data\\train\\images\\0001.jpg')
# 	x = sess.run(block1_conv1, feed_dict={x_in: data})
# 	print(x)


