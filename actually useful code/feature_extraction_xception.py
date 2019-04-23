import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
from skimage import transform, io

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

x_in = tf.placeholder(tf.float32, shape=(1, 299, 299, 3))

block1_conv1 = tf.nn.conv2d(input=x_in, filter=(3, 3, 3, 32), strides=(1, 2, 2, 1), padding='VALID')




# -------------------------- code to test stuff in progress ----------------

with tf.Session() as sess:
	data = read_single_image('..\\cv2_data\\train\\images\\0001.jpg')
	x = sess.run(block1_conv1, feed_dict={x_in: data})
	print(x)


