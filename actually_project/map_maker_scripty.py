# -*- coding: utf-8 -*-
# CV2 project submission by Angelie Kraft and Anton Volkov
# University of Hamburg, Summer 2019
# This script only serves the purpose of creating saliency prediction images (jpeg-encoded) on the test dataset
# This saliency predictor uses transfer learning on VGG and is strongly inspired by the architecture of Cornia et al. (2016)

import numpy as np
import os
import tensorflow as tf
import scipy
import tensorflow_probability as tfp
import cv2

tf.reset_default_graph()

WEIGHTS_PATH = '../cv2_data/vgg16-conv-weights.npz'
DATA_PATH = '../cv2_data/'
MODEL_PATH = '../cv2_data/models/'
RESULTS_PATH = '../cv2_data/results_angelie_kraft_anton_volkov/'

weights = np.load(WEIGHTS_PATH)
for k in weights.keys():
  print(k + 'shape: {}'.format(weights[k].shape))

def create_dataset_from_files(files, grey=False):
    def load_img(img):
      np_image = cv2.imread(img)
      if grey:
        np_image= cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        np_image = np.expand_dims(np_image, axis=-1).astype(np.float32)
      return np_image
    listing = os.listdir(files)
    res = np.array([load_img(os.path.join(files, img)) for i, img in enumerate(listing)])
    return res

cv2_validation_imgs = os.path.join(DATA_PATH, 'val/images')
cv2_validation_fixs = os.path.join(DATA_PATH, 'val/fixations')
cv2_testing = os.path.join(DATA_PATH, 'test/images')

print('Loading validation_X')
validation_X = create_dataset_from_files(cv2_validation_imgs)
print(validation_X.shape, validation_X.dtype)

print('Loading validation_y')
validation_y = create_dataset_from_files(cv2_validation_fixs, True)
print(validation_y.shape, validation_y.dtype)

print('Loading test_X')
test_X = create_dataset_from_files(cv2_testing)
print(test_X.shape, test_X.dtype)

images = tf.placeholder(tf.uint8, [1, 180, 320, 3]) # None to indicate a dimension can vary at runtime
gaussian = tf.placeholder(tf.float32, [None, 45, 80, 1])

# Calc 2d gaussian

def tf_diff(a):
    return a[1:]-a[:-1]

def gaussian2d(mean, std):
    x = tf.linspace(0., 1., 45+1)
    y = tf.linspace(0., 1., 80+1)
    tfd = tfp.distributions
    dist = tfd.Normal(loc=mean, scale=std)
    gaussian1d_x = tf_diff(dist.cdf(x))
    gaussian1d_y = tf_diff(dist.cdf(y))
    gaussian2d = tf.einsum('i,j->ij', gaussian1d_x, gaussian1d_y)
    gaussian2d = tf.reshape(gaussian2d, shape=(-1, 45, 80, 1))
    return gaussian2d

with tf.name_scope('preprocess') as scope:
  imgs = tf.image.convert_image_dtype(images, tf.float32) * 255.0
  mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32
                       , shape=[1 , 1, 1, 3], name='img_mean')
  imgs_normalized = imgs - mean

with tf.name_scope('conv1_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(imgs_normalized, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv1_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_2_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_2_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool1') as scope:
	pool1 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv2_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv2_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv2_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_2_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv2_2_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool2') as scope:
	pool2 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv3_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_2_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_2_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_3') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_3_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_3_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool3') as scope:
	pool3 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(1,1), padding='same')
  
with tf.name_scope('conv4_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv4_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=weights['conv4_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('concat') as scope:
  out = tf.concat([pool2, pool3, act], -1)

# Doesn't use dropout
#with tf.name_scope('dropout') as scope:
 # out = tf.keras.layers.Dropout(rate=1-0.5)(out)
  
with tf.name_scope('conv_sal_1') as scope:
  init = tf.initializers.glorot_normal()
  kernel = tf.Variable(init([3, 3, 896, 64]), name="kernel_1")
  biases = tf.Variable(tf.zeros([64,], tf.float32))
  conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')
  out = tf.nn.bias_add(conv, biases)
  act = tf.nn.relu(out, name=scope)
  
with tf.name_scope('conv_sal_2') as scope:
  init = tf.initializers.glorot_normal()
  kernel = tf.Variable(init([1, 1, 64, 1]), name="kernel_2")
  biases = tf.Variable(tf.zeros([1,], tf.float32))
  conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
  out = tf.nn.bias_add(conv, biases)
  saliency_raw = tf.nn.relu(out, name=scope)

with tf.name_scope('gaussian') as scope:
  shape = tf.convert_to_tensor([-1, 45, 80, 1])

  mean_1 = tf.Variable(initial_value=tf.constant(0.5), trainable=True, name="mean_1")
  std_1 = tf.Variable(initial_value=tf.constant(0.5), trainable=True, name="std_1")
  gaussian_1 = gaussian2d(mean_1, std_1)
  gaussian_1 = tf.reshape(gaussian_1, shape)

  mean_2 = tf.Variable(initial_value=tf.constant(0.), trainable=True, name="mean_2")
  std_2 = tf.Variable(initial_value=tf.constant(0.5), trainable=True, name="std_2")
  gaussian_2 = gaussian2d(mean_2, std_2)
  gaussian_2 = tf.reshape(gaussian_2, shape)

  mean_3 = tf.Variable(initial_value=tf.constant(1.), trainable=True, name="mean_3")
  std_3 = tf.Variable(initial_value=tf.constant(0.5), trainable=True, name="std_3")
  gaussian_3 = gaussian2d(mean_3, std_3)
  gaussian_3 = tf.reshape(gaussian_3, shape)

  stacked_gaussians = tf.stack([gaussian_1, gaussian_2, gaussian_3], axis=0)
  mean_of_gaussians = tf.reduce_mean(stacked_gaussians, axis=0)

  saliency_raw = tf.math.multiply(saliency_raw, mean_of_gaussians, name=None)
  max_value_per_image = tf.reduce_max(saliency_raw, axis=[1,2,3], keepdims=True)
  predicted_saliency = (saliency_raw / max_value_per_image)


import matplotlib.pyplot as plt

saver = tf.train.Saver()
test_filenames = ["".join((os.path.splitext(img)[0], '_prediction.jpg')) for img in os.listdir(cv2_testing)]

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())
  saver.restore(sess, os.path.join(MODEL_PATH, 'learnable_gaussian/trained_model-29'))

  for i in range(len(test_X[:5])):
    res = sess.run(predicted_saliency, feed_dict={images: np.expand_dims(test_X[i], 0)})
    img = tf.reshape(res, [45, 80, 1])
    img = tf.image.resize(img, [120, 320])
    img = tf.image.convert_image_dtype(img, tf.uint8)
    img_jpeg = tf.image.encode_jpeg(img, format='grayscale')
    img_path = os.path.join(RESULTS_PATH, test_filenames[i])
    print(img_path)
    fwrite = tf.io.write_file(img_path, img_jpeg.eval(), name="file_writer")
    sess.run(fwrite) 
    
