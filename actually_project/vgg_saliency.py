# -*- coding: utf-8 -*-
"""vgg_saliency.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13jcTEhoAtzd8_pEq8SEJjzs5RsGoXIiD
"""

# from google.colab import drive
# drive.mount('/content/drive/')

import numpy as np
import os
import tensorflow as tf
# from tensorboardcolab import *
import scipy
import cv2

tf.reset_default_graph()

# !pip3 install opencv-python

# tf.test.gpu_device_name() # should print '/device:GPU:0' if Runtime is properly set to GPU

WEIGHTS_PATH = '../cv2_data/vgg16-conv-weights.npz'
DATA_PATH = '../cv2_data/'
MODEL_PATH = '../cv2_data/models/'

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
    res = np.array([load_img(os.path.join(files, img)) for i, img in enumerate(listing) if i % 5 == 0]) # every 5th image
    return res

# cv2_data_path = '/content/drive/My Drive/CV/cv2_data/' # replace with your path
cv2_training_imgs = os.path.join(DATA_PATH,'train/images')
cv2_training_fixs = os.path.join(DATA_PATH,'train/fixations')
cv2_validation_imgs = os.path.join(DATA_PATH, 'val/images')
cv2_validation_fixs = os.path.join(DATA_PATH, 'val/fixations')
cv2_testing = os.path.join(DATA_PATH, 'test/images')

print('Loading train_X')
train_X = create_dataset_from_files(cv2_training_imgs)
print(train_X.shape, train_X.dtype)

print('Loading train_y')
train_y = create_dataset_from_files(cv2_training_fixs, True)
print(train_y.shape, train_y.dtype)

print('Loading validation_X')
validation_X = create_dataset_from_files(cv2_validation_imgs)
print(validation_X.shape, validation_X.dtype)

print('Loading validation_y')
validation_y = create_dataset_from_files(cv2_validation_fixs, True)
print(validation_y.shape, validation_y.dtype)

print('Loading test_X')
test_X = create_dataset_from_files(cv2_testing)
print(test_X.shape, test_X.dtype)

images = tf.placeholder(tf.uint8, [None, 180, 320, 3]) # None to indicate a dimension can vary at runtime
labels = tf.placeholder(tf.int64, [None, 180, 320, 1])

with tf.name_scope('preprocess') as scope:
  imgs = tf.image.convert_image_dtype(images, tf.float32)
  mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32
                       , shape=[1, 1, 1, 3], name='img_mean')
  imgs_normalized = imgs - mean

with tf.name_scope('conv1_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(imgs_normalized, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv1_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_2_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_2_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool1') as scope:
	pool1 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv2_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv2_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv2_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_2_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv2_2_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool2') as scope:
	pool2 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv3_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_2_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_2_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_3') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_3_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_3_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool3') as scope:
	pool3 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(1,1), padding='same')
  
with tf.name_scope('conv4_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv4_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv4_1_b'], trainable=True, name="biases")
	conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('concat') as scope:
  out = tf.concat([pool2, pool3, act], -1)

with tf.name_scope('dropout') as scope:
  out = tf.keras.layers.Dropout(rate=1-0.5)(out)
  
with tf.name_scope('conv_sal_1') as scope:
  init = tf.initializers.glorot_normal()
  kernel = tf.Variable(init([3, 3, 896, 64]), name="kernel_1")
  conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')
  act = tf.nn.relu(conv, name=scope)
  
with tf.name_scope('conv_sal_2') as scope:
  init = tf.initializers.glorot_normal()
  kernel = tf.Variable(init([1, 1, 64, 1]), name="kernel_2")
  conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
  saliency_raw = tf.nn.relu(conv, name=scope)

with tf.name_scope('preprocess_labels') as scope:
  fixations_normalized = tf.image.convert_image_dtype(labels, tf.float32)

with tf.name_scope('loss') as scope:
	# normalize saliency
  max_value_per_image = tf.reduce_max(saliency_raw, axis=[1,2,3], keepdims=True)
  predicted_saliency = (saliency_raw / max_value_per_image)
  
  # Prediction is smaller than target, so downscale target to same size
  target_shape = predicted_saliency.shape[1:3]
  target_downscaled = tf.image.resize_images(fixations_normalized, target_shape)

	# Loss function from Cornia et al. (2016) [with higher weight for salient pixels]
  alpha = 1.01
  weights = 1.0 / (alpha - target_downscaled)
  loss = tf.losses.mean_squared_error(labels=target_downscaled, 
										predictions=predicted_saliency, 
										weights=weights)

	# Optimizer settings from Cornia et al. (2016) [except for decay]
  optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True)
  minimize_op = optimizer.minimize(loss)

# with tf.name_scope('accuracy') as scope:
#   acc, acc_op = tf.metrics.accuracy(labels=target_downscaled, 
#                                     predictions=predicted_saliency)

def data_shuffler(imgs, targets):
	while True: # produce new epochs forever
		# Shuffle the data for this epoch
		idx = np.arange(imgs.shape[0])
		np.random.shuffle(idx)

		imgs = imgs[idx]
		targets = targets[idx]
		for i in range(imgs.shape[0]):
			yield imgs[i], targets[i]

def get_batch(gen, batchsize):
	batch_imgs = []
	batch_fixations = []
	for i in range(batchsize):
		img, target = next(gen)
		batch_imgs.append(img)
		batch_fixations.append(target)
	return np.array(batch_imgs), np.array(batch_fixations)

batchsize = 32
num_batches = 100

saver = tf.train.Saver()
for i, var in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))

with tf.Session() as sess:
  # writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
  sess.run(tf.global_variables_initializer())

  gen = data_shuffler(train_X, train_y)

  for b in range(num_batches):
    batch_imgs, batch_fixations = get_batch(gen, batchsize)
    idx = np.random.choice(train_X.shape[0], batchsize, replace=False) # sample random indices
    _, batch_loss = sess.run([minimize_op, loss],
      feed_dict={images: train_X[idx,...], labels: train_y[idx]})

    print('Batch {} done: batch loss {}'.format(b, batch_loss))
    save_path = saver.save(sess, os.path.join(MODEL_PATH,'trained_model'), global_step=b)

      
  # run testing in smaller batches so we don't run out of memory.
  test_batch_size = 100
  num_test_batches = validation_X.shape[0]/test_batch_size
  test_losses = []
  test_accs = []

  for test_batch in range(int(num_test_batches)):
    start_idx = test_batch * test_batch_size
    stop_idx = start_idx + test_batch_size
    test_idx = np.arange(start_idx, stop_idx)

    test_loss = sess.run([loss], 
                                        feed_dict={images: validation_X[test_idx], labels: validation_y[test_idx]})
    print('Test batch {} done: batch loss {}'.format(test_batch, test_loss))
    test_losses.append(test_loss)
    #test_accs.append(test_acc)

    print('Test loss: {} -- test accuracy: {}'.format(np.average(test_losses), np.average(test_accs)))

  # we could save at end of training, for example



"""###TODO:
* check if loss is calculated correctly YES
* add prior (layer)
* add accuracy calculation NO
* hyperparameter optimzation
"""

