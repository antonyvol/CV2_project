# -*- coding: utf-8 -*-
# CV2 project submission by Angelie Kraft and Anton Volkov
# University of Hamburg, Summer 2019
# This saliency predictor uses transfer learning on VGG and is strongly inspired by the architecture of Cornia et al. (2016)

import numpy as np
import os
import tensorflow as tf
import scipy
import cv2
import tensorflow_probability as tfp

tf.reset_default_graph()

WEIGHTS_PATH = '../cv2_data/vgg16-conv-weights.npz'
DATA_PATH = '../cv2_data/'
#MODEL_PATH = '../cv2_data/models/'
MODEL_PATH_SAVE = '../cv2_data/models/learnable_gaussian/'

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

images = tf.placeholder(tf.uint8, [None, 180, 320, 3]) 
labels = tf.placeholder(tf.uint8, [None, 180, 320, 1])
is_training = tf.placeholder(tf.bool)

with tf.name_scope('preprocess') as scope:
  
  imgs = tf.image.convert_image_dtype(images, tf.float32) * 255.0
  mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32
                      , shape=[1, 1, 1, 3], name='img_mean')
  imgs_normalized = imgs - mean
  fixations_normalized = tf.image.convert_image_dtype(labels, tf.float32)
 
with tf.name_scope('conv1_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([64,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(imgs_normalized, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv1_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_2_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([64,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool1') as scope:
	pool1 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv2_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([128,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv2_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_2_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([128,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool2') as scope:
	pool2 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv3_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([256,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_2_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([256,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_3') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_3_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([256,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool3') as scope:
	pool3 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(1,1), padding= 'same')
  
with tf.name_scope('conv4_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv4_1_W'], trainable=True, name="weights")
	biases = tf.Variable(initial_value=tf.zeros([512,], tf.float32), trainable=True, name="biases")
	conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('concat') as scope:
  out = tf.concat([pool2, pool3, act], -1)

with tf.name_scope('dropout') as scope:
  out = tf.layers.dropout(out, rate=0.2, training=is_training, name="dropout")
  
with tf.name_scope('conv_sal_1') as scope:
  init = tf.initializers.glorot_normal()
  kernel = tf.Variable(init([3, 3, 896, 64]), trainable=True, name="kernel_1")
  biases = tf.Variable(tf.zeros([64,], tf.float32), trainable=True)
  conv = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding='SAME')
  out = tf.nn.bias_add(conv, biases)
  act = tf.nn.relu(out, name=scope)
  
with tf.name_scope('conv_sal_2') as scope:
  init = tf.initializers.glorot_normal()
  kernel = tf.Variable(init([1, 1, 64, 1]), trainable=True, name="kernel_2")
  biases = tf.Variable(tf.zeros([1,], tf.float32), trainable=True)
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
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9, use_nesterov=True)
  minimize_op = optimizer.minimize(loss)

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
num_batches = 2000
mean_std = tf.constant([0.5])

saver = tf.train.Saver()
for i, var in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))

l_summary = tf.summary.scalar(name="loss", tensor=loss)
m_summary = tf.summary.scalar(name="mean_1", tensor=mean_1)
s_summary = tf.summary.scalar(name="std_1", tensor=std_1)
f_img_summary = tf.summary.image(name="fixation", tensor=tf.image.convert_image_dtype(labels, tf.float32))
i_img_summary = tf.summary.image(name="input", tensor=imgs_normalized)
pred_img_summary = tf.summary.image(name="predicted_saliency", tensor=predicted_saliency)
targ_img_summary = tf.summary.image(name="learned_prior", tensor=mean_of_gaussians)
w_summary = tf.summary.scalar(name="weights", tensor=tf.reduce_min(weights))

with tf.Session() as sess:
  summary_writer = tf.summary.FileWriter(logdir='./learnable_prior_logs/', graph=sess.graph)
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, os.path.join(MODEL_PATH_SAVE, 'trained_model-29'))

  gen = data_shuffler(train_X, train_y)

  for b in range(num_batches):
    batch_imgs, batch_fixations = get_batch(gen, batchsize)
    idx = np.random.choice(train_X.shape[0], batchsize, replace=False)
    _, batch_loss, ls, ms, ss, fs, i_s, ps, ts, ws = sess.run([minimize_op, loss, l_summary, m_summary, s_summary, f_img_summary, i_img_summary, pred_img_summary, targ_img_summary, w_summary], 
    feed_dict={images: train_X[idx,...], labels: train_y[idx], is_training: True}) 

    summary_writer.add_summary(ls, global_step=b)
    summary_writer.add_summary(ms, global_step=b)
    summary_writer.add_summary(ss, global_step=b)
    summary_writer.add_summary(fs, global_step=b)
    summary_writer.add_summary(i_s, global_step=b)
    summary_writer.add_summary(ps, global_step=b)
    summary_writer.add_summary(ts, global_step=b)
    summary_writer.add_summary(ws, global_step=b)

    print('Batch {} done: batch loss {}'.format(b, batch_loss))
    save_path = saver.save(sess, os.path.join(MODEL_PATH_SAVE,'trained_model'), global_step=b)
      
  # run testing in smaller batches so we don't run out of memory.
  test_batch_size = 32
  num_test_batches = validation_X.shape[0]/test_batch_size
  test_losses = []
  test_accs = []

  for test_batch in range(int(num_test_batches)):
    start_idx = test_batch * test_batch_size
    stop_idx = start_idx + test_batch_size
    test_idx = np.arange(start_idx, stop_idx)

    test_loss = sess.run([loss], feed_dict={images: validation_X[test_idx], labels: validation_y[test_idx], is_training: False})
    print('Test batch {} done: batch loss {}'.format(test_batch, test_loss))
    test_losses.append(test_loss)

    print('Test loss: {} -- test accuracy: {}'.format(np.average(test_losses), np.average(test_accs)))


