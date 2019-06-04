import numpy as np
import os
import tensorflow as tf
import imageio

def load_train_data():
	training_img_directory = '/my/path/cv2_data/train/images'
	training_fixation_directory = '/my/path/cv2_data/train/fixations'

	train_imgs = np.zeros((1200, 180, 320, 3), dtype=np.uint8)
	train_fixations = np.zeros((1200, 180, 320, 1), dtype=np.uint8)
	for i in range(1, 1201):
		img_file = os.path.join(training_img_directory, '{:04d}.jpg'.format(i))
		fixation_file = os.path.join(training_fixation_directory, '{:04d}.jpg'.format(i))
		train_imgs[i-1] = imageio.imread(img_file)
		fixation = imageio.imread(fixation_file)
		train_fixations[i-1] = np.expand_dims(fixation, -1) # adds singleton dimension so fixation size is (180,320,1)
	
	return train_imgs, train_fixations

# Generator function will output one (image, target) tuple at a time,
# and shuffle the data for each new epoch
def data_generator(imgs, targets):
	while True: # produce new epochs forever
		# Shuffle the data for this epoch
		idx = np.arange(imgs.shape[0])
		np.random.shuffle(idx)

		imgs = imgs[idx]
		targets = targets[idx]
		for i in range(imgs.shape[0]):
			yield imgs[i], targets[i]

def get_batch_from_generator(gen, batchsize)
	batch_imgs = []
	batch_fixations = []
	for i in range(batchsize):
		img, target = gen.next()
		batch_imgs.append(img)
		batch_fixations.append(target)
	return np.array(batch_imgs), np.array(batch_fixations)

# Set up
num_batches = 10000 # you can experiment with this value, but remember training a large network requires a lot of iterations!
batchsize = # define your batch size

# load entire data to memory (this dataset is small, so we can do it)
train_imgs, train_fixations = load_train_data()

###
### At minimum, add preprocessing to convert image and target to tf.float32.
### Then enter your CNN definition.
### Name the target fixation map after preprocessing "fixations_normalized", 
### and name the output "saliency_raw" so they fit the code afterward.
### 

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
	

with tf.Session() as sess:
	writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
	sess.run(tf.global_variables_initializer())

	gen = data_generator(train_imgs, train_fixations)

	for b in range(num_batches):
		batch_imgs, batch_fixations = get_batch_from_generator(gen, batchsize)
		# add training here

