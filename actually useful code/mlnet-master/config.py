import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 10
# number of rows of input images
shape_r = 180
# number of cols of input images
shape_c = 320
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))
# number of epochs
nb_epoch = 20

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = 'C:/Users/anton/Documents/Study/CV2_project/cv2_data/train/images'
# path of training maps
maps_train_path = 'C:/Users/anton/Documents/Study/CV2_project/cv2_data/train/fixations'
# number of training images
nb_imgs_train = 1200
# path of validation images
imgs_val_path = 'C:/Users/anton/Documents/Study/CV2_project/cv2_data/val/images'
# path of validation maps
maps_val_path = 'C:/Users/anton/Documents/Study/CV2_project/cv2_data/val/fixations'
# number of validation images
nb_imgs_val = 400