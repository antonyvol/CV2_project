import scipy
import numpy as np
from feature_extractor import extract_features_batch


def downscale(img, size):
	pass


def upscale(img, size):
	pass


np.save('train_data_features', extract_features_batch('C:/Users/anton/Documents/Study/CV2_project/cv2_data/train/images'))
print('Saved')
np.save('validation_data_features', extract_features_batch('C:/Users/anton/Documents/Study/CV2_project/cv2_data/val/images'))
print('Saved')
np.save('test_data_features', extract_features_batch('C:/Users/anton/Documents/Study/CV2_project/cv2_data/test/images'))
print('Saved')