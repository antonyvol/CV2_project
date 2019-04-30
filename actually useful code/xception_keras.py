import keras
from keras.applications.xception import Xception

model = Xception(input_shape=(299, 299, 3), weights='imagenet', include_top=False)
model.summary()