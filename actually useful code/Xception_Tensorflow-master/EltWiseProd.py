from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf


class EltWiseProd(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EltWiseProd, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert isinstance(input_shape, list)
        #if input_shape[-1] is None:
        #    raise ValueError('Axis ' +  + ' of '
        #                     'input tensor should have a defined dimension '
        #                     'but the layer received an input with shape ' +
        #                     str(input_shape) + '.')
        #self.input_spec = InputSpec(ndim=len(input_shape),
        #                            axes=dict(list(enumerate(input_shape[1:], start=1))))
        
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim), initializer='uniform' trainable=True)

        super(EltWiseProd, self).build(input_shape)  # Be sure to call this at the end
    #    self.trainable = False

    def call(self, x):
        #assert isinstance(x, list)
        ls = np.array([])
        tf.map_fn(lambda y: np.append(ls, y), x)
        a, b = ls
        return SparseTensor(x.indices, np.multiply(a, b))

    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


        