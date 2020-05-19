from tensorflow.keras.layers import Layer, Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization, Activation, Lambda
import numpy as np
from stn import spatial_transformer_network as transformer

def SpatialTransformerModule(image):
	x = BatchNormalization()(image)
	x = Conv2D(32,kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32,kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32,kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32,kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Flatten()(x)
	x = Dense(100, activation='tanh')(x)
	x = AffineMatrix(name='mat')(x)
	out = SpatialTransformer(name='stn')([image, x])

	return out

class AffineMatrix(Layer):
	def __init__(self, **kwargs):
		super(AffineMatrix, self).__init__(**kwargs)

	def build(self, input_shape):
		self.x = Dense(6, name='mat')
		self.x.build(input_shape)
		weights = [np.zeros((input_shape[1], 6), dtype='float32'), np.array([1, 0, 0, 0, 1, 0], dtype='float32')]
		self.x.set_weights(weights)
		super(AffineMatrix, self).build(input_shape)

	def call(self, tensor):
		out = self.x(tensor)
		return out

class SpatialTransformer(Layer):
	def __init__(self, output_shape=None, **kwargs):
		super(SpatialTransformer, self).__init__(**kwargs)
		self.transformer_output_shape = output_shape

	def call(self, tensors):
		image = tensors[0]
		matrix = tensors[1]

		if self.transformer_output_shape is None:
			out = transformer(image, matrix)
		else:
			out = transformer(image, matrix, self.transformer_output_shape)

		return out