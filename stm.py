from tensorflow.keras.layers import Layer, Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization, Activation, Lambda
import numpy as np
from stn import spatial_transformer_network as transformer

def SpatialTransformerModule(image, output_shape):
	x = BatchNormalization(name='stm_bn_0')(image)
	x = Conv2D(32,kernel_size=(3,3), padding='same', name='stm_conv_1')(x)
	x = BatchNormalization(name='stm_bn_1')(x)
	x = Activation('relu', name='stm_relu_1')(x)
	x = Conv2D(32,kernel_size=(3,3), padding='same', name='stm_conv_2')(x)
	x = BatchNormalization(name='stm_bn_2')(x)
	x = Activation('relu', name='stm_relu_2')(x)
	x = Conv2D(32,kernel_size=(3,3), padding='same', name='stm_conv_3')(x)
	x = BatchNormalization(name='stm_bn_3')(x)
	x = Activation('relu', name='stm_relu_3')(x)
	x = Conv2D(32,kernel_size=(3,3), padding='same', name='stm_conv_4')(x)
	x = BatchNormalization(name='stm_bn_4')(x)
	x = Activation('relu', name='stm_relu_4')(x)
	x = Flatten(name='stn_flat')(x)
	x = Dense(100, activation='tanh', name='stm_dense_1')(x)
	x = AffineMatrix(name='stm_mat')(x)
	out = SpatialTransformer(output_shape, name='stm_st')([image, x])

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
	def __init__(self, output_shape, **kwargs):
		super(SpatialTransformer, self).__init__(**kwargs)
		self.transformer_output_shape = output_shape

	def call(self, tensors):
		image = tensors[0]
		matrix = tensors[1]

		out = transformer(image, matrix, self.transformer_output_shape)

		return out

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'output_shape': self.transformer_output_shape
			})
		return config