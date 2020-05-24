from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from stm import SpatialTransformerModule
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data() # use mnist dataset

x_train = np.reshape(x_train,(-1,28,28,1))
x_test = np.reshape(x_test,(-1,28,28,1))

def MyModel():
	input = Input(shape=(28,28,1))
	x = SpatialTransformerModule(input, (28,28)) # the module
	x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
	x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
	x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = Flatten()(x)
	output = Dense(10, activation='softmax')(x)

	return Model(inputs=input, outputs=output)

model = MyModel()

model.load_weights('model.h5')

model.compile(
	optimizer=Adam(lr=1e-4), # if result gets worse, try to decrease the learning rate
	loss='categorical_crossentropy',
	metrics=['acc']
	)

model.summary()

model.fit(
	x = x_train,
	y = to_categorical(y_train),
	batch_size = 32,
	epochs = 1,
	validation_data = (x_test, to_categorical(y_test))
	)

model.save('model.h5')