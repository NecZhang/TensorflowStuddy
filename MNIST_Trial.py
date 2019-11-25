import tensorflow as tf
import numpy as np


# print(tf.__version__)
# print(tf.keras.__version__)


(x_train, y_train), (x_text, y_text) = tf.keras.datasets.mnist.load_data(path = 'mnist.npz')

x_train = x_train.reshape(60000, 784)
x_text = x_text.reshape(10000, 784)
x_train = x_train.astype('float32')
x_text = x_text.astype('float32')
x_train /= 255
x_text /= 255
# print(x_train.shape[0], 'train samples')
# print(x_text.shape[0], 'test samples')
#
# print(y_train)
# print(y_text)


from tensorflow.keras.utils import to_categorical

num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_text = to_categorical(y_text, num_classes)
# print(y_train)
# print(y_text)


from tensorflow.keras.layers import Dense

model = tf.keras.Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (784,), name = 'Dense_0'))
model.add(Dense(512, activation = 'relu', name = 'Dense_1'))
model.add(Dense(num_classes, activation = 'softmax', name = 'Dense_2'))

model.summary()


# from tensorflow.keras.utils import plot_model
#
# plot_model(model, to_file = 'mnist.png')


from tensorflow.keras.optimizers import RMSprop

model.compile(loss = 'categorical_crossentropy',
               optimizer = RMSprop(),
               metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 128, epochs = 10, verbose = 1, validation_data = (x_text, y_text))

score = model.evaluate(x_text, y_text, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist.h5')