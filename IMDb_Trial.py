import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.datasets import imdb


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path = 'imdb.npz', num_words = 10000)
# print(train_data.shape)
# print(train_labels.shape)


def vectorize(seqs, dim = 10000):
    ret = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        ret[i, seq] = 1
    return ret

x_train = vectorize(train_data)
x_test = vectorize(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# print(x_train.shape)
# print(y_train.shape)
# print(x_train[0])
# print(y_train[0])


model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

# tf.keras.utils.plot_model(model, to_file = 'imdb.png')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

print(history.history.keys())
history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('TVLoss')

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.figure(2)
plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('TVAccuracy')

plt.show()

#Large Model
model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(521, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

history_dict = history.history
l_val_loss = history_dict['val_loss']

#Small Model
model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

history_dict = history.history
s_val_loss = history_dict['val_loss']

#Contrast of orinial, large and small models
plt.figure(3, figsize=(10,6))
plt.plot(epochs, val_loss, 'b', label = 'Original Model')
plt.plot(epochs, l_val_loss, 'r', label = 'Large Model')
plt.plot(epochs, s_val_loss, 'g', label = 'Small Model')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.xticks(np.arange(0, 11, step = 1))
plt.savefig('ctrOLS')
plt.show()

#Regularization
from tensorflow.keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16,
                       kernel_regularizer=regularizers.l2(0.001),
                       activation = 'relu',
                       input_shape = (10000,)))
model.add(layers.Dense(16,
                       kernel_regularizer=regularizers.l2(0.001),
                       activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

history_dict = history.history
l2_model_loss = history_dict['val_loss']

#Contrast of orinial and L2-regularized model
plt.figure(4, figsize=(10, 6))
plt.plot(epochs, val_loss, 'b', label = 'Original Model')
plt.plot(epochs, l2_model_loss, 'r', label = 'L2-Regularized Model')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.xticks(np.arange(0, 11, step = 1))
plt.savefig('ctrOL2')
plt.show()

#Dropput
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 0.001),
              metrics = ['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

history_dict = history.history
drp_model_loss = history_dict['val_loss']

#Contrast of orinial and dropout model
plt.figure(5, figsize=(10, 6))
plt.plot(epochs, val_loss, 'b', label = 'Original Model')
plt.plot(epochs, drp_model_loss, 'r', label = 'Dropout Model')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.xticks(np.arange(0, 11, step = 1))
plt.savefig('ctrODrp')
plt.show()