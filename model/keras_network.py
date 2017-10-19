from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
from keras import metrics
from keras.utils import to_categorical


batch_size = 64
num_classes = 1
epochs = 3
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'shoe_classifier.h5'

# The data, shuffled and split between train and test sets:
x_train = np.load("pickle/train_x.npy")
print("done reading images")
y_train = np.load("pickle/train_y.npy")
#y_train = to_categorical(y_train)
print("done indexing outputs")


x_test = np.load("pickle/test_x.npy")
y_test = np.load("pickle/test_y.npy")
#y_test = to_categorical(y_test)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

"""
i = keras.Input(x_train.shape)
v = keras.applications.VGG16(include_top=False)
v.trainable=False
o = v(i)
o = Dense(1, activation="sigmoid")(o)
model = keras.models.Model(inputs=[i], outputs=[o])
"""
opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy', metrics.binary_accuracy])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracyh:', scores[1])
