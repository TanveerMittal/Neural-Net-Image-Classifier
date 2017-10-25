from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
from tflearn.data_utils import shuffle
from keras import metrics
import gzip


batch_size = 50
num_classes = 1
epochs = 3
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'shoe_classifier.h5'

# The data, shuffled and split between train and test sets:
x_train = np.load(gzip.open("pickle/train_x.npy.gz"))
print("done reading images")
y_train = np.load(gzip.open("pickle/train_y.npy.gz"))
print("done indexing outputs")

x_train, y_train = shuffle(x_train, y_train)
x_train = np.array(x_train)
y_train = np.array(y_train)


x_test = np.load(gzip.open("pickle/test_x.npy.gz"))
y_test = np.load(gzip.open("pickle/test_y.npy.gz"))

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

def batch_generator(x, y, batch_size=64):
    while(True):
        batch = {}
        for i in range(x.__len__()):
            batch += {x[i]:y[i]}
            if i%batch_size == 0 or i == x.__len__() - 1:
                yield ([n for n in batch], [batch[b] for b in batch])
                batch = {}

model.fit_generator(batch_generator(x_train, y_train), steps_per_epoch=10, validation_data=(x_test, y_test), epochs=epochs)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracyh:', scores[1])

