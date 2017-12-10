from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
import numpy as np
import os
from keras import metrics
import gzip

batch_size = 5
num_classes = 1
epochs = 25
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'shoe_classifier.h5'

# The data, shuffled and split between train and test sets:
x_train = np.load(gzip.open("pickle/train_x.npy.gz"))
print("done reading images")
y_train = np.load(gzip.open("pickle/train_y.npy.gz"))
print("done indexing outputs")
print(str(len(x_train)) + " training samples")

x_test = np.load(gzip.open("pickle/test_x.npy.gz"))
y_test = np.load(gzip.open("pickle/test_y.npy.gz"))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy', metrics.binary_accuracy])


model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, callbacks=[CSVLogger("data.csv", append=True)] ,shuffle=True, validation_data=(x_test, y_test))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)



