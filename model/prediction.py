from keras import models
from numpy import array
import numpy as np

model = models.load_model("saved_models/shoe_classifier.h5")

input = np.load("pickle/test_x.npy")

#print input

print model.predict(input)