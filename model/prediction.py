from keras import models
from numpy import array
import numpy as np
import gzip
import glob
import cv2

model = models.load_model("saved_models/shoe_classifier.h5")
count = 0
for x in glob.glob("validate/positive/*"):
    print(x)
    if count % 2 == 0:
        input1 = cv2.imread(x)
    elif count % 2 ==1:
        input = [input1, cv2.imread(x)]
        input = array(input)
        print(model.predict(input))
    count += 1

print("negatives")
for x in glob.glob("validate/negative/*"):
    if count % 2 == 0:
        input1 = cv2.imread(x)
    elif count % 2 ==1:
        input = [input1, cv2.imread(x)]
        input = array(input)
        print(model.predict(input))
    count += 1
 