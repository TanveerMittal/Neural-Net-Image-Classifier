import glob
import cv2
import pickle
from numpy import array
import gzip
import random
"""


"""
train_x = [cv2.imread(x) for x in glob.glob("./train/Athletic shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/Running shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/bare feet/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/Dress shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/Formal shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/High heels/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/sneakers/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/negatives/*")]
train_y = [1 for x in glob.glob("./train/Athletic shoes/*")] + \
            [1 for x in glob.glob("./train/Running shoes/*")] + \
            [1 for x in glob.glob("./train/sneakers/*")] + \
            [1 for x in glob.glob("./train/bare feet/*")] + \
            [1 for x in glob.glob("./train/Dress shoes/*")] + \
            [1 for x in glob.glob("./train/Formal shoes/*")] + \
            [1 for x in glob.glob("./train/High heels/*")] + \
            [0 for x in glob.glob("./train/negatives/*")]
validation_x = [cv2.imread(x) for x in glob.glob("./validate/positive/*")] + \
                [cv2.imread(x) for x in glob.glob("./validate/negative/*")]
validation_y = [1 for x in glob.glob("./validate/positive/*")] + [0 for x in glob.glob("./validate/negative/*")]

pickle.dump(array(train_x), gzip.open("pickle/train_x.npy.gz", "w+"))
pickle.dump(array(train_y), gzip.open("pickle/train_y.npy.gz", "w+"))
pickle.dump(array(validation_x), gzip.open("pickle/test_x.npy.gz", "w+"))
pickle.dump(array(validation_y), gzip.open("pickle/test_y.npy.gz", "w+"))
