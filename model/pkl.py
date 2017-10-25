import glob
import cv2
import pickle
from numpy import array
import gzip
import random

train_x = [cv2.imread(x) for x in glob.glob("./train/Athletic shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/bare feet/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/Dress shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/Formal shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/High heels/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/Running shoes/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/sneakers/*")] + \
            [cv2.imread(x) for x in glob.glob("./train/negatives/*")]
train_y = [1 for x in glob.glob("./train/Athletic shoes/*")] + \
            [1 for x in glob.glob("./train/bare feet/*")] + \
            [1 for x in glob.glob("./train/Dress shoes/*")] + \
            [1 for x in glob.glob("./train/Formal shoes/*")] + \
            [1 for x in glob.glob("./train/High heels/*")] + \
            [1 for x in glob.glob("./train/Running shoes/*")] + \
            [1 for x in glob.glob("./train/sneakers/*")] + \
            [0 for x in glob.glob("./train/negatives/*")]
validation_x = [cv2.imread(x) for x in glob.glob("./validate/positive/*")] + \
                [cv2.imread(x) for x in glob.glob("./validate/negative/*")]
validation_y = [1 for x in glob.glob("./validate/positive/*")] + [0 for x in glob.glob("./validate/negative/*")]

for i in range(len(train_x)):
    a = random.randint(0, len(train_x) - 1)
    b = random.randint(0, len(train_x) - 1)
    x = train_x[a]
    y = train_y[a]
    train_x[a] = train_x[b]
    train_y[a] = train_y[a]
    train_x[b] = x
    train_y[b] = y

pickle.dump(train_x, gzip.open("pickle/train_x.pkl.gz", "w+"))
pickle.dump(train_y, gzip.open("pickle/train_y.pkl.gz", "w+"))
pickle.dump(array(validation_x), gzip.open("pickle/test_x.npy.gz", "w+"))
pickle.dump(array(validation_y), gzip.open("pickle/test_y.npy.gz", "w+"))
