import glob
import cv2
import pickle

pickle.dump([cv2.imread(x) for x in glob.glob("./shoes/*")] + [cv2.imread(x) for x in glob.glob("./negatives/*")],
            open("images.pkl", "w+"))
pickle.dump([0 for x in glob.glob("./shoes/*")] + [1 for x in glob.glob("./negatives/*")], open("array.pkl", "w+"))
