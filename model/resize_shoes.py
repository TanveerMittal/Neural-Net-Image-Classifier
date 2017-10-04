import imutils
import cv2
import glob

count = 0
for x in glob.glob("shoes2/*"):
    image = cv2.imread(x)
    cv2.imwrite("resized/image%d.jpg" % count, imutils.resize(image, width=1280, height=720))
    count += 1