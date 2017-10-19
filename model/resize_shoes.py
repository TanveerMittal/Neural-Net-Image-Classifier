import imutils
import cv2
import glob

count = 0
"""
for x in glob.glob("shoes2/*"):
    image = cv2.imread(x)
    cv2.imwrite("resized/image%d.jpg" % count, imutils.resize(image, width=1280, height=720))
    count += 1
"""
image = cv2.imread("/Users/tanveermittal/Desktop/autonomous planner/images/in-the-zone-field.jpg")
cv2.imwrite("/Users/tanveermittal/Desktop/autonomous planner/images/image.jpg", imutils.resize(image, width=849, height=849))