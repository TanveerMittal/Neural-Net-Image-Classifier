import cv2
import imutils

vidcap = cv2.VideoCapture('negatives.MOV')

success,image = vidcap.read()

count = 0
success = True

while success:
  success,image = vidcap.read()
  print('Frame%d: ', success)
  cv2.imwrite("train/negatives/frame%d.jpg" % count, imutils.resize(image, width=1280, height=720) )    # save frame as JPEG file
  count += 1