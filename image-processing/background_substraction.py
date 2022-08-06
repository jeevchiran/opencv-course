from shutil import which
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fbbg = cv2.createBackgroundSubtractorMOG2(history = 10,detectShadows = False,varThreshold=10)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgMask = fbbg.apply(frame)
    cv2.imshow('Frame',frame)
    cv2.imshow('FG Mask',fgMask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
