import numpy as np
import cv2
import time

# CASC_PATH = "haarcascade_frontalface_default.xml"

# Create cascade
# faceCascade = cv2.CascadeClassifier(CASC_PATH)
# Capture from camera
# cap = cv2.VideoCapture(0)
largestBox = (0,0,0,0)
largest = 0

def zoom(img, zoom_factor=2):
    return cv2.resize(img, (512,512), fx=zoom_factor, fy=zoom_factor,interpolation = cv2.INTER_NEAREST)

boi = [(188,128), (384,256)]

canvas = np.zeros((512,512,3), np.uint8)
cv2.rectangle(canvas, (50,50), (100,125), (255,0,0), 5)
cv2.circle(canvas, (125, 125), 65, (15,75,50), 5) 
cv2.rectangle(canvas, boi[0], boi[1], (255,255,0), 5)
zoom_factor = 0.1
while True:
    time.sleep(1)
    if(zoom_factor >= 2):
        zoom_factor = 0.1
        canvas = np.zeros((512,512,3), np.uint8)
        cv2.rectangle(canvas, (50,50), (100,125), (255,0,0), 1)
        cv2.circle(canvas, (125, 125), 65, (15,75,50), 1) 
        cv2.rectangle(canvas, boi[0], boi[1], (255,255,0), 1)
    canvas = cv2.resize(canvas, (0,0), fx=zoom_factor, fy=zoom_factor)
    zoom_factor = zoom_factor + 0.5

    # (boxX, boxY, boxW, boxH) = boi.dim
    # distX1 = boxX
    # distY1 = boxY                               # dist refers to the distances in front of and
    # distX2 = 1024 - distX1 - boxW        # behind the face detection box
    # distY2 = 1024 - distY1 - boxH       # EX: |---distX1----[ :) ]--distX2--|

    # # Equalize x's and y's to shortest length
    # if distX1 > distX2:
    #     distX1 = distX2
    # if distY1 > distY2:
    #     distY1 = distY2

    # distX = distX1      # Set to an equal distance value
    # distY = distY1

    # # Trim sides to match original aspect ratio
    # centerX = distX + (boxW / 2.0)
    # centerY = distY + (boxH / 2.0)
    # distsRatio = centerX / centerY

    # Read image
    # _, img = cap.read()

    # Approximate phone camera resolution testing
    # widOff = int(img.shape[1] / 3.0)
    # img = img[0:img.shape[0], widOff:img.shape[1] - widOff]

    # Grayscale image for facial box detection
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     boxes = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.22,
#         minNeighbors=8,
#         minSize=(60,60),
#     )
#     print(largestBox)
    
#     for (x,y,w,h) in boxes:
#         area = (w-x)*(h-y)
#         if largest < area:
#             largest = area
#             largestBox = (x,y,w,h)
#         cv2.rectangle(img,(largestBox[0],largestBox[1]),(largestBox[0]+largestBox[2],largestBox[1]+largestBox[3]),(0,0,255),2)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    
    # Using resizeWindow()
    cv2.resizeWindow("img", 512, 512)
    cv2.imshow('img', canvas)  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break  

# cap.release()