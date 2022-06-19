from hashlib import new
import cv2
import numpy as np
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

myColors = [[35, 155, 103, 105, 255, 255]]
myColorValues = [[199, 147, 54]]


def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv2.circle(imgResult, (x, y), 5, myColorValues[count], cv2.FILLED)
        if(x!=0 and y!=0):
            newPoints.append([x,y,count])
        count+=1
    return newPoints


def getContours(img):
    contours, hierarchies = cv2.findContours(
        img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = 0,0,0,0
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area > 200):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, peri*0.02, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y

def drawOnCanvas(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)
myPoints =  [] # x,y, color
while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints,myColorValues)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
