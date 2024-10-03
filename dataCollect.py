import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Image/pinky"
counter = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
        if h>w:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            WidthGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, WidthGap:wCal+WidthGap] = imgResize
        if w>h:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            heightGap = math.ceil((imgSize-hCal)/2)
            imgWhite[heightGap:hCal+heightGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key=cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    if key==ord('q'):
        break