import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time
from cvzone.ClassificationModule import Classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/name"
counter = 0



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hands = hands[0]
        x, y, w, h = hands['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            wGap = math.ceil((imgSize-wCal)/2)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            hGap = math.ceil((imgSize - hCal) / 2)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            imgWhite[hGap:hGap+hCal, :] = imgResize


        cv2.imshow("Imagecrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        counter+=1
        print(counter)