import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np #helps in creating image matrix
import math
import time
import mediapipe as mp
import os


offset = 20 #for better view(line 19)
imagesize = 300 #to keep all image size constant

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("talk/keras_model.h5", "talk/labels.txt")
sequence = ""
prev = ""

count = 0
labels = ["A", "B", "H", "Y", "THANK YOU", "HELLO", "MY", "NAME", "IS", "WHAT", "OK", "NO", "HELP"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)


    def switch_case(argument):
        switcher = {
            0: "A",
            1: "B",
            2: "H",
            3: "Y",
            4: "THANK YOU",
            5: "HELLO",
        }
        return switcher.get(argument, "nothing")



    if hands:
        hand=hands[0]
        #this will give us the values of the highest tip and widest tip of our hand
        x, y, w, h = hand['bbox']

        #image white background
        imgwhite = np.ones((imagesize, imagesize, 3), np.uint8)*255

        imgcrop = img[y-offset : y+h+offset ,x-offset : x+w+offset] #this will give us bounding box

        imagecropshape = imgcrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imagesize/h
            wCal = math.ceil(k*w)
            wGap = math.ceil((imagesize-wCal)/2) #it is the gap we need to push forward to center the image

            imgResize = cv2.resize(imgcrop, (wCal, imagesize))
            imgResizeShape = imgResize.shape

            imgwhite[0 : imgResizeShape[0],wGap : wCal+wGap ] = imgResize #imgwhite[ height, width]=<..>

            prediction, index = classifier.getPrediction(imgwhite, draw=False)

            # print("prediction")
            # print(prediction)
            # print(index)


        else:
            k = imagesize/w
            hCal = math.ceil(k*h)
            hGap = math.ceil((imagesize-hCal)/2) #it is the gap we need to push forward to center the image
            imgResize = cv2.resize(imgcrop, (imagesize, hCal))
            imgResizeShape = imgResize.shape
            imgwhite[hGap:hCal+hGap, 0:imgResizeShape[1]] = imgResize
            prediction, index = classifier.getPrediction(imgwhite, draw=False)
        sequence = labels[index]
        delay = 0
        time.sleep(delay)




        cv2.putText(imgOutput, sequence, (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        # cv2.imshow("imageCrop", imgcrop)
        cv2.imshow("imageWhite", imgwhite)


    cv2.imshow("image", imgOutput)
    cv2.waitKey(1)