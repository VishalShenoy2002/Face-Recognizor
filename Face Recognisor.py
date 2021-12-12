import cv2
import numpy as np
import os

def faceDetection(img):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)

    return faces



if __name__=="__main__":
    img=cv2.VideoCapture(0)
    while True:
        # bimg=cv2.imread("E:\\Programs\\Projects\\Computer Vision\\Picture as Resource\\king.jpeg")
        success,bimg=img.read()
        
        for (x,y,w,h) in faceDetection(bimg):
            cv2.rectangle(bimg,(x,y),(x+w,y+h),(255,0,255),thickness=7)


        
        cv2.imshow('FaceDetection',cv2.flip(bimg,1))
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
            cv2.destroyAllWindows()
        