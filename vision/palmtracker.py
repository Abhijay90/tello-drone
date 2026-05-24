#palmtracker.py

import cv2
import numpy as np

w,h = 640,480

def findPalm(img):
	imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lower_hsv = np.array([0,48,0])
	upper_hsv = np.array([20,255,255])
	mask1 = cv2.inRange(imgHSV,lower_hsv,upper_hsv)
	lower_hsv = np.array([160,48,0])
	upper_hsv = np.array([180,255,255])
	mask2 = cv2.inRange(imgHSV,lower_hsv,upper_hsv)
	mask = mask1 + mask2
	kernel = np.ones((5,5),np.uint8)
	dilatedMask = cv2.dilate(mask,kernel,iterations=2)
	erodedMask = cv2.erode(dilatedMask,kernel,iterations=1)
	contours,_ = cv2.findContours(erodedMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	cx,cy,area = 0,0,0
	if contours:
		c = max(contours,key=cv2.contourArea)
		if cv2.contourArea(c) > 500:
			x,y,w,h=cv2.boundingRect(c)
			cx=x+w//2
			cy=y+h//2
			area=w*h
			
	faceCascade = cv2.CascadeClassifier("vision/haarcascade_hand.xml")
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	hands = faceCascade.detectMultiScale(imgGray,1.1,5)
	
	hand_cx,hand_cy,hand_area = 0,0,0
	if len(hands) > 0:
		h = sorted(hands,key=lambda k:k[2]*k[3],reverse=True)[0]
		x,y,w,h = h
		hand_cx = x + w // 2
		hand_cy = y + h // 2
		hand_area = w * h
	
	if area > 0 and hand_area > 0:
		if hand_area > area:
			cx,cy,area = hand_cx,hand_cy,hand_area
		cx = cx * 0.5 + hand_cx * 0.5
		cy = cy * 0.5 + hand_cy * 0.5
		area = area * 0.5 + hand_area * 0.5
	elif hand_area > 0:
		cx,cy,area = hand_cx,hand_cy,hand_area
	elif area > 0:
		cx,cy = cx,cy
		
	cv2.rectangle(img,(cx-80,cy-80),(cx+80,cy+80),(0,255,0),2)
	cv2.line(img,(cx-40,cy),(cx+40,cy),(0,255,0),2)
	cv2.line(img,(cx,cy-40),(cx,cy+40),(0,255,0),2)
	
	return img,[cx,cy,area]
