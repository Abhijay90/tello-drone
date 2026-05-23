#facetracker.py


import cv2
import numpy as np


fbRange = [6200,6800]
pid=[0.4,0.4,0]
pError=0
w,h = 360,240

def findFace(img):
	faceCascade = cv2.CascadeClassifier("vision/haarcascade_frontalface_default.xml")
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(imgGray,1.2,8)

	myFaceListC = []
	myFaceListArea = []
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		cx=x+w//2 # equivalent to int x+int(w/2), it is floor division.
		cy=y+h//2
		area= w*h
		cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)
		myFaceListC.append([cx,cy])
		myFaceListArea.append(area)
	if len(myFaceListC) !=0:
		i=myFaceListArea.index(max(myFaceListArea)) #find highest area image
		return img,[myFaceListC[i],myFaceListArea[i]]
	else:
		return img,[[0,0],0]


def trackface(me,info,w,pid,pError):
	area= info[1]
	x,y=info[0]
	error = x-w//2
	speed= pid[0]*error+pid[1]*(error-pError)
	speed=int(np.clip(speed,-100,100)) # angular change , focusing on face
	fb=0

	if area>fbRange[0] and area<fbRange[1]: #do not move
		fb = 0
	if area>fbRange[1]: #move back it's too close
		fb=-20 
	elif area<fbRange[0] and area!=0: #move forward it's too far
		fb=20

	# print(speed,fb)

	if x==0:
		speed= 0
		error = 0

	me.send_rc_control(0,fb,0,speed)
	return error
