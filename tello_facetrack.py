# drone movement dependent on face tracking
from djitellopy import tello
from time import sleep
from vision.facetracker import findFace,trackface
import cv2

fbRange = [6200,6800]
pid=[0.4,0.4,0]
pError=0
w,h = 360,240
me = tello.Tello()
me.connect()

print(me.get_battery())

me.streamon()

me.takeoff()
sleep(2)

# me.send_rc_control(0,0,30,0) # go high to see face



drone=1

while True:
	if not drone:
		cap=cv2.VideoCapture(0) # for using webcam of system
		_, img=cap.read()
	else:
		img=me.get_frame_read().frame
	img=cv2.resize(img,(w,h))
	img,info= findFace(img)
	# trackface(me,info,w,pid,pError)
	# print("center",info[0],"Area",info[1]) #get the tracking face 
	cv2.imshow("output",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		print("initiate landing")
		cv2.destroyAllWindows()
		me.streamoff()
		me.land()
		break

# print("hello")

# me.takeoff()
# me.send_rc_control(0,30,0,0) #forward movement
# sleep(2)
# me.send_rc_control(0,-30,0,0) #backward movement
# sleep(2)
# me.send_rc_control(0,30,0,0) #momentem breaker
# sleep(1)
# me.send_rc_control(30,0,0,0) # move right
# sleep(2)
# me.send_rc_control(-30,0,0,0) # move left
# sleep(2)
# me.send_rc_control(30,0,0,0) #momentem breaker
# sleep(1)
# me.send_rc_control(0,0,-20,0) #momentem breaker
# sleep(2)
# me.send_rc_control(0,0,0,0)
# me.land()