from djitellopy import tello
from time import sleep
import cv2
# import matplotlib.pyplot as plt

me = tello.Tello()
me.connect()

me.streamon()

# plt.figure()
# plt.ion()

while True:
	img=me.get_frame_read().frame
	# cv2.startWindowThread()
	# img=cv2.resize(img,(360,240))
	# img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	# plt.imshow(img_rgb)
	# plt.pause(0.01)
	# plt.clf()
	cv2.imshow("Image",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
	