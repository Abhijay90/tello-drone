import cv2
import matplotlib.pyplot as plt

class video_stream:
	def __init__(self,drone_obj):
		self.me=drone_obj
		self.me.streamon()

	def show_vid(self,chng_col_code=0):
		plt.figure()
		plt.ion()
		while True:
			img=self.me.get_frame_read().frame
			img=cv2.resize(img,(360,240))
			if chng_col_code:
				img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				plt.imshow(img_rgb)
			else:
				plt.imshow(img)
			plt.pause(0.01)
			plt.clf()
			# yield

if __name__=="__main__":
	from djitellopy import tello
	from time import sleep
	me = tello.Tello()
	me.connect()
	s=video_stream(me)
	s.show_vid()