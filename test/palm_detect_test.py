#palm_detect_test.py

import cv2
from vision.palmtracker import findPalm

def main():
	cap = cv2.VideoCapture(0)
	cap.set(3,640)
	cap.set(4,480)
	
	while True:
		success, img = cap.read()
		if not success:
			break
			
		img, info = findPalm(img)
		
		cv2.putText(img,f"Status: DETECTED" if info[2] > 0 else "Status: NO HAND",
			(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0) if info[2] > 0 else (0,0,255),2)
		cv2.putText(img,f"FPS:{int(cap.get(cv2.CAP_PROP_FPS))}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
		
		print(f"Center: ({info[0]},{info[1]}) Area: {info[2]}")
		
		cv2.imshow("Palm Detection",img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
