import cv2
import sys,os
img_name = sys.argv[1]
rename = "./rename.jpg"
while(True):
	os.system('cp ' + img_name + ' ' + rename)
	if(69040 != os.path.getsize(rename)):
		continue
	img = cv2.imread(rename)
	cv2.imshow('test', img)
	cv2.waitKey(100)
