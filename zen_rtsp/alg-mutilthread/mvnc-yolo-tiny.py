import numpy
import cv2
import cv
import sys
import time
import ctypes
from PIL import Image
from datetime import datetime
from ObjectWrapper import *

def process(picture,width,height):
    cv_img =cv.CreateImage((width, height), cv2.IPL_DEPTH_8U, 3)
    cv.SetData(cv_img, picture, 3 * width)
    cv_mat = cv_img[:]
    image_to_classify = numpy.asarray(cv_mat)
    start = datetime.now()
    results = detector.Detect(image_to_classify)
    cv2.imwrite('test.jpg',image_to_classify)
    end = datetime.now()
    elapsedTime = end-start
    objs = []
    detectedNum = len(results)
    print("results="+ str(len(results)))
    if detectedNum > 0:
            for i in range(detectedNum):
                
                classID = results[i].objType
                left = results[i].left
                top = results[i].top
                right = results[i].right
                bottom = results[i].bottom
                confidence = results[i].confidence
    objs=objs+[ClassID,confidence,left,top,right-left+1,bottom-top+1]
    return objs
def init():
    global detector
    gf="graph_file/yolov2tiny.graph"
    detector = ObjectWrapper(gf)