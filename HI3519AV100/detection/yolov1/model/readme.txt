=======================================================================================================
yolo v1: https://pjreddie.com/darknet/yolov1/

caffe-yolo github: https://github.com/xingwangsfu/caffe-yolo/tree/master/

Please follow the guide of "caffe-yolo github" to Convert yolo's (.weight) files to caffemodel.
The svp simulator yolo v1 wk convert from yolo_small.caffemodel.

=======================================================================================================
1. The input image need to be preprocessed by multiplied by 1/255.
2. This yolov1 is trained by using the Pascal VOC set. It can detect 20 object which VOC covered£º
class 0: '__background__', 
class 1 to 4: 'aeroplane', 'bicycle', 'bird', 'boat',
class 5 to 8: 'bottle', 'bus', 'car', 'cat', 
class 9 to 12: 'chair','cow', 'diningtable', 'dog', 
class 13 to 16: 'horse','motorbike', 'person', 'pottedplant',
class 17 to 20:  'sheep', 'sofa', 'train', 'tvmonitor'.
