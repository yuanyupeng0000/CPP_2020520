1. The detection types of FasterRcnn Resnet18 are:  class 0:background  class 1:pedestrian  class 2:cyclist
2. resnet18_roi_coordi_caffe.txt is needed by compiler. It come from caffe inference of this net with ref image, which is the output of proposal layer.
3. This example show that the reshape->softmax->reshape layers in RPN are implemented by nnie.
