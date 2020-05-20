1. The detection types of FasterRcnn Alexnet are:  class 0:background     class 1:pedestrian 
2. alex_roi_coordi_quant_1p.txt is needed by compiler. It come from caffe inference of this net with ref image, which is the output of proposal layer.
3. This example show that the reshape->softmax->reshape layers in RPN are implemented by nnie.