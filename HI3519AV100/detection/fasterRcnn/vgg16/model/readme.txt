1. The detection types of FasterRcnn Vgg16 are: 
 class 0:background     class 1:car             class 2:van  
 class 3:truck          class 4:bus             class 5:buggy
 class 6:pedestrian     class 7:person_sitting  class 8:cyclist
 class 9:bike           class10:motorbike       class11:tricycle
2. vgg_roi_coordi_caffe.txt is needed by compiler. It come from caffe inference of this net with ref image, which is the output of proposal layer.
3. This example show that the reshape->softmax->reshape layers in RPN are implemented by nnie.
 