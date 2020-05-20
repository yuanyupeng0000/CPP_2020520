1. The detection types of RFCN are:  
 class 0:background     class 1:plane           class 2:bicycle  
 class 3:bird           class 4:boat            class 5:bottle
 class 6:bus            class 7:car             class 8:cat
 class 9:chair          class10:cow             class11:diningtable
 class 12:dog           class13:horse           class14:motorbike
 class 15:person        class16:pottedplant     class17:sheep
 class 18:sofa          class19:train           class20:tvmonitor
 
2. bbox_cls_rois.txt and bbox_loc_rois.txt are needed by compiler. They come from caffe inference of this net with ref image, which is the output of proposal layer.
3. This example show that the reshape->softmax->reshape layers in RPN are implemented by nnie.