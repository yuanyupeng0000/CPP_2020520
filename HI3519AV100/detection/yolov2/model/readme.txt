1. YOLOv2 is based on Darknet. It contains some new layer such as passthrough, reorg, yolov2loss.
2. The input image's pixel need to be preprocessed by subtracting [105, 117, 123].
3. This YOLOv2 model is based on KITTI data set. It's trained by using only 5000 images, which contains the following 5 class of objects:
class 0: Car
class 1: Van
class 2: Truck
class 3: Pedestrian
class 4: Cyclist
4. It's not used any ImageNet pre-trained model. The effect of this model is not very good. It's just for sample. 
