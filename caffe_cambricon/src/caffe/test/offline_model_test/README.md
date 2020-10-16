# MLU Offline Examples: using Cambricon Neuware Runtime API

Cambricon Neuware Runtime (CNRT) provides the ability to run neural networks without the need of deep learning frameworks.

## environment
```
Ubuntu16.04
MLU270 / MLU220 M.2
```

## install
The first execution requires the installation of the runtime dependency library.
```
  sudo apt-get install libopencv-dev  libgflags-dev libgoogle-glog-dev libboost-all-dev
  pip install Cython
  pip install pycocotools  
```

## Usage
```
  ./run_offline.sh
```
Select the test case to run according to the prompt.
Parameter:
  1. Hardware Device: Specify the hardware device and run configuration.
  2. network: Select the offline model to use;
  3. the number of iterations. Calculate 64 images at a time;

The output for one of the networks should look like this:
```
--------------------------
running multicore offline test...
using model: models/resnet50_MLU270_16batch_16core.cambricon
Global accuracy : 
top1: 0.75 (4800/6400)
top5: 0.875 (5600/6400)
throughput: 5121.56
End2end throughput fps: 2705.74
Latency: 3124.05
Interval time:  ave: 2532.28  min: 0  max: 15880
```

## Test case
```
network       
resnet50      
mobilenet_v2
ssd_vgg16 
faster-rcnn   
yolov3        
yolov2        
mtcnn
```
Model accuracy and perfomance data are recorded in the data.xlsx file.

Support for recording hardware elapsed time:
The runtime sets the environment variables:
```  
export INTERVAL_TIME=true
```

## Neuware version
```
sdk v1.0
20191226: release_mlu270mlu220_1.3.0
  caffe master commit 73524ce
  sopa  master commit 5b67ded 
```
```
sdk v1.1
20200215:
  caffe master 0eee1a56
  sopa  master da522252
  cnplugin master 44d0f1c1
```
```
sdk v1.2
20200307:
  caffe master 71bd5d7
  sopa release_mlu270mlu220_v1.2.5 c1e5185
  cnplugin release_mlu270mlu220_v1.2.5 5873d80
```

