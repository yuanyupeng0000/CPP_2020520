# Caffe AI Module

## Introduction

This repository provides some offline models for the Caffe framework to run on MLU hardware devices.
These offline models were generated using the CambriconCaffe tool.

## Prerequisites

The model execution environment supports dependencies on the following environments.

1) X86 Ubuntu16.04 MLU220
```
  Executing a program requires installing a program dependent library:
  sudo apt-get install libopencv-dev libgflags-dev libgoogle-glog-dev libboost-all-dev
  
  export PLATFORM=x86
```

2) aarch64 MLU220
```
  export PLATFORM=aarch64
  export THIRD_PARTY=${neuware_path}/neuware/AI_module/thirdparty
```

## Directory Layout
| Path                            | Description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| bin                             | This path holds the executable files that allow the offline model to run. |
| data                            | This path contains some images from the dataset.                          |
| examples                        | This path provides the offline model execution program source code.       | 
| model_zoo                       | This path provides some offline models of common networks                 |
| model_zoo/classification        | The model in this directory is a classification network model.            |
| model_zoo/object_detection      | The model in this directory is the target detection model.                |
| model_zoo/face_detection        | The model in this directory is the target detection model.                |
| model_zoo/semantic_segmentation | The model in this directory is the image segmentation model.              |
| model_zoo/ video                | The model in this directory is video processing model.                    | 

## Offline Modle Description

#### Input Data Description:
The input image should be pre-processed to resize the BGRA fout-channel image of the specified size into the model input.

Image format: BGRA; Data dimension: NHWC; Data type: UINT8
#### Output Data Description:
The output will result in the inference of the corresponding function.

Output dimension: NHWC; Data type: FLOAT16

## Running 
There is a corresponding cambricon file in each model path in modelzoo, which is the MLU offline model file.

In the execution path, the run_offline.sh script will reason about the executable offline model, and the reasoning
result will be terminal printed or the resulting picture will be generated in the outputfolder in the execution path.
```
  ./run_offline.sh
```

The final output of one of the network should be printed like this:
```
  Global accuracy:
  top1: 0.73 (47/64)
  top5: 0.875 (56/64)
  throughput: 721.745
  Latency: 22168.50
```

'throughput' means the rate calculated by hardware equipment for the task execution: the number of pictures per_second.

For classification networks, the above results are generally printed on the terminal, recording the accuracy and performance.

For detection or image segmentation, the network will generate results that can be viewed in the output folder of the
corresponding model path, and performance records will be printed on the terminal at the end of the program.

