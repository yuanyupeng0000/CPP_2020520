---
title: Cambricon Offline Classification example
description: A simple example performing image classification using offline interface of Cambricon SDK
category: example
include_in_docs: true
priority: 10
---

# Classifying ImageNet: using Cambricon Neuware Runtime API

Cambricon Neuware Runtime (CNRT) provides the ability to run neural networks
without the need of deep learning frameworks. This is an example of running
classification network with CNRT API.

## Compiling

The ++ example is built automatically when compiling Caffe. To
compile Caffe you should follow the documented instructions. The
classification example will be built as `examples/clas_offline_singlecore/clas_offline_singlecore.bin`
in your build directory.

## Usage

This example is used a way very similar with Caffe's original `cpp_classification`,
where the difference is you will run with customized "offline model". Basically,
the running involes two steps.

### Generate Offline Model

Before run `clas_offline_singlecore`, you need to generate an offline model from command follow:
```
caffe genoff -model some.prototxt -weights some.caffemodel -mname model_name
```
where `genoff` tells `caffe` executable to generate Cambricon offline model;
`some.prototxt` and `some.caffemodel` are network definition and pre-trained models;
`model_name` is the offline model name. This command will generate a `model_name.cambricon` file
with offline model named `model_name` which can be called through function tag `subnet1`.

### Run Offline Model

With the generated offline model (`.cambricon` file), command below will invoke
the classification procedure:
```
clas_offline_singlecore model.cambricon img.jpg  labels.txt mean_file
```

The output should look like this:
```
-------------- detection for /data/imagenet/0.jpg ----------------
0.1824  -  n03594734 jean, blue jean, denim
0.1355  -  n02910353 buckle
0.1187  -  n03492542 hard disc, hard disk, fixed disk
0.0687  -  n03527444 holster
0.0645  -  n03197337 digital watch
```

This tool also generates `offline_output` which is the dump of the output of
the "last" layer of offline model for debug purpose.
