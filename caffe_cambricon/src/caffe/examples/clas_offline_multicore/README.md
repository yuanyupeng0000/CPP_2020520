# Cambricon Guide


## limitations

### supported networks
- googlenet
- densenet121
- densenet161
- densenet169
- densenet201
- mobilenet_v1
- mobilenet_v2
- resnet101
- resnet152
- resnet18
- resnet34
- resnet50
- vgg16
- vgg19
- squeezenet_v1.0
- squeezenet_v1.1
- inception-v3
- alexnet
- resnext26-32x4d(MLU220 is not supported)
- resnext50-32x4d(MLU220 is not supported)
- resnext101-32x4d

### supported cases

  MLU270 support batchsize: 1/4/16, corenumber: 1/4/16, quantization:int8/int16, data type: float16/float32.

  MLU220 m.2 support batchsize: 1/4/16, corenumber: 1/4, quantization:int8/int16, data type: float16/float32.

  MLU220 edge support batchsize: 1/4/16, corenumber: 1/4, quantization:int8/int16, data type: float16/float32.

### supported networks

- resnext101-64x4d

### supported cases

  MLU270 support batchsize: 1/4/16, corenumber: 1/4/16, quantization:int8/int16, data type: float16/float32.

  MLU220 m.2 support batchsize: 1/4/16, corenumber: 1/4, quantization:int8/int16, data type: float16/float32.

  MLU220 edge support batchsize: 1/4/16, corenumber: 1/4, quantization:int8, data type: float16/float32.