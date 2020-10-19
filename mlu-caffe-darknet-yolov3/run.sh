#!/bin/bash
export PATH=$PWD/cnrtexec:$PWD/cnrt_simple_demo:../../caffe/tools/:$PATH

export TFU_ENABLE=1
export TFU_NET_FILTER=0
export GLOG_minloglevel=5



# darknet model to caffe model
python $CAFFE_HOME/src/caffe/python/darknet2caffe-yoloV23.py 3 yolov3.cfg yolov3.weights yolov3.prototxt yolov3.caffemodel

# add yolodetection layer
cp yolov3.prototxt yolov3.prototxt.bak
echo '
layer {
    bottom: "layer82-conv"
    bottom: "layer94-conv"
    bottom: "layer106-conv"
    top: "yolo_1"
    name: "yolo-layer"
    type: "Yolov3Detection"
    yolov3_param {
        im_w:416
        im_h:416
        num_box:1024
        confidence_threshold:0.005
        nms_threshold:0.45
        biases:[116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23]
    }
}' >> yolov3.prototxt


# 量化模型int8
generate_quantized_pt -ini_file yolov3_intx.ini

#在线推理
python yolov3_forward_mlu.py yolov3_intx.prototxt yolov3.caffemodel


# 离线模型生成
caffe genoff -model yolov3_intx.prototxt -weights yolov3.caffemodel -mcore MLU270 -mname yolov3_intx_16_16 -core_number 16 -batchsize 16 -simple_compile 1
# 离线模型推理
# cnrtexec offline_model dev_id sample_count nTaskCount affinity
cnrtexec yolov3_intx_16_16.cambricon 0  1024 2 0
#Model:yolov3_intx_16_16.cambricon Batch:16 HwTime:62.404999(ms) FPS:255.224898 [ samples:16384.000000 usetime:64.194364(s) ]


caffe genoff -model yolov3_intx.prototxt -weights yolov3.caffemodel -mcore MLU270 -mname yolov3_intx_4_4 -core_number 4 -batchsize 4 -simple_compile 1
cnrtexec yolov3_intx_4_4.cambricon 0  1024 8 1
#Model:yolov3_intx_4_4.cambricon Batch:4 HwTime:39.269001(ms) FPS:400.574417 [ samples:4096.000000 usetime:10.225316(s) ]

caffe genoff -model yolov3_intx.prototxt -weights yolov3.caffemodel -mcore MLU270 -mname yolov3_intx_1_1 -core_number 1 -batchsize 1 -simple_compile 1
cnrtexec yolov3_intx_1_1.cambricon 0  1024 32 1
#Model:yolov3_intx_1_1.cambricon Batch:1 HwTime:78.273003(ms) FPS:190.848124 [ samples:1024.000000 usetime:5.365523(s) ]

