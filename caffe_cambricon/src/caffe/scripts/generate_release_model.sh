#!/bin/bash

###################################################
#
# generate_release_model.sh
# Usage:
# 1) Check out the model project:
#   git clone git@git.software.cambricon.com:caffe_sh/caffe_pts.git
# 2) Copy the prototxt files into current caffe directory:
#   cp -r caffe_pts/caffe/* /path/to/models_and_data/caffe/
# 3) Put this script under /path/to/models_and_data/
# 4) Run: ./generate_release_model.sh, then caffe_mp/ will be generated
#
###################################################

CAFFE_RELEASE="caffe_mp"
PROTO_ALL="./$CAFFE_RELEASE/proto"
MODEL_ALL="./$CAFFE_RELEASE/model"

mkdir -p $CAFFE_RELEASE/{proto,model}
echo "collecting all prototxt files..."
find caffe/ -name *prototxt -exec cp -f {} $PROTO_ALL/ \;
echo "collecting all caffemodel files..."
find caffe/ -name *caffemodel -exec cp -f {} $MODEL_ALL/ \;

mkdir -p $CAFFE_RELEASE/{combination,alexnet,faster-rcnn,googlenet,inception-v3,mobilenet,resnet101,resnet152,resnet18,resnet34,resnet50,rfcn,squeezenet,ssd,vgg16,vgg19,yolov2}

echo "classifying all prototxt files..."

pushd $CAFFE_RELEASE &> /dev/null

mv -f proto/{googlenet22_faster-rcnn_1080P*,mobilenet_faster-rcnn_1080P*,resnet18_faster-rcnn_1080P*,resnet34_ssd_1080P*,resnet50_ssd_1080P*} combination/

mv -f proto/alexnet* alexnet/
mv -f proto/faster-rcnn* faster-rcnn/
mv -f proto/googlenet* googlenet/
mv -f proto/inception* inception-v3/
mv -f proto/mobilenet* mobilenet/
mv -f proto/resnet101* resnet101/
mv -f proto/resnet152* resnet152/
mv -f proto/resnet18* resnet18/
mv -f proto/resnet34* resnet34/
mv -f proto/resnet50* resnet50/
mv -f proto/squeezenet* squeezenet/
mv -f proto/ssd* ssd/
mv -f proto/vgg16* vgg16/
mv -f proto/vgg19* vgg19/
mv -f proto/yolov2* yolov2/
popd &> /dev/null

echo "classifying all caffemodel files..."

pushd $CAFFE_RELEASE &> /dev/null

mv -f model/alexnet* alexnet/
mv -f model/faster-rcnn* faster-rcnn/
mv -f model/googlenet* googlenet/
mv -f model/inception* inception-v3/
mv -f model/mobilenet* mobilenet/
mv -f model/resnet101* resnet101/
mv -f model/resnet152* resnet152/
mv -f model/resnet18* resnet18/
mv -f model/resnet34* resnet34/
mv -f model/resnet50* resnet50/
mv -f model/rfcn* rfcn/
mv -f model/squeezenet* squeezenet/
mv -f model/ssd* ssd/
mv -f model/vgg16* vgg16/
mv -f model/vgg19* vgg19/
mv -f model/yolov2* yolov2/
popd &> /dev/null

/bin/rm -rf $PROTO_ALL
/bin/rm -rf $MODEL_ALL

echo "Done!"
