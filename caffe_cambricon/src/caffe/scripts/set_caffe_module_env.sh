#! /bin/bash

# please define DEBUG_CAFFE if you'd like to compile with debug version
if [ ! -z ${DEBUG_CAFFE} ]; then
    SUFFIX="-d"
fi

if [ ! -z ${CAFFE_EXTERNAL} ]; then
    ROOT=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))
    CAFFE_MODELS_DIR=$ROOT/"caffe_mp_c20"
    VOC_PATH="../../../../../datasets/VOC2012/Annotations"
    COCO_PATH="../../../../../datasets/COCO"
    FDDB_PATH=$ROOT/"dataset/FDDB"
    VOC2007_PATH="../../../../../datasets"
fi

FILE_LIST="file_list_for_release"
CORE_VERSION="MLU270"
OUTPUT_MODE="FLOAT16"
FILE_LIST_2015="file_list_for_release_2015"
