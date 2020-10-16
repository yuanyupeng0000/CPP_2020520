#!/bin/bash
CURRENT_DIR=$(dirname $(readlink -f $0))

# check caffe directory
if [ -z "$CAFFE_DIR" ]; then
    CAFFE_DIR=$CURRENT_DIR/../..
else
    if [ ! -d "$CAFFE_DIR" ]; then
        echo "[ERROR] Please check CAFFE_DIR."
        exit 1
    fi
fi

. $CAFFE_DIR/scripts/set_caffe_module_env.sh


proto_file=$CAFFE_MODELS_DIR/vgg16/vgg16_int8_scale_dense_1batch.prototxt
weight_file=$CAFFE_MODELS_DIR/vgg16/vgg16_int8_dense.caffemodel
core_number=16
synset=synset_words.txt
log_file=vgg16_mutable_offline_log

echo "running genoff ..."
echo "using protofile: $proto_file"
echo "using weight: $weight_file"

if [ -f "$log_file" ] ; then
rm $CURRENT_DIR/$log_file
fi


cmd="$CAFFE_DIR/build/tools/caffe genoff -model $proto_file -weights $weight_file -mcore MLU270 -dimmutable 1 -core_number $core_number -simple_compile 1  &>> $CURRENT_DIR/$log_file"

echo "genoff: $cmd"
eval $cmd
echo "running mutable vgg16 ..."

$CAFFE_DIR/build/examples/vgg_mutabledim/mutable_offline_multicore -offlinemodel offline.cambricon -images $FILE_LIST -labels $synset &>> $CURRENT_DIR/$log_file

grep "Global accuracy :" -A 4 $CURRENT_DIR/$log_file
