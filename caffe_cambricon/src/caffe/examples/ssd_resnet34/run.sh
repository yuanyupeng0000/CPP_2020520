CURRENT_DIR=`dirname $0`
. $CURRENT_DIR/../../scripts/set_caffe_module_env.sh

$CURRENT_DIR/../../build/examples/ssd/ssd_online_singlecore$SUFFIX \
  -model $CURRENT_DIR/ssd/resnet34_ssd.prototxt \
  -weights $CAFFE_MODELS_DIR/resnet34/resnet34_ssd.caffemodel \
  -images $CURRENT_DIR/ssd/$FILE_LIST \
  -outputdir $CURRENT_DIR/ \
  -labelmapfile $CURRENT_DIR/ssd/labelmap_voc.prototxt \
  -confidencethreshold 0.6 \
  -mmode MFUS
