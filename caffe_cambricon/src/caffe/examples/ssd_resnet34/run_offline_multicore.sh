CURRENT_DIR=`dirname $0`
. $CURRENT_DIR/../../scripts/set_caffe_module_env.sh

$CURRENT_DIR/../../build/tools/caffe$SUFFIX genoff -model ssd/resnet34_ssd.prototxt \
  -weights $CAFFE_MODELS_DIR/resnet34/resnet34_ssd.caffemodel \
  -mcore MLU100

$CURRENT_DIR/../../build/examples/ssd/ssd_offline_multicore$SUFFIX \
	-offlinemodel $CURRENT_DIR/offline.cambricon \
	-images $CURRENT_DIR/ssd/$FILE_LIST  \
	-labelmapfile $CURRENT_DIR/ssd/labelmap_voc.prototxt \
	-confidencethreshold 0.6 \
  -threads 1 \
  -dump 1
