#! /bin/bash
CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))
DATA_DIR=${EXAMPLES_DIR}/data/
echo ${EXAMPLES_DIR}
#PLATFORM='x86'
BIN_DIR=${EXAMPLES_DIR}/bin/${PLATFORM}/

export LD_LIBRARY_PATH=${EXAMPLES_DIR}/bin/${PLATFORM}:${THIRD_PARTY}

#imagenet2012
IMAGENET_FILE=${DATA_DIR}/imageNet2012/file_list
IMAGENET_LABEL=${DATA_DIR}/imageNet2012/synset_words.txt

#imagenet2015
IMAGENET15_FILE=${DATA_DIR}/imageNet2015/file_list
IMAGENET15_LABEL=${DATA_DIR}/imageNet2015/synset_words.txt

#voc07
VOC07_FILE=${DATA_DIR}/VOC2007/file_list
VOC07_LABEL=${DATA_DIR}/VOC2007/labelmap_voc.prototxt
VOC07_LABELS=${DATA_DIR}/VOC2007/label_map.txt

#voc12
VOC12_FILE=${DATA_DIR}/VOC2012/file_list

#coco
COCO_FILE=${DATA_DIR}/COCO/file_list
COCO_LABEL=${DATA_DIR}/COCO/label_map_coco.txt

#fddb
FDDB_FILE=${DATA_DIR}/FDDB/file_list

#sport1m
SPORTS1M_FILE=${DATA_DIR}/Sports1M/file_list
SPORTS1M_LABEL=${DATA_DIR}/Sports1M/synset_words.txt








