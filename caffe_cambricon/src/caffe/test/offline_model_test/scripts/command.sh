EXAMPLES_DIR=$(dirname $(readlink -f $0))
BIN_DIR=$EXAMPLES_DIR/bin
LIB_DIR=$EXAMPLES_DIR/lib
DATA_DIR=$EXAMPLES_DIR/data
OUTPUT_DIR=$EXAMPLES_DIR/output

offline_model=" -offlinemodel "


resnet50_cmd="${BIN_DIR}/clas_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/imageNet2012/file_list \
    -labels ${DATA_DIR}/imageNet2012/synset_words.txt \
    -preprocess_option 1 "

vgg16_cmd="${BIN_DIR}/clas_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/imageNet2012/file_list \
    -labels ${DATA_DIR}/imageNet2012/synset_words.txt \
    -preprocess_option 1 "

inception_v3_cmd="${BIN_DIR}/clas_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/imageNet2015/file_list \
    -labels ${DATA_DIR}/imageNet2015/synset_words.txt \
    -preprocess_option 3 "

ssd_vgg16_cmd="${BIN_DIR}/ssd_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/VOC2007/file_list \
    -labelmapfile ${DATA_DIR}/VOC2007/labelmap_voc.prototxt  \
    -outputdir $EXAMPLES_DIR/output "

yolov3_cmd="${BIN_DIR}/yolov3_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/COCO/file_list \
    -labels ${DATA_DIR}/COCO/label_map_coco.txt \
    -outputdir $EXAMPLES_DIR/output \
    -preprocess_option 4 "

yolov2_cmd="${BIN_DIR}/yolov2_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/VOC2007/file_list \
    -labels ${DATA_DIR}/VOC2007/label_map.txt  \
    -outputdir $EXAMPLES_DIR/output \
    -preprocess_option 4 \
    -Bangop 1"

faster_rcnn_cmd="${BIN_DIR}/faster-rcnn_offline_multicore \
    -offlinemodel ${EXAMPLES_DIR}/${OFFLINE} \
    -images ${DATA_DIR}/VOC2007/file_list \
    -outputdir $EXAMPLES_DIR/output \
    -Bangop 1 \
    -dump 1"

check_voc="python ${EXAMPLES_DIR}/scripts/meanAp_VOC.py \
        ${DATA_DIR}/VOC2007/file_list \
        ${OUTPUT_DIR} \
        ${DATA_DIR}/VOC2007 "

check_coco="python ${EXAMPLES_DIR}/scripts/meanAP_COCO.py  \
    --file_list ${DATA_DIR}/COCO/file_list \
    --result_dir ${OUTPUT_DIR} \
    --ann_dir  ${DATA_DIR}/COCO \
    --data_type val2014"

mtcnn_cmd="${BIN_DIR}/mtcnn-offline_multicore \
    -images ${DATA_DIR}/FDDB/file_list \
    -models ${EXAMPLES_DIR}/${OFFLINE} \
    -threads 16 -int8 1 && \
    mv ${CURRENT_DIR}/*.jpg ${CURRENT_DIR}/output && \
    mv ${CURRENT_DIR}/mtcnn.txt ${CURRENT_DIR}/output"

check_fddb="python ${EXAMPLES_DIR}/scripts/meanAP_FDDB.py \
    ${OUTPUT_DIR}/mtcnn.txt \
    ${DATA_DIR}/FDDB/ellipseList.txt \
    ${OUTPUT_DIR}/mtcnn_roc.png"






