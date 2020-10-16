CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))
CAFFE_DIR=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))

if [ -z "${SDK_DATA}" ]; then
  echo "[ERROR] SDK_DATA NOT SET. EXIT."
  exit 1
fi

echo "Configuring the runtime environment."
echo ${SDK_DATA}
mkdir -p ${EXAMPLES_DIR}/data
mkdir -p ${EXAMPLES_DIR}/models
mkdir -p ${EXAMPLES_DIR}/1080P_models
mkdir -p ${EXAMPLES_DIR}/bin
mkdir -p ${EXAMPLES_DIR}/lib
cp -rf ${SDK_DATA}/* ${EXAMPLES_DIR}/data
cp ${CAFFE_DIR}/scripts/meanAP_VOC.py ${CURRENT_DIR}/meanAp_VOC.py
cp ${CAFFE_DIR}/scripts/meanAP_COCO.py ${CURRENT_DIR}/meanAP_COCO.py
cp ${CAFFE_DIR}/scripts/meanAP_FDDB.py ${CURRENT_DIR}/meanAP_FDDB.py
sed -i "s/cls_ap = voc_ap(rec, prec, use_07_metric=False)/\
cls_ap = voc_ap(rec, prec, use_07_metric=True)/g" ${CURRENT_DIR}/meanAp_VOC.py
chmod +x ${CURRENT_DIR}/meanAp_VOC.py
chmod +x ${CURRENT_DIR}/meanAP_COCO.py

# cp ${CAFFE_DIR}/build/lib/libcaffe.so ${EXAMPLES_DIR}/lib
cp ${CAFFE_DIR}/mlu/x86/lib64/libcnrt.so ${EXAMPLES_DIR}/lib
#cp ${CAFFE_DIR}/mlu/x86/lib64/libcnml.so ${EXAMPLES_DIR}/lib
#cp ${CAFFE_DIR}/mlu/x86/lib64/libcnplugin.so ${EXAMPLES_DIR}/lib
strip ${EXAMPLES_DIR}/lib/lib*.so
#cp ${CAFFE_DIR}/build/tools/caffe ${EXAMPLES_DIR}/bin
cp ${CAFFE_DIR}/build/examples/clas_offline_multicore/clas_offline_multicore ${EXAMPLES_DIR}/bin
cp ${CAFFE_DIR}/build/examples/ssd/ssd_offline_multicore ${EXAMPLES_DIR}/bin
cp ${CAFFE_DIR}/build/examples/yolo_v3/yolov3_offline_multicore ${EXAMPLES_DIR}/bin
cp ${CAFFE_DIR}/build/examples/yolo_v2/yolov2_offline_multicore ${EXAMPLES_DIR}/bin
cp ${CAFFE_DIR}/build/examples/faster-rcnn/faster-rcnn_offline_multicore ${EXAMPLES_DIR}/bin
cp ${CAFFE_DIR}/build/examples/mtcnn/mtcnn-offline_multicore ${EXAMPLES_DIR}/bin
strip ${EXAMPLES_DIR}/bin/*
