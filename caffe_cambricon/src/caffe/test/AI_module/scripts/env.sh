CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))
CAFFE_DIR=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))

#platform x86/aarch64
platform='x86'

if [ -z "${MODULE_DATA}" ]; then
  echo "[ERROR] MODULE_DATA NOT SET. EXIT."
  exit 1
fi

echo "Configuring the runtime environment."
mkdir -p ${EXAMPLES_DIR}/data
mkdir -p ${EXAMPLES_DIR}/model_zoo
mkdir -p ${EXAMPLES_DIR}/bin
cp -rf ${MODULE_DATA}/* ${EXAMPLES_DIR}/data

## bin
BIN_DIR=${EXAMPLES_DIR}/bin/${platform}/
mkdir -p ${BIN_DIR}
cp ${CAFFE_DIR}/mlu/x86/lib64/libcnrt.so ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/clas_offline_multicore/clas_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/ssd/ssd_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/yolo_v3/yolov3_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/yolo_v2/yolov2_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/rfcn/rfcn_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/mtcnn/mtcnn-offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/segnet/segnet_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/C3D/c3d_offline_multicore ${BIN_DIR}
cp ${CAFFE_DIR}/build/examples/faster-rcnn/faster-rcnn_offline_multicore ${BIN_DIR}
strip ${BIN_DIR}/*

