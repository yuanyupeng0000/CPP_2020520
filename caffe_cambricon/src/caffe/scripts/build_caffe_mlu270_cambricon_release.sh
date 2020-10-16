#!/bin/bash

# build caffe without cnml
CAFFE_SRC=$(dirname $(dirname $(readlink -f $0)))
echo "CAFFE_SRC: "$CAFFE_SRC
${CAFFE_SRC}/scripts/build_cambriconcaffe.sh

# ---- generate caffe other directories ----
echo "Start Generating Other Caffe Directories For Release Version..."

# --- generate examples ---
SOURCE_EXAMPLE_C="$CAFFE_SRC/examples"
DST_EXAMPLE_ONLINE_C="$CAFFE_SRC/../../examples/online/c++"
DST_EXAMPLE_OFFLINE_C="$CAFFE_SRC/../../examples/offline/c++"

SCRIPT_STR='CUR_DIR=$(dirname $(readlink -f $0)) \n
echo "CUR_DIR: " $CUR_DIR\n
if [[ "$#" -ne 1 ]]; then\n
  echo "Usage:"\n
      echo "  $0 [MLU220|MLU270]"\n
  exit 1\n
fi\n
core_version=$1\n\n'

SCRIPT_CHECK=`echo -e ${SCRIPT_STR}`

EXAMPLE_DIR='$CUR_DIR/../../../../src/caffe/examples'

# -- classification online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/classification"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/clas_online_singlecore/*.cpp $EXAMPLE_NETWORK/src/
cp $SOURCE_EXAMPLE_C/clas_online_multicore/*.cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/clas_online_singlecore/ &> /dev/null
./run_all_online_sc.sh 1 \$core_version
popd &> /dev/null

pushd ${EXAMPLE_DIR}/clas_online_multicore/ &> /dev/null
./run_all_online_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of classification online --


# -- classification offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/classification"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/clas_offline_singlecore/*.cpp $EXAMPLE_NETWORK/src/
cp $SOURCE_EXAMPLE_C/clas_offline_multicore/*.cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/clas_offline_singlecore/ &> /dev/null
./run_all_offline_sc.sh 1 \$core_version
popd &> /dev/null

pushd ${EXAMPLE_DIR}/clas_offline_multicore/ &> /dev/null
./run_all_offline_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of classification offline --


# -- yolov2 online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/yolov2"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v2/*online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/yolo_v2/ &> /dev/null
./run_all_online_sc.sh 1 2 \$core_version
./run_all_online_mc.sh 1 2 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of yolov2 online --

# -- yolov2 offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/yolov2"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v2/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/yolo_v2/ &> /dev/null
./run_all_offline_sc.sh 1 \$core_version
./run_all_offline_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of yolov2 offline --

# -- yolov3 online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/yolov3"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v3/*online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/yolo_v3/ &> /dev/null
./run_all_online_sc.sh 1 2 \$core_version
./run_all_online_mc.sh 1 2 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of yolov3 online --

# -- yolov3 offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/yolov3"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v3/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/yolo_v3/ &> /dev/null
./run_all_offline_sc.sh 1 \$core_version
./run_all_offline_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of yolov3 offline --


# -- ssd online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/ssd"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/ssd/ssd_online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/ssd/ &> /dev/null
./run_all_online_sc.sh 1 \$core_version
./run_all_online_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of ssd online --

# -- ssd offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/ssd"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/ssd/ssd_offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/ssd/ &> /dev/null
./run_all_offline_sc.sh 1 \$core_version
./run_all_offline_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of ssd offline --

# -- faster_rcnn online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/faster_rcnn"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/faster-rcnn/*online*.cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/faster-rcnn/ &> /dev/null
./run_all_online_sc.sh 1 2 \$core_version
./run_all_online_mc.sh 1 2 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of faster_rcnn_resnet18 online --

# -- faster_rcnn offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/faster_rcnn"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/faster-rcnn/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/faster-rcnn/ &> /dev/null
./run_all_offline_sc.sh 1 \$core_version
./run_all_offline_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of faster_rcnn offline --

# -- mtcnn offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/mtcnn"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/mtcnn/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/mtcnn/ &> /dev/null
./run_offline_multicore.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of mtcnn offline --

# -- rfcn online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/rfcn"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/rfcn/*online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/pvanet/ &> /dev/null
./run_all_online_sc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of rfcn online --


# -- rfcn offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/pvanet"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/rfcn/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
$SCRIPT_CHECK

pushd ${EXAMPLE_DIR}/rfcn/ &> /dev/null
./run_all_offline_sc.sh 1 \$core_version
./run_all_offline_mc.sh 1 \$core_version
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh
# -- end of rfcn offline --

# --- generate include dir files ---
SOURCE_DIR="$CAFFE_SRC/include/caffe"
DST_DIR="$CAFFE_SRC/../../include/caffe"

cp -rf $SOURCE_DIR/* $DST_DIR/

# --- generate lib dir files ---
SOURCE_DIR="$CAFFE_SRC/build/lib"
DST_DIR="$CAFFE_SRC/../../lib"

cp -rf $SOURCE_DIR/* $DST_DIR/

# --- generate tools dir files ---
SOURCE_DIR="$CAFFE_SRC/build/tools"
DST_DIR="$CAFFE_SRC/../../tools"

find $SOURCE_DIR/ -maxdepth 1 -type f -perm /a+x -exec cp -f {} $DST_DIR/ \;
rm -f $DST_DIR/*debug*


#  create softlink under caffe/models to outer dir
ln -sf ../../models/caffe/alexnet ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/faster-rcnn ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/googlenet ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/inception-v3 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/mobilenet_v1 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/mobilenet_v2 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet101 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet152 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet18 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet34 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet50 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/rfcn ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/squeezenet_v1.0 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/squeezenet_v1.1 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/ssd ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/vgg16 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/vgg19 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/yolov2 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/yolov3 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/mtcnn ${CAFFE_SRC}/../../models/

# --- change all example's imagenet path and VOC2012 path to the new one which in release directory

sed -i 's%FILE_LIST=.*$%FILE_LIST="file_list_for_release"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%FILE_LIST_2015=.*$%FILE_LIST_2015="file_list_for_release_2015"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%VOC_PATH=.*$%VOC_PATH="../../../../../datasets/VOC2012/Annotations"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%VOC2007_PATH=.*$%VOC2007_PATH="../../../../../datasets"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%COCO_PATH=.*$%COCO_PATH="../../../../../datasets/COCO"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%CAFFE_MODULES_DIR=.*$%CAFFE_MODULES_DIR="../../../../../models/caffe"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh

# MLU270
sed -i 's%^csv_path\s*=.*$%csv_path = caffe_for_release.csv%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_clas\s*=.*$%images_file_clas    = ../../examples/clas_offline_multicore/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_clas_2015\s*=.*$%images_file_clas_2015    = ../../examples/clas_offline_multicore/file_list_for_release_2015%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_ssd\s*=.*$%images_file_ssd     = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_fastrcnn\s*=.*$%images_file_fastrcnn= ../../examples/faster-rcnn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_yolov2\s*=.*$%images_file_yolov2  = ../../examples/yolo_v2/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_yolov3\s*=.*$%images_file_yolov3  = ../../examples/yolo_v3/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_mtcnn\s*=.*$%images_file_mtcnn  = ../../examples/mtcnn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_rfcn\s*=.*$%images_file_rfcn  = ../../examples/rfcn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^g_caffe_voc_path\s*=.*$%g_caffe_voc_path            = "../../../../../datasets/VOC2012/Annotations"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^g_caffe_coco_path\s*=.*$%g_caffe_coco_path          = "../../../../../datasets/COCO"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^g_caffe_fddb_path\s*=.*$%g_caffe_fddb_path            = "../../../../../datasets/FDDB"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^images_file_ssd_vgg16\s*=.*$%images_file_ssd_vgg16 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_ssd_mobilenetv1\s*=.*$%images_file_ssd_mobilenetv1 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_ssd_mobilenetv2\s*=.*$%images_file_ssd_mobilenetv2 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf
sed -i 's%^images_file_segnet\s*=.*$%images_file_segnet  = ../../examples/segnet/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu270.conf

# MLU220
sed -i 's%^csv_path\s*=.*$%csv_path = caffe_for_release.csv%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_clas\s*=.*$%images_file_clas    = ../../examples/clas_offline_multicore/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_clas_2015\s*=.*$%images_file_clas_2015    = ../../examples/clas_offline_multicore/file_list_for_release_2015%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_ssd\s*=.*$%images_file_ssd     = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_fastrcnn\s*=.*$%images_file_fastrcnn= ../../examples/faster-rcnn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_yolov2\s*=.*$%images_file_yolov2  = ../../examples/yolo_v2/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_yolov3\s*=.*$%images_file_yolov3  = ../../examples/yolo_v3/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_mtcnn\s*=.*$%images_file_mtcnn  = ../../examples/mtcnn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_rfcn\s*=.*$%images_file_rfcn  = ../../examples/rfcn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^g_caffe_voc_path\s*=.*$%g_caffe_voc_path            = "../../../../../datasets/VOC2012/Annotations"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^g_caffe_coco_path\s*=.*$%g_caffe_coco_path          = "../../../../../datasets/COCO"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^g_caffe_fddb_path\s*=.*$%g_caffe_fddb_path            = "../../../../../datasets/FDDB"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^images_file_ssd_vgg16\s*=.*$%images_file_ssd_vgg16 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_ssd_mobilenetv1\s*=.*$%images_file_ssd_mobilenetv1 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_ssd_mobilenetv2\s*=.*$%images_file_ssd_mobilenetv2 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf
sed -i 's%^images_file_segnet\s*=.*$%images_file_segnet  = ../../examples/segnet/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest_mlu220.conf

echo "Caffe Done."
