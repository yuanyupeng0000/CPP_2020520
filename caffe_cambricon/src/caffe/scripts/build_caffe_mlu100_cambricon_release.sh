#!/bin/bash

# build caffe without cnml
CAFFE_SRC=$(dirname $(dirname $(readlink -f $0)))
echo "CAFFE_SRC: "$CAFFE_SRC
${CAFFE_SRC}/scripts/build_caffe.sh -release

# ---- generate caffe other directories ----
echo "Start Generating Other Caffe Directories For Release Version..."

# --- generate examples ---
SOURCE_EXAMPLE_C="$CAFFE_SRC/examples"
DST_EXAMPLE_ONLINE_C="$CAFFE_SRC/../../examples/online/c++"
DST_EXAMPLE_OFFLINE_C="$CAFFE_SRC/../../examples/offline/c++"

# -- classification online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/classification"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/clas_online_singlecore/*.cpp $EXAMPLE_NETWORK/src/
cp $SOURCE_EXAMPLE_C/clas_online_multicore/*.cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_online_singlecore/ &> /dev/null
./run_all_online_sc.sh 1
popd &> /dev/null

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_online_multicore/ &> /dev/null
./run_all_online_mc.sh 1 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_online_singlecore/ &> /dev/null
./run_all_online_sc.sh 0
popd &> /dev/null

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_online_multicore/ &> /dev/null
./run_all_online_mc.sh 0 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of classification online --


# -- classification offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/classification"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/clas_offline_singlecore/*.cpp $EXAMPLE_NETWORK/src/
cp $SOURCE_EXAMPLE_C/clas_offline_multicore/*.cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_offline_singlecore/ &> /dev/null
./run_all_offline_sc.sh 1
popd &> /dev/null

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_offline_multicore/ &> /dev/null
./run_all_offline_mc.sh 1 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_offline_singlecore/ &> /dev/null
./run_all_offline_sc.sh 0
popd &> /dev/null

pushd \$CUR_DIR/../../../../src/caffe/examples/clas_offline_multicore/ &> /dev/null
./run_all_offline_mc.sh 0 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of classification offline --


# -- yolov2 online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/yolov2"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v2/*online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v2/ &> /dev/null
./run_all_online_sc.sh 1 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v2/ &> /dev/null
./run_all_online_sc.sh 0 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of yolov2 online --

# -- yolov2 offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/yolov2"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v2/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v2/ &> /dev/null
./run_all_offline_sc.sh 1
./run_all_offline_mc.sh 1 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v2/ &> /dev/null
./run_all_offline_sc.sh 0
./run_all_offline_mc.sh 0 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of yolov2 offline --

# -- yolov3 online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/yolov3"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v3/*online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v3/ &> /dev/null
./run_all_online_sc.sh 1 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v3/ &> /dev/null
./run_all_online_sc.sh 0 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of yolov3 online --

# -- yolov3 offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/yolov3"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/yolo_v3/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v3/ &> /dev/null
./run_all_offline_sc.sh 1
./run_all_offline_mc.sh 1 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/yolo_v3/ &> /dev/null
./run_all_offline_sc.sh 0
./run_all_offline_mc.sh 0 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of yolov3 offline --




# -- ssd_vgg16 online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/ssd_vgg16"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/ssd/ssd_online*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/ssd/ &> /dev/null
./run_all_online_sc.sh 1 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/ssd/ &> /dev/null
./run_all_online_sc.sh 0 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of ssd_vgg16 online --

# -- ssd_vgg16 offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/ssd_vgg16"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/ssd/ssd_offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/ssd/ &> /dev/null
./run_all_offline_sc.sh 1
./run_all_offline_mc.sh 1 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/ssd/ &> /dev/null
./run_all_offline_sc.sh 0
./run_all_offline_mc.sh 0 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of ssd_vgg16 offline --


# -- faster_rcnn_resnet18 online --
EXAMPLE_NETWORK="$DST_EXAMPLE_ONLINE_C/faster_rcnn_resnet18"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/faster-rcnn/*demo.cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/faster-rcnn/ &> /dev/null
./run_all_online_sc.sh 1 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/faster-rcnn/ &> /dev/null
./run_all_online_sc.sh 0 2
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of faster_rcnn_resnet18 online --

# -- faster_rcnn_resnet18 offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/faster_rcnn_resnet18"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/faster-rcnn/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_int8.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/faster-rcnn/ &> /dev/null
./run_all_offline_mc.sh 1 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_int8.sh

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/faster-rcnn/ &> /dev/null
./run_all_offline_mc.sh 0 0
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of faster_rcnn_resnet18 offline --

# -- mtcnn offline --
EXAMPLE_NETWORK="$DST_EXAMPLE_OFFLINE_C/mtcnn"
mkdir -p $EXAMPLE_NETWORK/src
cp $SOURCE_EXAMPLE_C/mtcnn/*offline*cpp $EXAMPLE_NETWORK/src/

cat << EOF > $EXAMPLE_NETWORK/run_fp16.sh
CUR_DIR=\$(dirname \$(readlink -f \$0))
echo "CUR_DIR: " \$CUR_DIR
pushd \$CUR_DIR/../../../../src/caffe/examples/mtcnn/ &> /dev/null
./run_offline.sh
popd &> /dev/null
EOF
chmod a+x $EXAMPLE_NETWORK/run_fp16.sh

# -- end of mtcnn offline --


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
ln -sf ../../models/caffe/mobilenet ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet101 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet152 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet18 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet34 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/resnet50 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/rfcn ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/squeezenet ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/ssd ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/vgg16 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/vgg19 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/yolov2 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/yolov3 ${CAFFE_SRC}/../../models/
ln -sf ../../models/caffe/mtcnn ${CAFFE_SRC}/../../models/

# --- change all example's imagenet path and VOC2012 path to the new one which in release directory

sed -i 's%FILE_LIST=.*$%FILE_LIST="file_list_for_release"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%FILE_LIST_2015=.*$%FILE_LIST="file_list_for_release_2015"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%VOC_PATH=.*$%VOC_PATH="../../../../../datasets/VOC2012/Annotations"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%VOV2007_PATH=.*$%VOC2007_PATH="../../../../../datasets"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%COCO_PATH=.*$%COCO_PATH="../../../../../datasets/COCO"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%CAFFE_MODULES_DIR=.*$%CAFFE_MODULES_DIR="../../../../../models/caffe"%g' ${CAFFE_SRC}/scripts/set_caffe_module_env.sh
sed -i 's%^csv_path\s*=.*$%csv_path = caffe_for_release.csv%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_clas\s*=.*$%images_file_clas    = ../../examples/clas_offline_multicore/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_clas_2015\s*=.*$%images_file_clas_2015    = ../../examples/clas_offline_multicore/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_ssd\s*=.*$%images_file_ssd     = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_fastrcnn\s*=.*$%images_file_fastrcnn= ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_yolov2\s*=.*$%images_file_yolov2  = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_yolov3\s*=.*$%images_file_yolov3  = ../../examples/yolo_v3/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_mtcnn\s*=.*$%images_file_mtcnn  = ../../examples/mtcnn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_rfcn\s*=.*$%images_file_rfcn  = ../../examples/rfcn/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^g_caffe_voc_path\s*=.*$%g_caffe_voc_path            = "../../../../../datasets/VOC2012/Annotations"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^g_caffe_coco_path\s*=.*$%g_caffe_coco_path          = "../../../../../datasets/COCO"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^g_caffe_fddb_path\s*=.*$%g_caffe_fddb_path            = "../../../../../datasets/FDDB"%g' ${CAFFE_SRC}/scripts/onetest/onetest.py
sed -i 's%^images_file_ssd_vgg16\s*=.*$%images_file_ssd_vgg16 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_ssd_mobilenetv1\s*=.*$%images_file_ssd_mobilenetv1 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf
sed -i 's%^images_file_ssd_mobilenetv2\s*=.*$%images_file_ssd_mobilenetv2 = ../../examples/ssd/file_list_for_release%g' ${CAFFE_SRC}/scripts/onetest/onetest.conf

echo "Caffe Done."
