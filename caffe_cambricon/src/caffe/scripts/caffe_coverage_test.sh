set -e

#export the variables

if [ -z $COVERAGE_TRACE_FILE_NAME ]; then
  echo "please set environment variable COVERAGE_TRACE_FILE_NAME, using COVERAGE_TRACE_FILE_NAME=caffe.coverage"
  COVERAGE_TRACE_FILE_NAME=caffe.coverage
fi

if [ -z $COVERAGE_REPORT_DIR ]; then
  echo "please set environment variable COVERAGE_REPORT_DIR, using COVERAGE_REPORT_DIR=caffe_coverage"
  COVERAGE_REPORT_DIR=caffe_coverage
fi

if [ -z $CAFFE_MODELS_DIR ]; then
  echo "CAFFE_MODELS_DIR must be set!"
fi
CAFFE_HOME=$(pwd)
echo "Current caffe dir: $CAFFE_HOME"

rebuild=0
if [ $# = 1 ];then
rebuild=$1
fi

export TEST_COVERAGE=ON

if [ "$rebuild" = "1" ]; then
echo "cleaning up..."
pushd $CAFFE_HOME
rm -rf build
find . -name "*.gc*" | xargs rm -rf
popd

echo "====================build caffe================================="
./scripts/build_caffe.sh -r -v MLU270

exit 1;
fi
# echo "====================run the test================================"

pushd $CAFFE_HOME/test
Folder=$CAFFE_HOME/src/caffe/test
for File in ${Folder}/*
do
  file=`basename ${File}`
  file_a=${file##*.}
  str='cpp'
  if [ "${file_a}" = ${str} ]
  then
    temp_file=`basename ${file} .cpp`
    ./gtest.sh  ${temp_file#*_}
  fi
done
popd

pushd $CAFFE_HOME/examples/clas_online_multicore
./run_all_online_mc.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/clas_offline_multicore
./run_all_offline_mc.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/ssd
./run_all_offline_mc.sh 1 MLU270
./run_all_online_sc.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/yolo_v3/
./run_all_offline_mc.sh 1 MLU270
./run_all_online_sc.sh 1 2 MLU270
./run_all_online_sc.sh 1 1 MLU270
popd

pushd $CAFFE_HOME/examples/yolo_v2/
./run_all_offline_mc.sh 1 MLU270
./run_all_online_mc.sh 1 2 MLU270
./run_all_online_mc.sh 1 1 MLU270
popd

pushd $CAFFE_HOME/examples/vgg_mutabledim/
./run_offline_multicore.sh
popd

pushd $CAFFE_HOME/examples/faster-rcnn/
./run_all_offline_mc.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/rfcn
./run_all_offline_mc.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/pvanet/
./run_all_offline_mc.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/mtcnn/
./run_offline_multicore.sh 1 MLU270
popd

pushd $CAFFE_HOME/examples/C3D/
sed -i '16,1000d' file_list
./run_all_online_mc.sh MLU270
./run_all_offline_mc.sh MLU270
popd

pushd $CAFFE_HOME/examples/segnet/
./run_all_offline_mc.sh 1 MLU270
./run_all_online_mc.sh 1 2 MLU270
popd

pushd $CAFFE_HOME/examples/offline_full_run/
../../build/tools/caffe$SUFFIX genoff -model $CAFFE_MODELS_DIR/vgg16/vgg16_int8_scale_dense_1batch.prototxt -weights $CAFFE_MODELS_DIR/vgg16/vgg16_int8_dense.caffemodel -mcore MLU270  -cpu_info 1
#../../build/examples/offline_full_run/offline_full_run --offlinemodel ./offline.cambricon -images ../clas_online_multicore/file_list
popd

pushd $CAFFE_HOME/examples
cp ssd/file_list file_list_voc
sed '4c save_model_path = newpt ' ini_file/ssd_quantized.ini &> initmp
sed '12c used_images_num = 4 ' initmp &> inifile
../build/tools/generate_quantized_pt -ini_file inifile -model $CAFFE_MODELS_DIR/ssd_vgg16/ssd_vgg16_int8_scale_dense_1batch.prototxt -weights $CAFFE_MODELS_DIR/ssd_vgg16/ssd_vgg16_int8_dense.caffemodel
rm inifile
rm initmp
popd

pushd $CAFFE_HOME
lcov -c -d . -o $CAFFE_HOME/clog  --rc lcov_branch_coverage=1
lcov --remove clog */src/caffe/solvers/\* */src/caffe/util/\* -o $CAFFE_HOME/tmpclog  --rc lcov_branch_coverage=1
lcov --remove tmpclog */src/caffe/layers/region\* */src/caffe/layers/\*data_layer\* */src/caffe/layers/embed\* */src/caffe/layers/proposal\* */src/caffe/layers/hdf5\* */src/caffe/layers/silence\* */src/caffe/layers/mlu_rnn\* */src/caffe/layers/mlu_lstm\* */src/caffe/layers/axpy\* */src/caffe/layers/bn\* */src/caffe/layers/pool3d\* */src/caffe/layers/normalize\* */src/caffe/layers/conv_dep\* */src/caffe/layers/convolution3d\* */src/caffe/layers/ */src/caffe/mlu/data\* */src/caffe/layers/detection_output\* */src/caffe/layers/relu6\* -o $CAFFE_HOME/layerclog  --rc lcov_branch_coverage=1
lcov --extract layerclog  */src/caffe/layers\* */src/caffe/mlu/\* */src/caffe/net.cpp */src/caffe/blob.cpp */src/caffe/syncedmem.cpp  */src/caffe/syncedmem.cpp */src/caffe/internal_thread.cpp */include/caffe/mlu/\*  -o $COVERAGE_TRACE_FILE_NAME  --rc lcov_branch_coverage=1
genhtml $COVERAGE_TRACE_FILE_NAME -o ${COVERAGE_REPORT_DIR}  --rc lcov_branch_coverage=1
echo "Result saved to $COVERAGE_REPORT_DIR"
popd
