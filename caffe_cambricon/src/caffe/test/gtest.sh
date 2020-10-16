#!/bin/bash
##gtest

CAFFE_DIR=$(dirname $(dirname $(readlink -f $0)))
LOG_FILE=${CAFFE_DIR}/test/recompile.log
BUILD_TYPE="Release"
PLATFORM="x86"

if [ -z "${CNML_HOME}" ]; then
    CNML_HOME=${CAFFE_DIR}/mlu/${PLATFORM}
    export CNML_HOME
fi
if [ -z "${CNRT_HOME}" ]; then
    CNRT_HOME=${CAFFE_DIR}/mlu/${PLATFORM}
    export CNRT_HOME
fi

echo "Operator Unit Test."
if [ $# -gt 1 ]; then
  echo "eg: 1. run all test case: ./gtest.sh"
  echo "2. run the filtered test case: ./gtest.sh add_layer"
  exit 1
fi

echo "-------------TEST--------------"
if [ $# -eq 1 ]; then
  echo "single operator test: ${1}"
else
  echo "All the test cases are running."
fi

if [ ! -d ${CAFFE_DIR}/build ]; then
  mkdir ${CAFFE_DIR}/build
fi
cd ${CAFFE_DIR}/build

cmake -DCNML_HOME=${CNML_HOME} \
      -DCNRT_HOME=${CNRT_HOME} \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DBUILD_only_tests=$1 \
      -DCMAKE_SKIP_RPATH=${SKIP_RPATH} \
      .. &>${LOG_FILE}

make -j20 &>${LOG_FILE}

make runtest -j20

echo "-------------TEST--------------"
