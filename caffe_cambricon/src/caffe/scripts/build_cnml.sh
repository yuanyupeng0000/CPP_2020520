#!/bin/bash
shopt -s nocasematch

function usage
{
    echo "Usage:"
    echo "  $0 [x86|aarch64]"
    echo ""
    echo "  Parameter description:"
    echo "    x86: build cnml with x86 platform."
    echo "    aarch64: build cnml with aarch64 platform."
}

function checkMluHome
{
    if [ $# -ne 1 ]; then
        echo "[Error] checkMluHome:Invalid parameter."
        usage
        exit 1
    fi

    if [ -z "${NEUWARE_HOME}" ]; then
        NEUWARE_HOME=$CAFFE_DIR/"mlu"/$1
        if [ ! -d "${NEUWARE_HOME}" ]; then
            mkdir -p "${NEUWARE_HOME}"
        fi
    fi
}

if [ $# -ne 1 ]; then
   echo "[ERROR] Invaild parameter."
   usage
   exit 1
fi

if [ -z "${CAFFE_DIR}" ]; then
    CAFFE_DIR=$(dirname $(dirname $(readlink -f $0)))
fi

CNML_DIR=${CAFFE_DIR}/cnml
# check cnml folder
if [ ! -d "${CNML_DIR}" ]; then
  echo "[ERROR]: ${CNML_DIR} does not exist."
  exit 1
fi

unset NEUWARE_HOME
SUFFIX="64"
pushd $CNML_DIR
if [[ $1 == "x86" ]]; then
   echo "Building cnml for x86..."
   $CNML_DIR/compileSP_c20.sh
elif [[ $1 == "aarch64" ]]; then
   echo "Building cnml for aarch64..."
   $CNML_DIR/compileSP_mlu220_aarch64.sh
else
  echo "[ERROR] Invalid parameter."
  usage
  exit 1
fi
popd

# check mlu folder
checkMluHome $1

if [ ! -d ${NEUWARE_HOME}/include ]; then
  mkdir -p ${NEUWARE_HOME}/include
fi
if [ ! -d ${NEUWARE_HOME}/lib${SUFFIX} ]; then
  mkdir -p ${NEUWARE_HOME}/lib${SUFFIX}
fi

cp `find $CNML_DIR -path $CNML_DIR/out -prune -o -name cnml.h -print` ${NEUWARE_HOME}/include
cp $CNML_DIR/runtime_api/cnrt.h ${NEUWARE_HOME}/include

if [[ $1 == "arm64_h" ]]; then
  cp ${CNML_DIR}/build/lib/libipu_smmu.so ${NEUWARE_HOME}/lib${SUFFIX}
fi
if [[ $1 == "arm64_u" ]]; then
  cp ${CNML_DIR}/build/lib/libmlu_driver.so ${NEUWARE_HOME}/lib${SUFFIX}
  cp ${CNML_DIR}/build/lib/libipu_smmu.so ${NEUWARE_HOME}/lib${SUFFIX}
fi
cp ${CNML_DIR}/build/lib/libcnml.so ${NEUWARE_HOME}/lib${SUFFIX}
cp ${CNML_DIR}/build/lib/libcnrt.so ${NEUWARE_HOME}/lib${SUFFIX}
