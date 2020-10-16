#!/bin/bash
shopt -s nocasematch

function usage
{
    echo "Usage:"
    echo "  $0 [x86|aarch64]"
    echo ""
    echo "  Parameter description:"
    echo "    x86: build cnplugin on x86 platform."
    echo "    aarch64: build cnplugin on aarch64 platform."
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

    if [ -z "$COMPILER_HOME" ]; then
       echo 'COMPILER_HOME NOT SET.'
       exit 1
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

CNPLUGIN_DIR=${CAFFE_DIR}/cnplugin
# check cnml folder
if [ ! -d "${CNPLUGIN_DIR}" ]; then
  echo "[ERROR]: ${CNPLUGIN_DIR} does not exist."
  exit 1
fi

# check mlu folder
checkMluHome $1

if [ ! -d ${NEUWARE_HOME}/bin ]; then
  mkdir -p ${NEUWARE_HOME}/bin
fi

ln -sf $COMPILER_HOME/cnas ${NEUWARE_HOME}/bin
ln -sf $COMPILER_HOME/cncc ${NEUWARE_HOME}/bin

SUFFIX="64"
pushd $CNPLUGIN_DIR
export NEUWARE_HOME=$NEUWARE_HOME
if [ $1 == "x86" ]; then
     $CNPLUGIN_DIR/build.sh --mlu270
 elif [ $1 == "aarch64" ]; then
     $CNPLUGIN_DIR/build_aarch64.sh --mlu270
else
  echo "[ERROR] Invalid parameter."
  usage
  exit 1
fi
popd

if [ ! -d ${NEUWARE_HOME}/include ]; then
  mkdir -p ${NEUWARE_HOME}/include
fi
if [ ! -d ${NEUWARE_HOME}/lib${SUFFIX} ]; then
  mkdir -p ${NEUWARE_HOME}/lib${SUFFIX}
fi

cp `find $CNPLUGIN_DIR -path $CNPLUGIN_DIR/out -prune -o -name cnplugin.h -print` ${NEUWARE_HOME}/include
cp $CNPLUGIN_DIR/common/include/cnplugin.h ${NEUWARE_HOME}/include

cp ${CNPLUGIN_DIR}/build/libcnplugin.so ${NEUWARE_HOME}/lib${SUFFIX}
