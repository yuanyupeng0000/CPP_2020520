#!/bin/bash
shopt -s nocasematch

function usage
{
    echo "Usage:"
    echo "  $0 [-help|-h] [-debug|-d] [-release|-r] [-clean|-c] [-platform|-p x86|arm32|arm64|aarch64]
        [-version|-v MLU100|MLU270] [-job|-j <jobnum>]"
    echo ""
    echo "  Parameter description:"
    echo "    -help or -h: usage instructions."
    echo "    -debug or -d: build cambricon caffe with debug version, the default option is release."
    echo "    -release or -r: build cambricon caffe with release version, the default option is release."
    echo "    -clean or -c: clear build folder and rebuild caffe."
    echo "    -platform or -p: specity platform for cambricon caffe, the default platform is x86."
    echo "    -version or -v: specity platform version for cambricon caffe, the default platform is MLU100."
    echo "    -job or -j: specifies the number of jobs to compile concurrently."
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
        export NEUWARE_HOME
    fi
}

CAFFE_DIR=$(dirname $(dirname $(readlink -f $0)))
BUILD_DIR=$CAFFE_DIR/"build"
PLATFORM="x86"
JOB_NUM="8"
SKIP_RPATH='TRUE'

# check build type
if [ -z "${BUILD_TYPE}" ]; then
    BUILD_TYPE="Release"
fi

while [[ $# -ge 1 ]]
do
    arg="$1"
    case $arg in
        -help | -h)
            usage
            exit 0
            ;;
        -debug | -d)
            BUILD_TYPE="Debug"
            ;;
        -release | -r)
            BUILD_TYPE="Release"
            ;;
        -clean | -c)
            CLEAN="yes"
            ;;
        -rpath)
            SKIP_RPATH="FALSE"
            ;;
        -platform | -p)
            PLATFORM="$2"
            shift
            ;;
        -version | -v)
            VERSION="$2"
            shift
            ;;
        -job | -j)
           JOB_NUM="$2"
            shift
            ;;
        *)
            echo "[ERROR] Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
    shift
done

# check mlu folder
checkMluHome $PLATFORM

# check build folder
if [ ! -d ${BUILD_DIR} ]; then
    echo "mkdir build folder"
    mkdir ${BUILD_DIR}
else
    if [ "$CLEAN" == "yes" ]; then
        rm -fr $BUILD_DIR/*
    fi
fi
export CAFFE_DIR
export BUILD_DIR
export BUILD_TYPE
export PLATFORM
export JOB_NUM
export SKIP_RPATH

# please make sure this is the last step due to that user can get the compile status
if [ "$PLATFORM" == "x86" ]; then
    $CAFFE_DIR/scripts/build_caffe_x86.sh
elif [ "$PLATFORM" == "aarch64" ]; then
    $CAFFE_DIR/scripts/build_caffe_aarch64.sh
else
    echo "[ERROR] Invalid platform: $PLATFORM. Exit."
    usage
    exit 1
fi
