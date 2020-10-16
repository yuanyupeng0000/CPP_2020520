#!/bin/bash
shopt -s nocasematch

function usage
{
    echo "Usage:"
    echo "  $0 [-help|-h] [-debug|-d] [-release|-r] [-cnml] [-clean|-c]
        [-platform|-p x86|aarch64] [-version|-v MLU220|MLU270] [-job|-j <jobnum>] [-rpath]"
    echo ""
    echo "  Parameter description:"
    echo "    -help or -h: usage instructions."
    echo "    -debug or -d: build cambricon caffe with debug version, the default option is debug."
    echo "    -release or -r: build cambricon caffe with release version, the default option is debug."
    echo "    -cnml: build cnml or not, the default option is off."
    echo "    -clean or -c: clean build folder and rebuild caffe."
    echo "    -platform or -p: specity platform for cambricon caffe, the default platform is x86."
    echo "    -version or -v: specity platform version for cambricon caffe, the default platform is MLU100."
    echo "    -job or -j: specifies the number of jobs to compile concurrently."
    echo "    -rpath: keep RPATH in build binary."
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

function checkstatus
{
    if (($1!=0)); then
        echo "compile failed. $2"
        exit 1
    fi
}

CAFFE_DIR=$(dirname $(dirname $(readlink -f $0)))
BUILD_TYPE="Debug"
PLATFORM="x86"
CNML="nocnml"
CLEAN="no"
VERSION="MLU270"
JOB_NUM=20
SKIP_RPATH='TRUE'

PARAMETERS=`echo $@ |sed 's/-cnml//;'`

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
        -cnml)
            CNML="cnml"
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

export CAFFE_DIR
export NEUWARE_HOME
export BUILD_TYPE
export PLATFORM
export CNML
export CLEAN
export VERSION
export JOB_NUM
export SKIP_RPATH

if [[ "$PLATFORM" != "x86" && "$PLATFORM" != "aarch64" ]]; then
  echo "[ERROR] Invalid platform: $PLATFORM. Exit."
fi

# check platform and compile cnml
if [ "$CNML" == "cnml" ]; then
    ${CAFFE_DIR}/scripts/build_cnml.sh $PLATFORM
    checkstatus $? cnml
    ${CAFFE_DIR}/scripts/build_cnplugin.sh $PLATFORM
    checkstatus $? cnplugin
else
  echo "CNML and CNPLUGIN will not be compiled."
fi
# build caffe
${CAFFE_DIR}/scripts/build_cambriconcaffe.sh $PARAMETERS
