#!/bin/bash
echo "usage:"
echo "param[0]: prototxt file path"
echo "param[1]: caffe model path"
echo "param[2]: mode 0/1/2, 0: dump to file, 1: dump to memory, 2: verify offline model file"
echo "param[3]: hardware reshape 2: NCHW, 3: NHWC, others: no"

CURRENT_DIR=`dirname $0`
. $CURRENT_DIR/../../scripts/set_caffe_module_env.sh

$CURRENT_DIR/../../build/tools/caffe genoff \
                    -model $1 \
                    -weights $2 \
                    -mcore MLU100 \
                    -cpu_info 1

$CURRENT_DIR/../../build/examples/Task_Compile/test_compile$SUFFIX \
                    -model $1  \
                    -weights $2  \
                    -mode $3  \
                    -hdreshape $4  \
                    -mcore MLU100 \
                    -offlinemodel $5
