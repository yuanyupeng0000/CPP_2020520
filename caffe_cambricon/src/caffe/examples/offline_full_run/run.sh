CURRENT_DIR=`dirname $0`
. $CURRENT_DIR/../../scripts/set_caffe_module_env.sh

$CURRENT_DIR/../../build/tools/caffe$SUFFIX genoff -model $1 -weights $2 -mcore MLU100 -hd_reshape 0  -cpu_info 1
if [ $? -ne 0 ]
then
    exit 1
fi

$CURRENT_DIR/../../build/examples/offline_full_run/offline_full_run$SUFFIX \
                          -offlinemodel ./offline.cambricon --images $3
