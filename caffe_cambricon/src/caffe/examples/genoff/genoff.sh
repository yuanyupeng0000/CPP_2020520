# generating 1H16 offline model
# argv1: prototxt
# argv2: caffemodel
# argv3: core version
# argv4: model parallism
# argv5: Bangop
#../../build/tools/caffe genoff -model $1 -weights $2 -mcore $3
CURRENT_DIR=`dirname $0`
. $CURRENT_DIR/../../scripts/set_caffe_module_env.sh

if [[ "$5" == "" || "$5" == "0" ]]; then
  $CURRENT_DIR/../../build/tools/caffe$SUFFIX genoff -model $1 \
    -weights $2 \
    -mcore $3 \
    -hd_reshape 0 \
    -Bangop 0
exit
else
  $CURRENT_DIR/../../build/tools/caffe$SUFFIX genoff -model $1 \
  -weights $2 \
  -mcore $3 \
  -hd_reshape 0 \
  -Bangop 1
fi
