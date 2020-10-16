#!/bin/bash

usage()
{
    echo "Usage:"
    echo "  $0 [MLU220|MLU270]"
    echo ""
    echo "  Parameter description:"
    echo "    parameter1: core version: MLU270/MLU220"
}

checkFile()
{
    if [ -f $1 ]; then
        return 0
    else
        echo $1
        return 1
    fi
}

if [[ "$#" -ne 1 ]]; then
  echo "[ERROR] Unknown parameter."
  usage
  exit 1
fi

# config
core_version=$1
dataset='ucf101'
mmode='CPU'

bscn_list=(
  # '1  1 '
   '1  4 '
  # '16  4 '
  # '1  16 '
  # '16 16'
)

if [[ 'MLU220' == $core_version ]]; then
  bscn_list=(
   #  '1  1'
   #  '1  4'
     '1  4'
   #  '16  4'
  )
fi

network_list=(
    c3d
    c3d_v1.1
)

sample_rate=1

do_run()
{
    echo "----------------------"
    echo "multiple core"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
    echo "batchsize:  $batchsize,  core_number:  $core_number"

    log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    echo > $CURRENT_DIR/$log_file

    run_cmd="$CAFFE_DIR/build/examples/C3D/c3d_online_multicore$SUFFIX \
        -model $proto_file \
        -weights $model_file \
        -mcore $core_version \
        -labels $CURRENT_DIR/$dataset/$synset \
        -images $CURRENT_DIR/$dataset/$file_list \
        -mmode $mmode \
        -sampling_rate $sample_rate"
    concurrent_run=" -batchsize $batchsize -core_number $core_number &>> $CURRENT_DIR/$log_file"
    run_cmd="$run_cmd $concurrent_run"

    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file

    echo "running online test..."
    eval "$run_cmd"
    #tail -n 4 $CURRENT_DIR/$log_file
    grep "Global accuracy: $" -A 3 $CURRENT_DIR/$log_file
}

CURRENT_DIR=$(dirname $(readlink -f $0))

# check caffe directory
if [ -z "$CAFFE_DIR" ]; then
    CAFFE_DIR=$CURRENT_DIR/../..
else
    if [ ! -d "$CAFFE_DIR" ]; then
        echo "[ERROR] Please check CAFFE_DIR."
        exit 1
    fi
fi

. $CAFFE_DIR/scripts/set_caffe_module_env.sh

/bin/rm *.log &> /dev/null

for network in "${network_list[@]}"; do
   model_file=$CAFFE_MODELS_DIR/c3d/${network}_int8_dense.caffemodel
   checkFile $model_file
   if [ $? -eq 1 ]; then
       continue
   fi

   file_list=$FILE_LIST
   synset=synset_words.txt
   if [ ${network} == 'c3d_v1.1' ]; then
      dataset='sports1m'
      sample_rate=2 
   fi
   
   echo "===================================================="
   echo "running ${network} online ..."
	 for proto_file in $CAFFE_MODELS_DIR/c3d/${network}_int8_scale_dense_1batch.prototxt; do
	    checkFile $proto_file
	    if [ $? -eq 1 ]; then
		    continue
	    fi
	    for bscn in "${bscn_list[@]}"; do
		    batchsize=${bscn:0:2}
		    core_number=${bscn:3:2}
		    do_run
	    done
	done
done
