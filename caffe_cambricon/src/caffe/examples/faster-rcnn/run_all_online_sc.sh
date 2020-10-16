#!/bin/bash
# Parameter1 quantize_option: 0-int16; 1-int8
# Parameter2 mlu_option: 1-MLU; 2-MFUS
# Parameter3 core_version: MLU270; MLU220

usage()
{
    echo "Usage:"
    echo "  $0 [0|1] [1|2] [MLU270 | MLU220]"
    echo ""
    echo "  Parameter description:"
    echo "    parameter1: int8 mode or int16 mode. 0:int16, 1:int8"
    echo "    parameter2: layer by layer or fusion. 1:layer by layer; 2:fusion"
}

checkFile()
{
    if [ -f $1 ]; then
        return 0
    else
        return 1
    fi
}

if [[ "$#" -ne 3 ]]; then
  echo "[ERROR] Unknown parameter."
  usage
  exit 1
fi

# used to enable Bangop or not,default is disabled
bang_option=1
core_version=$3

network_list=(
    faster-rcnn
)

do_run()
{
    /bin/rm *.jpg &> /dev/null
    /bin/rm 200*.txt &> /dev/null

    echo "----------------------"
    echo "single core"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
    echo "core_version:   $core_version, output_dtype: ${OUTPUT_MODE}"
    log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    echo > $CURRENT_DIR/$log_file

    run_cmd="$CAFFE_DIR/build/examples/faster-rcnn/faster-rcnn_online_singlecore$SUFFIX \
                   -model $proto_file \
                   -weights $model_file \
                   -images $CURRENT_DIR/$FILE_LIST \
                   -outputdir $CURRENT_DIR \
                   -mmode $mlu_option \
                   -mcore $core_version \
                   -output_dtype ${OUTPUT_MODE} \
                   -Bangop $bang_option &>> $CURRENT_DIR/$log_file"
    check_cmd="python $CAFFE_DIR/scripts/meanAP_VOC.py $CURRENT_DIR/$FILE_LIST $CURRENT_DIR/ $VOC_PATH &>> $CURRENT_DIR/$log_file"

    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file
    echo "check_cmd: $check_cmd" &>> $CURRENT_DIR/$log_file

    echo "running online test..."
    eval "$run_cmd"
    grep "^Total execution time:" -A 2 $CURRENT_DIR/$log_file
    eval "$check_cmd"

    tail -n 1 $CURRENT_DIR/$log_file
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

mlu_option=""
if [[ $2 -eq 1 ]]; then
    mlu_option="MLU"
elif [[ $2 -eq 2 ]]; then
    mlu_option="MFUS"
else
    echo "[ERROR] Unknown parameter."
    usage
    exit 1
fi

quantize_type=$1
ds_name=""
if [[ $quantize_type -eq 1 ]]; then
    ds_name="int8"
elif [[ $quantize_type -eq 0 ]]; then
    ds_name="int16"
else
    echo "[ERROR] Unknown parameter."
    usage
    exit 1
fi

/bin/rm *.jpg &> /dev/null
/bin/rm 200*.txt &> /dev/null
/bin/rm *.log &> /dev/null

for network in "${network_list[@]}"; do
    model_file=$CAFFE_MODELS_DIR/${network}/${network}_int8_dense.caffemodel
    checkFile $model_file
    if [ $? -eq 1 ]; then
        continue
    fi

    echo "===================================================="
    echo "running ${network} online - ${ds_name}, Bangop..."

    bang_option=1
    for proto_file in $CAFFE_MODELS_DIR/${network}/${network}_${ds_name}*dense_1batch.prototxt; do
        checkFile $proto_file
        if [ $? -eq 1 ]; then
            continue
        fi
        do_run
    done
done
