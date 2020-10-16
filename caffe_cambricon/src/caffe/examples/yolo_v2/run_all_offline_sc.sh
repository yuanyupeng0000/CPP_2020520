#!/bin/bash
# Parameter1 quantize_type: 1-int8; 0-int16
# Parameter2 core_version: MLU270; MLU220

usage()
{
    echo "Usage:"
    echo "  $0 [0|1] [MLU220|MLU270]"
    echo ""
    echo "  Parameter description:"
    echo "    parameter1: int8 mode or int16 mode. 0:int16, 1:int8"
    echo "    parameter2: core version: MLU270 or MLU220"
}

checkFile()
{
    if [ -f $1 ]; then
        return 0
    else
        return 1
    fi
}

#used to enable Bangop or not, default is disabled
bang_option=1
core_version=$2

do_run()
{
    echo "----------------------"
    echo "single core"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
    echo "core version:   $core_version, output_dtype: ${OUTPUT_MODE}"

    #first remove any offline model
    /bin/rm offline.cambricon* &> /dev/null
    /bin/rm -r result &> /dev/null

    if [[ "$bang_option" == "0" ]]; then
      log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    else
      log_file=$(echo $proto_file | sed 's/prototxt$/log_Bangop/' | sed 's/^.*\///')
    fi
    echo > $CURRENT_DIR/$log_file

    genoff_cmd="$CAFFE_DIR/build/tools/caffe${SUFFIX} genoff \
                 -model $proto_file \
                 -weights $model_file \
                 -output_dtype ${OUTPUT_MODE} \
                 -mcore $core_version \
                 -Bangop $bang_option &>> $CURRENT_DIR/$log_file"

    run_cmd="$CAFFE_DIR/build/examples/yolo_v2/yolov2_offline_singlecore$SUFFIX \
                 -offlinemodel $CURRENT_DIR/offline.cambricon \
                 -images $CURRENT_DIR/$FILE_LIST \
                 -outputdir $CURRENT_DIR/output \
                 -labels $CURRENT_DIR/label_map.txt \
                 -Bangop ${bang_option} \
                 -preprocess_option 4 &>> $CURRENT_DIR/$log_file"

    check_cmd="python $CAFFE_DIR/scripts/meanAP_VOC.py $CURRENT_DIR/$FILE_LIST $CURRENT_DIR/output $VOC_PATH &>> $CURRENT_DIR/$log_file &>> $CURRENT_DIR/$log_file"


    echo "genoff_cmd: $genoff_cmd" &>> $CURRENT_DIR/$log_file
    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file
    echo "check_cmd: $check_cmd" &>> $CURRENT_DIR/$log_file

    echo "generating offline model..."
    eval "$genoff_cmd"

    if [[ "$?" -eq 0 ]]; then
        echo "running offline test..."
        eval "$run_cmd"
        #tail -n 2 $CURRENT_DIR/$log_file
        grep "^Total execution time: " -A 2 $CURRENT_DIR/$log_file
        eval "$check_cmd"
        tail -n 1 $CURRENT_DIR/$log_file
    else
        echo "generating offline model failed!"
    fi
}

network_list=(
    yolov2
)

if [[ "$#" -ne 2 ]]; then
    echo "[ERROR] Unknown parameter."
    usage
    exit 1
fi

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
/bin/rm *00*.txt &> /dev/null
/bin/rm *.log &> /dev/null
/bin/rm -rf ${CURRENT_DIR}/output &> /dev/null
mkdir -p ${CURRENT_DIR}/output &> /dev/null

for network in "${network_list[@]}"; do
    model_file=$CAFFE_MODELS_DIR/${network}/${network}_int8_dense.caffemodel
    checkFile $model_file
    if [ $? -eq 1 ]; then
        continue
    fi

    echo "===================================================="
    echo "running ${network} offline - ${ds_name},..."

    for proto_file in $CAFFE_MODELS_DIR/${network}/${network}_${ds_name}*dense_1batch.prototxt; do
        checkFile $proto_file
        if [ $? -eq 1 ]; then
            continue
        fi

        do_run
    done
done
