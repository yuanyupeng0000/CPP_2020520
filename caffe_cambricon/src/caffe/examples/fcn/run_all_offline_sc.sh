#!/bin/bash
# Parameter1 int8_mode: 1-int8; 0-float16

usage()
{
    echo "Usage:"
    echo "  $0 [0|1] [MLU220|MLU270]"
    echo ""
    echo "  Parameter description:"
    echo "    parameter1: int8 mode or int16 mode. 0:int16, 1:int8"
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

if [[ "$#" -ne 2 ]]; then
  echo "[ERROR] Unknown parameter."
  usage
  exit 1
fi

# config
core_version=$2
network_list=(
    fcn
)

do_run()
{
    echo "----------------------"
    echo "single core"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
    echo "core version:   $core_version,  output_dtype: ${OUTPUT_MODE}"

    #first remove any offline model
    /bin/rm offline.cambricon* &> /dev/null

    log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    echo > $CURRENT_DIR/$log_file

    genoff_cmd="$CAFFE_DIR/build/tools/caffe${SUFFIX} genoff \
                 -model $proto_file \
                 -weights $model_file \
                 -output_dtype ${OUTPUT_MODE} \
                 -mcore $core_version &>> $CURRENT_DIR/$log_file"

    run_cmd="$CAFFE_DIR/build/examples/fcn/fcn_offline_singlecore$SUFFIX \
                 -offlinemodel $CURRENT_DIR/offline.cambricon \
                 -images $CURRENT_DIR/$FILE_LIST \
                 -outputdir $CURRENT_DIR/output \
                 --dump 1 &>> $CURRENT_DIR/$log_file"

    check_cmd="python $CAFFE_DIR/scripts/compute_mIOU.py --pred_dir $CURRENT_DIR/output --gt_dir $VOC_PATH --file_list $CURRENT_DIR/$FILE_LIST &>> $CURRENT_DIR/$log_file"

    echo "genoff_cmd: $genoff_cmd" &>> $CURRENT_DIR/$log_file
    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file
    echo "check_cmd: $check_cmd" &>> $CURRENT_DIR/$log_file

    echo "generating offline model..."
    eval "$genoff_cmd"

    if [[ "$?" -eq 0 ]]; then
        echo "running offline test..."
        eval "$run_cmd"
        #tail -n 3 $CURRENT_DIR/$log_file
        grep "^Total execution time: " -A 2 $CURRENT_DIR/$log_file
        eval "$check_cmd"
        tail -n 1 $CURRENT_DIR/$log_file
    else
        echo "generating offline model failed!"
    fi
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

quantize_mode=$1
bbox_anchor_path=$CURRENT_DIR/bbox_anchor

ds_name=""
if [[ $quantize_mode -eq 1 ]]; then
    ds_name="int8"
elif [[ $quantize_mode -eq 0 ]]; then
    ds_name="int16"
else
    echo "[ERROR] Unknown parameter."
    usage
    exit 1
fi

/bin/rm *.log &> /dev/null
/bin/rm -r output &> /dev/null
mkdir output &> /dev/null

for network in "${network_list[@]}"; do
   model_file=$CAFFE_MODELS_DIR/${network}/${network}_int8_dense.caffemodel
   checkFile $model_file
   if [ $? -eq 1 ]; then
       continue
   fi

   echo "===================================================="
   echo "running ${network} offline - ${ds_name},${desp}..."

   for proto_file in $CAFFE_MODELS_DIR/${network}/${network}_${ds_name}*dense_1batch.prototxt; do
       checkFile $proto_file
       if [ $? -eq 1 ]; then
           continue
       fi

       do_run
   done
done
