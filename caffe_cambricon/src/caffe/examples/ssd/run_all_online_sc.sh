#!/bin/bash
# Parameter1 int8_mode: 1-int8; 0-int16

usage()
{
    echo "Usage:"
    echo "  $0 [0|1] [MLU270 | MLU220]"
    echo ""
    echo "  Parameter description:"
    echo "    parameter1: int8 mode or int16 mode. 0:int16, 1:int8"
}

if [[ "$#" -ne 2 ]]; then
  echo "[ERROR] Unknown parameter."
  usage
  exit 1
fi

# config
core_version=$2
threshold=0.01
opt_level=1
mlu_option="MFUS"

network_list=(
  ssd_vgg16
  ssd_mobilenetv1
  ssd_mobilenetv2
)


checkFile()
{
    if [ -f $1 ]; then
        return 0
    else
        echo $1
        return 1
    fi
}

do_run()
{
    echo "----------------------"
    echo "single core"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
    echo "core_version:   $core_version, output_dtype: ${OUTPUT_MODE}"
    log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    echo > $CURRENT_DIR/$log_file

    run_cmd="$CAFFE_DIR/build/examples/ssd/ssd_online_singlecore$SUFFIX \
                  -model $proto_file \
                  -weights $model_file \
                  -images $CURRENT_DIR/$FILE_LIST \
                  -outputdir $CURRENT_DIR/output \
                  -output_dtype ${OUTPUT_MODE} \
                  -labelmapfile $CURRENT_DIR/labelmap_voc.prototxt \
                  -confidencethreshold $threshold \
                  -mmode $mlu_option \
                  -mcore $core_version \
                  -opt_level $opt_level &>> $CURRENT_DIR/$log_file"

    check_cmd="python $CURRENT_DIR/../../scripts/meanAP_VOC.py $CURRENT_DIR/$FILE_LIST $CURRENT_DIR/output $VOC_PATH &>> $CURRENT_DIR/$log_file"

    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file
    echo "check_cmd: $check_cmd" &>> $CURRENT_DIR/$log_file

    echo -e "running online test...\n"
    eval "$run_cmd"
    #tail -n 2 $CURRENT_DIR/$log_file
    grep "^Total execution time: " -A 1 $CURRENT_DIR/$log_file
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

int8_mode=$1
ds_name=""
if [[ $int8_mode -eq 1 ]]; then
    ds_name="int8"
else
    ds_name="int16"
fi

/bin/rm *.jpg &> /dev/null
/bin/rm 200*.txt &> /dev/null
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
    echo "running ${network} online - ${ds_name},${desp}..."

	  for proto_file in $CAFFE_MODELS_DIR/${network}/${network}_${ds_name}*dense_1batch.prototxt; do
        checkFile $proto_file
        if [ $? -eq 1 ]; then
            continue
        fi

        do_run
    done
done
