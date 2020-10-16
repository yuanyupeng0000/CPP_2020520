#!/bin/bash
#Parameter int8_mode: 1-int8; 0-float16
# usage:
# 1) ./run_offline_mc.sh 0
# run all mtcnn offline tests with float16
# 2) ./run_offline_mc.sh 1
# run all mtcnn offline tests with int8
usage()
{
    echo "Usage:"
    echo "  $0 [0|1] [MLU100|MLU270]"
    echo ""
    echo "  Parameter description:"
    echo "    0: run all networks with float16 (not support on MLU270)"
    echo "    1: run all networks with int8"
    echo "    [MLU100|MLU270]: run all networks on MLU100 or MLU270."
}

checkFile()
{
    if [ -f $1 ]; then
      return 0
    else
      return 1
    fi
}

do_run()
{
    echo "----------------------"
    echo "multiple core"
    echo "using prototxt: $proto_file"
    echo "using model file:    $model_file"
    echo "threadnum:  ${thread_num}"

    #first remove any offline model
    /bin/rm *.cambricon* &> /dev/null

    log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
    echo > $CURRENT_DIR/$log_file
    echo $FILE_LIST

    genoff_cmd="python $CURRENT_DIR/gen_models.py --debug-suffix \'${SUFFIX}\' --file-list $CURRENT_DIR/$FILE_LIST --model-path ${CAFFE_MODELS_DIR}/${network} --caffe-dir ${CAFFE_DIR} --core-version ${core_version} >> $CURRENT_DIR/$log_file"

    run_cmd="$CAFFE_DIR/build/examples/mtcnn/mtcnn-offline_multicore$SUFFIX -images $CURRENT_DIR/$FILE_LIST -models $CURRENT_DIR/model_list_${ds_name} -threads ${thread_num} -int8 $int8_mode"

    evaluate_cmd="$CURRENT_DIR/evaluation/evaluate -f 0 -a $FDDB_PATH/ellipseList.txt -d $CURRENT_DIR/mtcnn.txt -i $FDDB_PATH/originalPics/ -l $CURRENT_DIR/$FILE_LIST -r mtcnn_result &> $CURRENT_DIR/evaresult"

    check_cmd="python $CAFFE_DIR/scripts/meanAP_FDDB.py $CURRENT_DIR/mtcnn.txt $FDDB_PATH/ellipseList.txt $CURRENT_DIR/mtcnn_roc.png"
    echo "genoff_cmd: $genoff_cmd" &>> $CURRENT_DIR/$log_file
    echo "run_cmd: $run_cmd" &>> $CURRENT_DIR/$log_file
    echo "check_cmd: $check_cmd" &>> $CURRENT_DIR/$log_file
    echo "generating offline model..."
    eval "$genoff_cmd"
    if [[ "$?" -eq 0 ]]; then
        echo -e "running multicore offline test...\n"
        eval "$run_cmd"
        eval "$evaluate_cmd"
        eval "$check_cmd"
        #tail -n 5 $CURRENT_DIR/$log_file

    else
        echo "generating offline model failed!"
    fi
}

stage_list=(
    1
    2
    3
)

network_list=(
    mtcnn
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

int8_mode=$1
ds_name=""
scale=""
core_version=$2
if [[ $int8_mode -eq 1 ]]; then
    ds_name="int8"
elif [[ $int8_mode -eq 0 ]]; then
    ds_name="float16"
else
    echo "[ERROR] Unknown parameter."
    usage
    exit 1
fi

thread_num="16"

/bin/rm *.log &> /dev/null

for network in "${network_list[@]}"; do
      model_file=$CURRENT_DIR/model_list_${ds_name}
      checkFile $model_file
      if [ $? -eq 1 ]; then
          continue
      fi

      echo -e "\n===================================================="
      echo "running ${network} offline - ${ds_name}..."
      for proto_file in $CAFFE_MODELS_DIR/${network}/det${stage_list}.prototxt; do
          checkFile $proto_file
          if [ $? -eq 1 ]; then
              continue
          fi
      done
      do_run
done
