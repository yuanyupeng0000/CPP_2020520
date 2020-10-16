CURRENT_DIR=$(dirname $(readlink -f $0))
MODELS_DIR="./models"
export LD_LIBRARY_PATH=./lib
export OUTPUT_E2E=ON
#export INTERVAL_TIME=true
driver=`cat /proc/driver/*/*/*/information | grep "Device name"`
echo "$driver"

check_mlu220()
{
  if [[ $driver != "Device name: mlu-220 M2" ]]; then
    echo "NO_DRIVER";
    exit 1
  fi
}
check_mlu270()
{
  if [[ $driver != "Device name: mlu-270 F4" ]]; then
    echo "NO_DRIVER";
    exit 1
  fi
}


if [ -f $1 ]; then
  echo "MLU offline examples:"
  echo "1) MLU220 U1 default 1batch"
  echo "2) MLU220 U1 default 4batch"
  echo "3) MLU220 U1 default 16batch"
  echo "4) MLU220 U1 1080P 1batch"
  echo "5) MLU270 U4 default 1batch"
  echo "6) MLU270 U4 default 4batch"
  echo "7) MLU270 U4 default 16batch"
  echo "8) MLU270 U4 1080P 1batch"
  echo "9) MLU270 U4 1080P 4batch"
  echo "10) MLU270 U4 1080P 16batch"
  echo "Select the hardware device to use:"
  read core
else
  core=$1
fi

batchsize=16
core_number=4
core_version="MLU270"

if [[ $core -eq 1 ]]; then
   core_version="MLU220" 
   core_number=4
	 batchsize=1
elif [[ $core -eq 2 ]]; then
   core_version="MLU220" 
   core_number=4
	 batchsize=4
elif [[ $core -eq 3 ]]; then
   core_version="MLU220" 
   core_number=4
	 batchsize=16
elif [[ $core -eq 4 ]]; then
   core_version="MLU220" 
   core_number=4
	 batchsize=1
	 MODELS_DIR="./1080P_models"
elif [[ $core -eq 5 ]]; then
   core_version="MLU270" 
   core_number=16
	 batchsize=1
elif [[ $core -eq 6 ]]; then
   core_version="MLU270" 
   core_number=16
	 batchsize=4
elif [[ $core -eq 7 ]]; then
   core_version="MLU270" 
   core_number=16
	 batchsize=16
elif [[ $core -eq 8 ]]; then
   core_version="MLU270" 
   core_number=16
	 batchsize=1
	 MODELS_DIR="./1080P_models"
elif [[ $core -eq 9 ]]; then
   core_version="MLU270" 
   core_number=16
	 batchsize=4
	 MODELS_DIR="./1080P_models"
elif [[ $core -eq 10 ]]; then
   core_version="MLU270" 
   core_number=16
	 batchsize=16
	 MODELS_DIR="./1080P_models"
else
   echo "Please select the support device."
   exit 1
fi

if [ -f $2 ]; then
echo "MLU offline examples:"
echo "1) resnet50"
echo "2) mobilenet_v2"
echo "3) ssd-vgg16"
echo "4) faster-rcnn"
echo "5) yolov3"
echo "6) yolov2"
echo "7) mtcnn"
echo "Select the offline model to run:"
read model
else
model=$2
fi

if [[ "$#" -eq 0 ]]; then
  echo "Specifies the number of iterations to run:"
  read iterations
fi

iterations=1

if [[ $iterations == "" ]]; then
   iterations=1
fi

## running

mkdir -p ${CURRENT_DIR}/output &> /dev/null
/bin/rm $log_file &> /dev/null
/bin/rm ${CURRENT_DIR}/output/* &> /dev/null

log_file=$CURRENT_DIR/output/offline.log
if [[ $MODELS_DIR == "./1080P_models" ]]; then
  run_cmd=" -preprocess_option 0 -iterations $iterations &>> $log_file"
else
  run_cmd=" -iterations $iterations &>> $log_file"
fi
log_cmd=" &>> $log_file"
interval_cmd="&& grep 'Interval time: ' $log_file  && grep 'Interval time: ' $log_file &> $CURRENT_DIR/output/interval.out "
global_cmd="grep 'Global accuracy : ' -A 5 $log_file $interval_cmd "
perf_cmd="grep '^throughput: ' -A 3 $log_file $interval_cmd "


checkFile()
{
    if [ -f $1 ]; then
        return 0
    else
				echo "[ERROR] NOT_IMPLEMENTED: not exist: $1"
        exit 1
    fi
}

do_run()
{
  OFFLINE="${MODELS_DIR}/${1}_${core_version}_${batchsize}batch_${core_number}core.cambricon"
  if [[ ${1} == "mtcnn" ]]; then
    OFFLINE="models/mtcnn_${core_version}/model_list_int8"
  fi
  echo "using model: $OFFLINE"
  source $CURRENT_DIR/scripts/command.sh
	checkFile $OFFLINE
  export OFFLINE
}

## running offline model
echo "--------------------------"
echo "running multicore offline test..."
case $model in
    1)
        do_run resnet50
        eval "${resnet50_cmd} ${run_cmd}"
        eval "$global_cmd"
        ;;
    2)
        do_run mobilenet_v2
        eval "${resnet50_cmd} ${run_cmd}"
        eval "$global_cmd"
        ;;
    3)
        do_run ssd_vgg16
        eval "${ssd_vgg16_cmd} ${run_cmd}"
        eval "$check_voc ${log_cmd} "
        eval $perf_cmd
        tail -n 1 output/offline.log
        ;;
    4)
        do_run faster-rcnn
        eval "${faster_rcnn_cmd} ${run_cmd}"
        eval "${check_voc} ${log_cmd} "
        eval "$perf_cmd "
        tail -n 1 output/offline.log
        ;;
    5)
        do_run yolov3
        eval "${yolov3_cmd} ${run_cmd}"
        eval "$check_coco $log_cmd"
        eval $perf_cmd
        mAp=`grep "IoU=0.50 " output/offline.log | cut -d "=" -f5`
        echo "mAp:$mAp"
        ;;
    6)
        do_run yolov2
        eval "${yolov2_cmd} ${run_cmd}"
        eval "$check_voc $log_cmd"
        eval $perf_cmd
        tail -n 1 output/offline.log
        ;;
    7)
        do_run mtcnn
        eval $mtcnn_cmd ${log_cmd}
        eval $check_fddb $log_cmd
        eval $perf_cmd
        tail -n 1 output/offline.log
        ;; 
    *)
        echo "[ERROR] Specified model error."
esac

/bin/rm ${CURRENT_DIR}/*.json &> /dev/null
