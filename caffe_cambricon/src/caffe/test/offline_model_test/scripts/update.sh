CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))
MODELS_DIR=$EXAMPLES_DIR/1080P_models
CAFFE_DIR=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))
echo ${CAFFE_DIR}

if [ -z "${CAFFE_MODELS_DIR}" ]; then
    echo "[ERROR] CAFFE_MODELS_DIR NOT SET. EXIT."
    exit 1
fi
echo "MODEL_ZOO: ${CAFFE_MODELS_DIR}"

# param
core_version='MLU220'
echo "CORE_VERSION: $core_version"

batchsize=1
core_number=4
echo  "SIMPLE_COMPILE: batchsize $batchsize core_number $core_number"

network_list=(
   resnet50
   mobilenet_v2
   ssd_vgg16
   faster-rcnn
 	 yolov3
	 yolov2
)

pt_name="_int8_scale_dense_1batch.prototxt"
model_name="_int8_dense.caffemodel"

checkFile()
{
    if [ -f $1 ]; then
        return 0
    else
				echo "not exist: $1"
        return 1
    fi
}

do_run()
{
    echo "----------------------"
    echo "generate offline model: $network"
    echo "using prototxt: $proto_file"
    echo "using model:    $model_file"
		
		log_file=$(echo $proto_file | sed 's/prototxt$/log/' | sed 's/^.*\///')
		genoff_cmd="$CAFFE_DIR/build/tools/caffe genoff \
				-model $proto_file \
				-weights $model_file \
				-mcore $core_version \
        -mname ${MODELS_DIR}/${network}_${core_version}_${batchsize}batch_${core_number}core \
				-simple_compile 1 \
        -Bangop 1"
		concurrent_genoff=" -batchsize $batchsize \
				-core_number $core_number \
        -Bangop 1 &>> $CURRENT_DIR/$log_file"

		genoff_cmd="$genoff_cmd $concurrent_genoff"
		echo "genoff_cmd: $genoff_cmd" &>> $CURRENT_DIR/$log_file
    echo "generating offline model..."
    
    eval "$genoff_cmd"
}

/bin/rm *.log &> /dev/null

for network in "${network_list[@]}"; do
   	proto_file=$CAFFE_MODELS_DIR/${network}/${network}${pt_name}
	  model_file=$CAFFE_MODELS_DIR/${network}/${network}${model_name}
		checkFile $proto_file
    checkFile $model_file
    do_run
done

/bin/rm ${MODELS_DIR}/*_twins &> /dev/null
/bin/rm ddr_type_graph.dot &> /dev/null
#/bin/rm ${CURRENT_DIR}/*.log &> /dev/null
/bin/rm nglog.json &> /dev/null

echo "----------------------"
# ${CURRENT_DIR}/env.sh
echo "Done!"

