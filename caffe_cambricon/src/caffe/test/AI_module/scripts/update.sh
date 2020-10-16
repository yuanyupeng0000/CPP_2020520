CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))
MODELS_DIR=$EXAMPLES_DIR/model_zoo
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

batchsize=4
core_number=4
echo  "SIMPLE_COMPILE: batchsize $batchsize core_number $core_number"

network_list=(
  alexnet
  c3d_v1.1
  densenet121
  densenet161
  densenet169
  densenet201
  faster-rcnn
  fcn
  googlenet
  inception-v3
  mobilenet_v1
  mobilenet_v2
  pvanet
  resnet101
  resnet152
  resnet18
  resnet34
  resnet50
  resnext101-32x4d
  resnext101-64x4d
  resnext26-32x4d
  resnext50-32x4d
  rfcn
  segnet
  squeezenet_v1.0
  squeezenet_v1.1
  ssd_mobilenetv1
  ssd_mobilenetv2
  ssd_vgg16
  vgg16
  vgg19
  yolov2
  yolov3
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
    mkdir -p ${MODELS_DIR}/${model_type}/${network} 
		echo ${MODELS_DIR}/${model_type}/${network}
    genoff_cmd="$CAFFE_DIR/build/tools/caffe genoff \
				-model $proto_file \
				-weights $model_file \
				-mcore $core_version \
        -mname ${MODELS_DIR}/${model_type}/${network}/${network} \
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
model_type='classification'

mkdir -p ${MODELS_DIR}/classification
mkdir -p ${MODELS_DIR}/face_recognition
mkdir -p ${MODELS_DIR}/object_detection
mkdir -p ${MODELS_DIR}/semantic_segmentation
mkdir -p ${MODELS_DIR}/video

for network in "${network_list[@]}"; do
    ## model
    if [ ${network} == 'ssd-vgg16' ] || [ ${network} == 'faster-rcnn' ] ||
       [ ${network} == 'mtcnn' ] || [ ${network} == 'pvanet' ] ||
       [ ${network} == 'rfcn' ] || [ ${network} == 'ssd_vgg16' ] ||
       [ ${network} == 'yolov2' ] || [ ${network} == 'yolov3' ] ||
       [ ${network} == 'ssd_mobilenetv1' ] || [ ${network} == 'ssd_mobilenetv2' ]; then
       model_type='object_detection'
    elif [ ${network} == 'fcn' ] || [ ${network} == 'segnet' ]; then
       model_type='semantic_segmentation'
    elif [ ${network} == 'c3d_v1.1' ]; then
       model_type='video'
    else
       model_type='classification' 
    fi
    
    proto_file=$CAFFE_MODELS_DIR/${network}/${network}${pt_name}
	  model_file=$CAFFE_MODELS_DIR/${network}/${network}${model_name}
		checkFile $proto_file
    checkFile $model_file
    do_run
done

mkdir -p ${MODELS_DIR}/face_detection/mtcnn
cp -rf ${CAFFE_DIR}/examples/mtcnn/*.cambricon ${MODELS_DIR}/face_detection/mtcnn

/bin/rm ${MODELS_DIR}/*/*/*_twins &> /dev/null
/bin/rm ddr_type_graph.dot &> /dev/null
/bin/rm ${CURRENT_DIR}/*.log &> /dev/null
/bin/rm nglog.json &> /dev/null
/bin/rm tfu_env.sh &> /dev/null
echo "----------------------"
# ${CURRENT_DIR}/env.sh
echo "Done!"

