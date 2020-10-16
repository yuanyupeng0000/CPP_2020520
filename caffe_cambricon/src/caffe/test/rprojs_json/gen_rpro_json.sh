#export RPROJ_MODELS_PATH=./your/R/models/path
network_list=(
face_detection
ocr_recognizer
ocr_detection
ratio_prediction
face_feature_extraction
densenet201
vgg16
inception-v3
resnet50
)

batch_list=(
1
4
16
)

qmode_value=(
int8
int16
)

dtype_value=(
FLOAT16
FLOAT32
)
do_run()
{
    genoff_cmd="../../build/tools/caffe genoff -model $RPROJ_MODELS_PATH/$qmode/${network}_${qmode}.prototxt \
                                   -weights $RPROJ_MODELS_PATH/${network}.caffemodel \
                                   -mcore MLU270 \
                                   -simple_compile 1  \
                                   -batchsize $batch_size \
                                   -core_number 1 \
                                   -output_dtype $dtype &> rproj.log "
      echo "---------------------------------------------------------------"
      echo "generating ${network} mode:${qmode} batchsize :${batch_size} dtype:${dtype} offline model..."
      eval "$genoff_cmd"
      if [[ "$?" -eq 0 ]]; then
        json_file=${network}_${qmode}_${batch_size}_${dtype}.json
	      if [ -f jsonfiles/abstract_fusion_1.json ];then
           echo "save jsonfile ./${network}/${json_file} success!"
		       mv jsonfiles/abstract_fusion_1.json ./${network}/${json_file}
        fi
      else
        echo "generating offline model failed!"
      fi
}
export CNML_ADDITIONAL_DEBUG=PrintJsonFile
for network in "${network_list[@]}"; do
    if [ ! -d "${network}" ]; then
      mkdir -p "${network}"
    fi
    for bscn in "${batch_list[@]}"; do
        batch_size=${bscn}
        for qmode in "${qmode_value[@]}"; do
            for dtype in "${dtype_value[@]}";do
                do_run
            done
        done
    done
done
