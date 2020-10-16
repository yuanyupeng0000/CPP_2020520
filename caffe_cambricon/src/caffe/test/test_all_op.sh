# !/bin/bash
function getdir() {
  for element in `ls $1`
  do
    name="${element%.*}"
    echo -e "\ntesting "$name" ......."
    model="prototxts/$element"
    weights="$name.caffemodel"
    if [[ $name == "roi_align" ]]; then
        ./test_roi_align_on.sh $model $weights &>> test_all_op_result
    elif [[ $name == "nms" ]]; then
        ./test_nms_on.sh $model $weights &>> test_all_op_result
    elif [[ $name == "yolo_detect" ]]; then
        ./test_yolo_detect_on.sh $model $weights &>> test_all_op_result
    elif [[ $name == "psroi_pooling" ]]; then
        ./test_psroi_pooling_on.sh &>> test_all_op_result
    elif [[ $name == "roi_pooling" ]]; then
        ./test_roi_pooling_on.sh &>> test_all_op_result
    else
        ./test_op.sh $model $weights &>> test_all_op_result
    fi
    if (($?!=0));
      then
        echo "Work on "$name" failed"
      else
        cmp_data="python cmpData.py"
        echo "mlu/cpu  "$($cmp_data mlu_out cpu_out)
        echo "mfus/cpu "$($cmp_data mfus_out cpu_out)
        echo "mlu/mfus "$($cmp_data mlu_out mfus_out)
        rm "$name"_log*
    fi
    rm $weights 2>/dev/null
  done
}
rm test_all_op_result
caffe_dir="$(pwd)/prototxts"
getdir $caffe_dir
echo "result file: test_all_op_result"
