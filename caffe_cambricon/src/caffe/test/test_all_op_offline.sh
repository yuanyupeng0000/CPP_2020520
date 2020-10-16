# !/bin/bash

rm output_data_subnet* &> /dev/null

function getdir() {
  for element in `ls $1`
  do
    name="${element%.*}"
    echo "testing "$name" ......"
    model="prototxts/$element"
    weights="$name.caffemodel"
    if [[ $name == "roi_align" ]]; then
        ./test_roi_align_off.sh $model $weights &>> test_all_op_offline_result
    elif [[ $name == "nms" ]]; then
        ./test_nms_off.sh $model $weights &>> test_all_op_offline_result
    elif [[ $name == "yolo_detect" ]]; then
        ./test_yolo_detect_off.sh $model $weights &>> test_all_op_offline_result
    elif [[ $name == "psroi_pooling" ]]; then
        ./test_psroi_pooling_off.sh &>> test_all_op_offline_result
    elif [[ $name == "roi_pooling" ]]; then
        ./test_roi_pooling_off.sh &>> test_all_op_offline_result
    else
        ./test_op_offline.sh $model $weights &>> test_all_op_offline_result
    fi
    if (($?!=0));
      then
        echo -e "Work on "$name" failed\n"
      else
        ls output_data_subnet* &> file_list
        while read line
        do
          cat $line >> off_out
        done < file_list
        rm output_data_subnet*
        cmpresult=`python cmpData.py 0.out off_out`
        printf "%-20s %s  %s %s\n\n" $name ${cmpresult}

        rm "$name"_log* 2>/dev/null
    fi
    rm $weights &> /dev/null
    rm $name.cambricon* &> /dev/null
    rm off_out &> /dev/null
    rm file_list &> /dev/null
  done
}
rm test_all_op_offline_result 2>/dev/null
caffe_dir="$(pwd)/prototxts"
getdir $caffe_dir
echo "result file: test_all_op_offline_result"
