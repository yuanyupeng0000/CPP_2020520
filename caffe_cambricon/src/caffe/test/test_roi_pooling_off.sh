#!/bin/bash
if [ ! -z ${DEBUG_CAFFE} ]; then
    SUFFIX="-d"
fi

name="roi_pooling"
mcore=MLU270
cpu_model=./special_proto/roi_pooling_cpu.prototxt
mlu_model=./special_proto/roi_pooling_mlu.prototxt
cpu_fake=roi_pooling_cpu.caffemodel
mlu_fake=roi_pooling_mlu.caffemodel

echo "===== Test Op Offline about $name ====== "
echo "===== Create fake caffemodel ====== "
../build/tools/create_fake_caffemodel.bin${SUFFIX} $cpu_model $cpu_fake &>log
if (($?!=0)); then
  echo "Create cpu fake caffemodel failed"
  exit 1
fi

../build/tools/create_fake_caffemodel.bin${SUFFIX} $mlu_model $mlu_fake &>log
if (($?!=0)); then
  echo "Create mlu fake caffemodel failed"
  exit 1
fi

echo "===== Generate offline caffemodel ====="
../build/tools/caffe${SUFFIX} genoff --model $mlu_model --weights $mlu_fake --mcore ${mcore} --mname $name
if (($?!=0)); then
  echo "Generate offline caffemodel failed"
  exit 1
fi

echo "===== Work on cpu ====== "
../build/test/test_roi_pooling_on${SUFFIX}  $cpu_model $cpu_fake  0 ${mcore} 0.out &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi

echo "===== Test offline caffemodel ====="
../build/test/test_roi_pooling_off${SUFFIX} "$name".cambricon output_data subnet0
if (($?!=0)); then
  echo "test_forward_offline failed"
  exit 1
fi

rm $cpu_fake
rm $mlu_fake
#Moved to calling script
#echo "===== Compare online offline results ====="
#python cmpData.py output_data_subnet0 0.out
