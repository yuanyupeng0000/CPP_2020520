#!/bin/bash

if [ ! -z ${DEBUG_CAFFE} ]; then
   SUFFIX="-d"
fi

name=${1%\.prototxt}
name=${name##*/}
mcore=MLU270

echo "===== Test Op Offline about $name ====== "
echo "===== Create fake caffemodel ====== "
../build/tools/create_fake_caffemodel.bin${SUFFIX} $1 $2 &>log
if (($?!=0)); then
  echo "Create fake caffemodel failed"
  exit 1
fi

echo "===== Generate offline caffemodel ====="
../build/tools/caffe${SUFFIX} genoff --model $1 --weights $2 --mcore $mcore --mname $name
if (($?!=0)); then
  echo "Generate offline caffemodel failed"
  exit 1
fi

echo "===== Work on cpu ====== "
../build/tools/test_forward_online${SUFFIX}  $1 $2  0  $mcore 0.out &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi

echo "===== Test offline caffemodel ====="
../build/tools/test_forward_offline${SUFFIX} "$name".cambricon output_data subnet0
if (($?!=0)); then
  echo "test_forward_offline failed"
  exit 1
fi

#echo "===== Compare online offline results ====="
#python cmpData.py output_data_subnet0 0.out
