#!/bin/bash

##test only layer
if [ ! -z ${DEBUG_CAFFE} ]; then
  SUFFIX="-d"
fi

mcore=MLU270

name=${1%\.prototxt}
name=${name##*/}

echo "===== Test Op about $name ====== "
echo "===== Create fake caffemodel ====== "
../build/tools/create_fake_caffemodel.bin${SUFFIX} $1 $2 &>log
if (($?!=0)); then
  echo "Create fake caffemodel failed"
  exit 1
fi

echo "===== Work on cpu ====== "
../build/tools/test_forward_online${SUFFIX}  $1 $2  0 $mcore cpu_out &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi

echo "===== Work on mlu ====== "
../build/tools/test_forward_online${SUFFIX}  $1 $2  1 $mcore mlu_out &> "$name"_log1
if (($?!=0)); then
  echo "Work on mlu failed"
  exit 1
fi

echo "===== Work on mfus ====== "
../build/tools/test_forward_online${SUFFIX}  $1 $2  2 $mcore mfus_out &> "$name"_log2
if (($?!=0)); then
  echo "Work on mfus failed"
  exit 1
fi

#cmp_data="../build/tools/cmp_data"
cmp_data="python cmpData.py"
echo "===== Compare mlu cpu results ====== "
echo $($cmp_data mlu_out cpu_out)

echo "===== Compare mfus cpu results ====== "
echo $($cmp_data mfus_out cpu_out)

echo "===== Compare mlu mfus results ====== "
echo $($cmp_data mlu_out mfus_out)
