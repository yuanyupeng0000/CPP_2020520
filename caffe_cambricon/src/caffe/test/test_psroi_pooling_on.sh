#!/bin/bash

if [ ! -z ${DEBUG_CAFFE} ]; then
    SUFFIX="-d"
fi

##test only layer

mcore=MLU270

name="psroi_pooling"

cpu_model=./special_proto/psroi_pooling_cpu.prototxt
mlu_model=./special_proto/psroi_pooling_mlu.prototxt
cpu_fake="fake_cpu_psroi.caffemodel"
mlu_fake="fake_mlu_psroi.caffemodel"

echo "===== Test Op about psroi pooling ====== "
echo "===== Create fake caffemodel ====== "
../build/tools/create_fake_caffemodel.bin${SUFFIX} $cpu_model $cpu_fake &>log
if (($?!=0)); then
  echo "Create cpu fake caffemodel failed"
  exit 1
fi

../build/tools/create_fake_caffemodel.bin${SUFFIX} $mlu_model $mlu_fake &> log
if (($?!=0)); then
 echo "Create mlu fake caffemodel failed"
 exit 1
fi

echo "===== Work on cpu ====== "
../build/test/test_psroi_pooling${SUFFIX}  $cpu_model $cpu_fake  0 $mcore cpu_out &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi

echo "===== Work on mlu ====== "
../build/test/test_psroi_pooling${SUFFIX}  $mlu_model $mlu_fake  1 $mcore mlu_out &> "$name"_log1
if (($?!=0)); then
  echo "Work on mlu failed"
  exit 1
fi

echo "===== Work on mfus ====== "
../build/test/test_psroi_pooling${SUFFIX}  $mlu_model $mlu_fake  2 $mcore mfus_out &> "$name"_log2
if (($?!=0)); then
  echo "Work on mfus failed"
  exit 1
fi

rm $cpu_fake
rm $mlu_fake
#cmp_data="../build/tools/cmp_data"
#cmp_data="python cmpData.py"
#echo "===== Compare mlu cpu results ====== "
#echo $($cmp_data mlu_out cpu_out)

#echo "===== Compare mfus cpu results ====== "
#echo $($cmp_data mfus_out cpu_out)

#echo "===== Compare mlu mfus results ====== "
#echo $($cmp_data mlu_out mfus_out)