#!/bin/bash

##test op about two bottoms

name=${1%\.prototxt}
name=${name##*/}

echo "=====  Create fake caffemodel: $2===="
../build/tools/create_fake_caffemodel.bin $1 $2 &>log
if (($?!=0)); then
    echo "Create fake caffemodel failed"
    exit 1
fi

echo "====  Work on cpu ===="
../build/tools/caffe test -model $1 -weights $2 -iterations 1 &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi
echo "==== save cpu_resut ===="
sed -n '/caffe.cpp:405] /p' "$name"_log |awk '{print $NF}' > data_cpu
echo "==== save successfully:  in data_cpu ===="


echo "====  Work on mlu ===="
../build/tools/caffe test -model $1 -weights $2 -mmode MLU -mcore MLU270  -iterations 1 &> "$name"_log1
if (($?!=0)); then
  echo "Work on mlu failed"
  exit 1
fi
echo "==== save mlu_resut ===="
sed -n '/caffe.cpp:405] /p' "$name"_log1 |awk '{print $NF}' > data_mlu
echo "==== save successfully:  in data_mlu ===="

echo "====  Work on mfus ===="
../build/tools/caffe test -model $1 -weights $2 -mmode MFUS -mcore MLU270 -iterations 1 &> "$name"_log2
if (($?!=0)); then
  echo "Work on mlu failed"
  exit 1
fi
echo "==== save mfus_resut ===="
sed -n '/caffe.cpp:405] /p' "$name"_log2 |awk '{print $NF}' > data_mfus
echo "==== save successfully:  in data_mfus ===="

echo "==== comparing mlu cpu ===="
python cmpData.py data_mlu data_cpu
echo "==== comparing mfus cpu ===="
python cmpData.py data_mfus data_cpu
echo "==== comparing mfus mlu ===="
python cmpData.py data_mfus data_mlu
