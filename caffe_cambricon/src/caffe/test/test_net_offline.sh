#!/bin/bash
# test_net_offline.sh
# ./test_net_offline.sh model.prototxt weights.caffemodel mean_file label file_list

mcore=MLU270

name=${1%\.prototxt}
name=${name##*/}

echo "===== Offline model testing ====== "
echo "===== Generate offline caffemodel ====="
../build/tools/caffe genoff --model $1 --weights $2 --mcore $mcore --mname $name
if (($?!=0)); then
  echo "Generate offline caffemodel failed"
  exit 1
fi

echo "===== Work on clas_online_singlecore ====== "
../build/examples/clas_online_singlecore/clas_online_singlecore $1 $2 $3 $4 $5 0 $mcore
if (($?!=0)); then
  echo "Work on clas_online_singlecore failed"
  exit 1
fi

echo "===== Work on clas_offline_singlecore ====="
../build/examples/clas_offline_singlecore/clas_offline_singlecore "$name".cambricon $5 $4 $3 0
if (($?!=0)); then
  echo "Work on clas_offline_singlecore failed"
  exit 1
fi

#echo "===== Compare online offline results ====="
#python cmpData.py offline_output online_output
