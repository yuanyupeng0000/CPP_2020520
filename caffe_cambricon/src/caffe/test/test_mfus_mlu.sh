#!/bin/bash

##test only layer
if [ ! -z ${DEBUG_CAFFE} ]; then
  SUFFIX="-d"
fi

mcore=MLU270

if [ $# != 3 ]; then
  echo "Usage: input_proto outputdir layerId(-1 to test all layers)"
  exit 1
fi

in_proto=$1
if [ ! -f $in_proto ]; then
  echo "input proto does not exist!"
  exit 1
fi

folder=$2

if [ ! -d "$folder" ]; then
    mkdir "$folder"
fi

folder=$folder"/"

python ../python/removeLayers.py $in_proto $3 $folder

echo "generated pt:"
ls $folder

for file in $folder/*pt; do
  ./test_op.sh $file test
  name=${file%\.pt}
  mv mfus_out $name"_mfus_out"
  mv mlu_out $name"_mlu_out"
  mv cpu_out $name"_cpu_out"
done

echo "All Finished! the output files are in $folder"
