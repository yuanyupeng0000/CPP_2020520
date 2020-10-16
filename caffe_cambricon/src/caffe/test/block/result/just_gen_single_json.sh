#/bin/sh
#get block name
BLOCK_PT=`ls *.pt`
BLOCK_NAME=${BLOCK_PT%.*}
BLOCK_CAFFEMODEL="$BLOCK_NAME.caffemodel"
echo "BLOCK_NAME: $BLOCK_NAME"
rm block.log 2> /dev/null
../../../build/tools/create_fake_caffemodel.bin $BLOCK_PT $BLOCK_CAFFEMODEL 1>> block.log 2>> block.log
../../../build/tools/caffe genoff -model $BLOCK_PT -weights $BLOCK_CAFFEMODEL -mcore MLU270 -simple_compile 1  -batchsize 1 -core_number 1 -output_dtype FLOAT16 1>> block.log 2>> block.log
