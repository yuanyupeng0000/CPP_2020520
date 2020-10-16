CUR_DIR=$(dirname $(readlink -f $0)) 
 echo "CUR_DIR: " $CUR_DIR
 if [[ "$#" -ne 1 ]]; then
 echo "Usage:"
 echo " $0 [MLU220|MLU270]"
 exit 1
 fi
 core_version=$1

pushd $CUR_DIR/../../../../src/caffe/examples/clas_online_singlecore/ &> /dev/null
./run_all_online_sc.sh 1 $core_version
popd &> /dev/null

pushd $CUR_DIR/../../../../src/caffe/examples/clas_online_multicore/ &> /dev/null
./run_all_online_mc.sh 1 $core_version
popd &> /dev/null
