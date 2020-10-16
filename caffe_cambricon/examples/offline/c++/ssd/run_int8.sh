CUR_DIR=$(dirname $(readlink -f $0)) 
 echo "CUR_DIR: " $CUR_DIR
 if [[ "$#" -ne 1 ]]; then
 echo "Usage:"
 echo " $0 [MLU220|MLU270]"
 exit 1
 fi
 core_version=$1

pushd $CUR_DIR/../../../../src/caffe/examples/ssd/ &> /dev/null
./run_all_offline_sc.sh 1 $core_version
./run_all_offline_mc.sh 1 $core_version
popd &> /dev/null
