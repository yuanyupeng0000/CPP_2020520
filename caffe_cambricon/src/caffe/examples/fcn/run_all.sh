#!/bin/bash

echo ">>>>>>>>>>>>>>>>>>run_all_offline_mc<<<<<<<<<<<<<<<<"
./run_all_offline_mc.sh 1 MLU270

echo ">>>>>>>>>>>>>>>>>>run_all_offline_sc<<<<<<<<<<<<<<<<"
./run_all_offline_sc.sh 1 MLU270

echo ">>>>>>>>>>>>>>>>>>run_all_online_mc<<<<<<<<<<<<<<<<"
./run_all_online_mc.sh 1  2 MLU270

echo ">>>>>>>>>>>>>>>>>>run_all_online_sc<<<<<<<<<<<<<<<<"
./run_all_online_sc.sh 1  2 MLU270

sed -i "s/OUTPUT_MODE=\"FLOAT32\"/OUTPUT_MODE=\"FLOAT16\"/g" `grep "OUTPUT_MODE=\"FLOAT32\"" -lr ../../scripts/set_caffe_module_env.sh`

echo ">>>>>>>>>>>>>>>>>>run_all_offline_mc<<<<<<<<<<<<<<<<"
./run_all_offline_mc.sh 1 MLU270

echo ">>>>>>>>>>>>>>>>>>run_all_offline_sc<<<<<<<<<<<<<<<<"
./run_all_offline_sc.sh 1 MLU270

echo ">>>>>>>>>>>>>>>>>>run_all_online_mc<<<<<<<<<<<<<<<<"
./run_all_online_mc.sh 1  2 MLU270

echo ">>>>>>>>>>>>>>>>>>run_all_online_sc<<<<<<<<<<<<<<<<"
./run_all_online_sc.sh 1  2 MLU270
