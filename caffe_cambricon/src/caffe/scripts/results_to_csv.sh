#!/bin/bash
# usage:
# 1) ./results_to_csv.sh <test_log>
# run all classification networks tests with float16

if [[ "$#" -ne 1 ]]; then
    echo "Usage:"
    echo "./results_to_csv.sh <test_log>"
    echo "The log file is the result after running scripts of examples/<network>/run_all_*.sh"
    exit -1
fi

log_file=$1
echo "log file is: $log_file"

file_nw="tmp_nw"
file_df="tmp_df"
file_dp="tmp_dp"
file_mp="tmp_mp"
file_acc1="tmp_acc1"
file_acc5="tmp_acc5"
file_hw_fps="tmp_hw_fps"
file_e2e_fps="tmp_e2e_fps"

echo "put the key information into the independent temp files..."
# get network name
grep -oP 'running \K.*(?= offline -)' $log_file > $file_nw

# get float16,dense,etc
grep -oP '\- \K.*(?=\.\.\.)' $log_file > $file_df

grep -oP 'accuracy1: \K.*(?= \()' $log_file > $file_acc1
grep -oP 'accuracy5: \K.*(?= \()' $log_file > $file_acc5
grep -oP 'Hardware fps: \K.*' $log_file > $file_hw_fps
grep -oP 'End2end throughput fps: \K.*' $log_file > $file_e2e_fps

# loop the temp files and concat all the stat data into one line
csv_file="${log_file##*/}.csv"
echo "csv file: $csv_file"
echo "network,dataType,dataMode,accuracy1,accuracy5,hardwareFps,end2endFps" > $csv_file
index=1
echo "aggregating result's data..."
while IFS='' read -r nw || [[ -n "$nw" ]]; do
    cmd="sed -n '${index}p' $file_df"
#    echo "$cmd"
    df=`eval $cmd`
    cmd="sed -n '${index}p' $file_dp"
    cmd="sed -n '${index}p' $file_mp"
    cmd="sed -n '${index}p' $file_acc1"
    acc1=`eval $cmd`
    cmd="sed -n '${index}p' $file_acc5"
    acc5=`eval $cmd`
    cmd="sed -n '${index}p' $file_hw_fps"
    hw_fps=`eval $cmd`
    cmd="sed -n '${index}p' $file_e2e_fps"
    e2e_fps=`eval $cmd`

    echo "$nw,$df,$acc1,$acc5,$hw_fps,$e2e_fps" >> $csv_file
    index=$((index+1))
done < $file_nw

/bin/rm -f tmp*
echo "Done."
