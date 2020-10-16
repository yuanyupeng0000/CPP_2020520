#!/bin/bash
if [ ! -z ${DEBUG_CAFFE} ]; then
    SUFFIX="-d"
fi

name=${1%\.prototxt}
name=${name##*/}
mcore=MLU270

function adjust_cpu_data_order() {
  file1_line=0
  file2_line=0

  # read the data of file1 into an array
  for line in $(cat $1)
  do
    file1_array[$file1_line]=$line
    file1_line=$[file1_line + 1]
  done

  # read the data of file2 into an array
  for line in $(cat $2)
  do
    file2_array[$file2_line]=$line
    file2_line=$[file2_line + 1]
  done

  # keep the same length of file1_array and file2_array,
  # if not same, use 0 to fill the difference.
  if [ $file1_line -gt $file2_line ];then
    for line in $(seq $file2_line $[file1_line - 1])
    do
      file2_array[$line]=0
    done
  elif [ $file1_line -lt $file2_line ];then
    for line in $(seq $file1_line $[file2_line - 1])
    do
      file1_array[$line]=0
    done
  fi

  i=0
  j=0
  tmp=0

  # modify cpu_output order, each of seven data is a group
  while [ ${file1_array[$i]} -eq 0 -a ${file1_array[$[i+1]]} -ne 0 ];do
    while [ ${file2_array[$j]} -eq 0 -a ${file2_array[$[j+1]]} -ne 0 ];do
      if [ ${file2_array[$[j+1]]} -eq ${file1_array[$[i+1]]} -a $i -ne $j ];then
        for k in $(seq 1 6)
        do
          tmp=${file2_array[$[j+k]]}
          file2_array[$[j+k]]=${file2_array[$[i+k]]}
          file2_array[$[i+k]]=$tmp
        done
      fi
      j=$[j + 7]
    done
    i=$[i + 7]
    j=$i
  done

  # after modifying cpu_output order, print the data of file2_array
  # into cpu_out file
  > $2
  for var in ${file2_array[@]};
  do
    echo $var >> $2
  done
}

echo "===== Test Op Offline about $name ====== "
echo "===== Create fake caffemodel ====== "
../build/tools/create_fake_caffemodel.bin${SUFFIX} $1 $2 &>log
if (($?!=0)); then
  echo "Create fake caffemodel failed"
  exit 1
fi

echo "===== Generate offline caffemodel ====="
../build/tools/caffe${SUFFIX} genoff --model $1 --weights $2 --mcore $mcore --mname $name
if (($?!=0)); then
  echo "Generate offline caffemodel failed"
  exit 1
fi

echo "===== Work on cpu ====== "
../build/test/test_yolo_detect_on${SUFFIX}  $1 $2  0 $mcore 0.out &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi

echo "===== Test offline caffemodel ====="
../build/test/test_yolo_detect_off${SUFFIX} "$name".cambricon output_data subnet0
if (($?!=0)); then
  echo "test_forward_offline failed"
  exit 1
fi

#Moved to calling script
echo "===== Compare online offline results ====="
adjust_cpu_data_order output_data_subnet0 0.out
python cmpData.py output_data_subnet0 0.out
