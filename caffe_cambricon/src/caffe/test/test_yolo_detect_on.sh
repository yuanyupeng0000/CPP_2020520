#!/bin/bash
if [ ! -z ${DEBUG_CAFFE} ]; then
    SUFFIX="-d"
fi

##test yolo detect out layer

mcore=MLU270

name=${1%\.prototxt}
name=${name##*/}

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

echo "===== Test Op about $name ====== "
echo "===== Create fake caffemodel ====== "
../build/tools/create_fake_caffemodel.bin${SUFFIX} $1 $2 &>log
if (($?!=0)); then
  echo "Create fake caffemodel failed"
  exit 1
fi

echo "===== Work on cpu ====== "
../build/test/test_yolo_detect_on${SUFFIX}  $1 $2  0 $mcore cpu_out &> "$name"_log
if (($?!=0)); then
  echo "Work on cpu failed"
  exit 1
fi

echo "===== Work on mlu ====== "
../build/test/test_yolo_detect_on${SUFFIX}  $1 $2  1 $mcore mlu_out &> "$name"_log1
if (($?!=0)); then
  echo "Work on mlu failed"
  exit 1
fi

echo "===== Work on mfus ====== "
../build/test/test_yolo_detect_on${SUFFIX}  $1 $2  2 $mcore mfus_out &> "$name"_log2
if (($?!=0)); then
  echo "Work on mfus failed"
  exit 1
fi

#cmp_data="../build/tools/cmp_data"
cmp_data="python cmpData.py"
echo "===== Compare mlu cpu results ====== "
adjust_cpu_data_order mlu_out cpu_out
echo "errRate = "$($cmp_data mlu_out cpu_out)

echo "===== Compare mfus cpu results ====== "
echo "errRate = "$($cmp_data mfus_out cpu_out)

echo "===== Compare mlu mfus results ====== "
echo "errRate = "$($cmp_data mlu_out mfus_out)
