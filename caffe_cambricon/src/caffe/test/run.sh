#!/bin/bash

export LD_LIBRARY_PATH=../build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=../../cnml/build:$LD_LIBRARY_PATH

mcore=MLU270

#name: main
#=====================
#function: test net
#=====================
#args: null
function main ()
{
  iter=0
  rm -rf ./prototxts/*.caffemodel
  ls ./prototxts/* > file_list  #comment this line, edit file_list mannually
  for line in `cat file_list`;
  do
    iter=$((iter+1))
    echo "======Iterations ${iter}========"
    echo "Testing $line ..."

    weights=${line%.prototxt*}
    weights=${weights}".caffemodel"
    echo "Generating random weights: ${weights}"
    ../build/tools/create_fake_caffemodel.bin $line $weights &> create_fake_caffemodel.log
    if (($?!=0)); then
      echo "Creating fake caffemodel failed!"
      exit 1
    fi

    echo "CPU mode ..."
    ../build/tools/test_forward $line $weights 0 $mcore cpu_output &> forward_cpu.log
    if (($?!=0)); then
      echo "Executing on CPU failed!"
      exit 2
    fi

    echo "MLU mode ..."
    ../build/tools/test_forward $line $weights 2 $mcore mlu_output 2> forward_mlu.log
    if (($?!=0)); then
      echo "Executing on MLU failed!"
      exit 3
    fi

    echo "Compare results ..."
    python cmpData.py cpu_output mlu_output > diff_result
    if (($?!=0)); then
      echo "Comparing failed!"
      exit 4
    fi

    echo "Estimate diff-result ..."
    err_str=`grep "errRate" diff_result`
    err_str=${err_str#*= }
    echo "error rate is ${err_str}"
    err_str=`echo $err_str|awk '{printf("%f\n"),$0;}'`
    if [ `echo "$err_str > 0.05"|bc` -eq 1 ]; then
      echo "Warning, err_rate is greater than 0.05"
      exit 5
    elif [ `echo "$err_str > 0.01"|bc` -eq 1 ]; then
      echo "Error, err_rate is greater than 0.01"
    fi


  done
}


main
