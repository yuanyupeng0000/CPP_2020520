#! /usr/bin/env python
#################################################################################
#     File Name           :     caffe_test_result_cmp.py
#     Created By          :     jinzhenhui jinzhenhui@cambricon.com
#     Creation Date       :     [2018-08-23 13:14]
#     Last Modified       :     [2018-08-23 13:44]
#     Description         :     compare the errRate from two caffe test log
#################################################################################
import sys
import os
def generate_data(file,output_name):
    with open(file,'r') as f:
        content = [line.strip() for line in f.readlines() if 'caffe.cpp' in line]
        keyword = content[-1].split("caffe.cpp:")[1].split(']')[0]
        data = [line.split(' = ')[1].strip() for line in content if 'caffe.cpp:'+keyword in line]
    with open(output_name,'w') as f:
        for datum in  data:
            f.write(datum + '\n')

arg_len = len(sys.argv)
if arg_len != 3:
    print "Usage: python caffe_test_result_cmp.py log_a log_b"
else:
    generate_data(sys.argv[1],'data_a')
    generate_data(sys.argv[2],'data_b')
    os.system('python cmpData.py data_a data_b')
