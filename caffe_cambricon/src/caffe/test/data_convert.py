#! /usr/bin/env python
#################################################################################
#     File Name           :     data_encoder.py
#     Created By          :     jinzhenhui jinzhenhui@cambricon.com
#     Creation Date       :     [2018-08-23 10:04]
#     Last Modified       :     [2018-08-23 13:45]
#     Description         :     encode data file in to an array in hpp, for test
#################################################################################

# Usage: to encode data to hpp array
# python data_convert encode data_file
# to get the data back from hpp array
# python data_convert decode data_file

import sys
arg_len = len(sys.argv)
file_path = sys.argv[-1]
if arg_len != 3 or sys.argv[1] not in ['encode','decode']:
    print("Usage:\nto encode data to hpp array")
    print("python data_convert encode data_file")
    print("to get the data back from hpp array")
    print("python data_convert decode data_file")
elif sys.argv[1] == 'encode':
    with open(file_path,'r') as f:
        data = [line.strip() for line in f.readlines()]
    with open('input_data.hpp','w') as f:
        f.write('namespace input_data {\n')
        f.write('static float array[] = {')
        f.write(','.join(data))
        f.write('};\n}')
    print("convert succeeded.output file:input_data.hpp")
    print('#include "input_data.hpp"')
    print('and use input_data::array[i]')
else:
    with open(file_path,'r') as f:
        content = ''.join(f.readlines())
    numbers = content.split('= {')[1].split('};')[0].split(',')
    with open('input_data','w') as f:
        for number in numbers:
            f.write(number+'\n')
    print("convert succeeded.ouput file:input_data")
