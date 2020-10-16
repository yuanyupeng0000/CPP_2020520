#!/usr/bin/env python

import sys
import os

def replace(dir, str1, str2):
    if os.path.isfile(dir):
        os.system("sed -i s/%s/%s/g %s" %(str1, str2, dir))
        print ("sed -i s/%s/%s/g %s" %(str1, str2, dir))
    elif os.path.exists(dir):
        for item in os.listdir(dir):
            dir = dir.rstrip('/')
            sub_dir = dir + '/' + item
            if os.path.isfile(sub_dir):
                os.system("sed -i s/%s/%s/g %s" %(str1, str2, sub_dir))
                print ("sed -i s/%s/%s/g %s" %(str1, str2, sub_dir))
            else:
                replace(sub_dir, str1, str2)

if __name__ == "__main__":
    replace(sys.argv[1], sys.argv[2], sys.argv[3])
