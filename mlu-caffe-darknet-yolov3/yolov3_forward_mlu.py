#!/usr/bin/env python2

import os
os.environ["GLOG_minloglevel"] = '5'
os.environ['TFU_ENABLE']='1' 
os.environ['TFU_NET_FILTER']='0' 
os.environ['CNRT_PRINT_INFO']='false' 
os.environ['CNRT_GET_HARDWARE_TIME']='false'
os.environ['CNML_PRINT_INFO']='false'  
import caffe
import math
import shutil
import stat
import subprocess
import sys
import numpy as np
import collections
import copy
import time
import traceback
import datetime
import cv2

np.set_printoptions(threshold=np.nan)

def get_boxes(prediction, batch_size, img_size=416):
    """
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    reshape_value = prediction.reshape((-1, 1))

    num_boxes_final = reshape_value[0].item()
    print(num_boxes_final)
    all_list = [[] for _ in range(batch_size)]
    max_limit = 1
    min_limit = 0
    for i in range(int(num_boxes_final)):
        batch_idx = int(reshape_value[64 + i * 7 + 0].item())
        if batch_idx >= 0 and batch_idx < batch_size:
            bl = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 3].item()) * img_size)
            br = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 4].item()) * img_size)
            bt = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 5].item()) * img_size)
            bb = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 6].item()) * img_size)

            if bt - bl > 0 and bb -br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 1].item())

    output = [np.array(all_list[i]).reshape(-1, 7) for i in range(batch_size)]
    return output
    
if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage:{} prototxt caffemodel".format(sys.argv[0]))
        sys.exit(1)
    
    batch_size=1
    prototxt=sys.argv[1]
    caffemodel=sys.argv[2]
    
    caffe.set_mode_mfus()
    #caffe.set_mode_mlu()
    caffe.set_core_number(1)
    caffe.set_batch_size(batch_size)
    caffe.set_simple_flag(1)
    
    caffe.set_rt_core("MLU270")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    input_name = net.blobs.keys()[0]

    if 0:
        input_img=cv2.imread("dog.jpg")      
        input_img=cv2.resize(input_img, (416,416))
        image = input_img[:, :, (2, 1, 0)]
        
        image=image.transpose((2,0,1))
        image=image[np.newaxis, :]
        images = np.repeat(image, batch_size, axis=0)
        
    else:
        img=cv2.imread("dog.jpg")      
        img = img[:, :, (2, 1, 0)]
        
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = input_img.shape
        input_img = cv2.resize(input_img, (416, 416), interpolation = cv2.INTER_AREA)
        cv2.imwrite("crop.jpg",input_img)    
        
        image = np.transpose(input_img, (2, 0, 1))
        image=image[np.newaxis, :].astype(np.float32)
        images = np.repeat(image, batch_size, axis=0)
    
    net.blobs[input_name].data[...]=images

    #output = net.forward()
    output = net.forward()
   
    output_keys=output.keys()
    print("lzc print ", output_keys)
    output=output[output_keys[0]].astype(np.float32)    
    outputs = get_boxes(output, batch_size, img_size=416)    
    out_image=input_img
    
    for si, pred in enumerate(outputs):
        img_h, img_w, _ = input_img.shape
        box = pred[:, :4]  #x1, y1, x2, y2
        scaling_factors = min(float(416) / img_w, float(416) / img_h)

        box[:, 0] = ((box[:, 0] - (416 - scaling_factors * img_w) / 2.0) / scaling_factors) / img_w
        box[:, 1] = ((box[:, 1] - (416 - scaling_factors * img_h) / 2.0) / scaling_factors) / img_h
        box[:, 2] = ((box[:, 2] - (416 - scaling_factors * img_w) / 2.0) / scaling_factors) / img_w
        box[:, 3] = ((box[:, 3] - (416 - scaling_factors * img_h) / 2.0) / scaling_factors) / img_h

        for di, d in enumerate(pred):
            box_temp = []
            box_temp.append(round(box[di][0], 3) * img_w)
            box_temp.append(round(box[di][1], 3) * img_h)
            box_temp.append((round(box[di][2], 3) - round(box[di][0], 3))  * img_w)
            box_temp.append((round(box[di][3], 3) - round(box[di][1], 3))  * img_h)
            if d[5]<0.2:
                continue     
                
            x1,y1,x2,y2=box_temp[0],box_temp[1],box_temp[0]+box_temp[2],box_temp[1]+box_temp[3]
            print("box:{},{},{},{} score:{} id:{}".format(x1,y1,x2,y2,round(d[5], 5),int(d[6])))           
            
            cv2.rectangle(out_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
            
    cv2.imwrite("mlu_result.jpg",out_image)    
