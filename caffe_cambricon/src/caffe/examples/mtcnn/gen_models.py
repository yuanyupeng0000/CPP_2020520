#!/usr/bin/env python2
import argparse

parser = argparse.ArgumentParser(description='generate offline model')
parser.add_argument('--debug-suffix', default='', help='debug suffix for caffe')
parser.add_argument('--model-path', default='', help='fddb path')
parser.add_argument('--file-list', default='file_list', help='image file list')
parser.add_argument('--caffe-dir', default='', help='caffe directory')
parser.add_argument('--core-version', default='MLU100', help='core version')

args = parser.parse_args()
image_list_filepath = args.file_list
model_path = args.model_path
debug_suffix = args.debug_suffix
caffe_dir = args.caffe_dir
core_version = args.core_version

import os
import sys
os.environ["GLOG_minloglevel"] = '5'
sys.path.append(os.path.join(caffe_dir,"python"))

import caffe
from google.protobuf import text_format
from caffe import layers as L, params as P, to_proto
import caffe.proto.caffe_pb2 as caffe_pb2
from subprocess import Popen, PIPE
import numpy as np
import copy
import traceback
#from tqdm import tqdm

'''
create int8 prototxt
'''
channel_quantization_flag="int8_channel"
channel_quantization_flag="common"

transform_param_scale = 0.0078125
transform_param_mean_value_r = 127.5
transform_param_mean_value_g = 127.5
transform_param_mean_value_b = 127.5
batch_size = 16

convert_int8_template = '''
[model]
original_models_path = {}
save_model_path = {}

[data]
images_folder_path = {}
used_images_num = {}

[weights]
original_weights_path = {}

[preprocess]
mean = {},{},{}
std = {}
scale = {}, {}
crop = {},{}

[config]
int8_op_list = Conv, FC, LRN
use_firstconv = 1
'''
def gen_convert_int8_config(source_prototxt, source_caffemodel, target_prototxt, input_shape):
    original_models_path = source_prototxt
    save_model_path = target_prototxt
    images_folder_path = image_list_filepath
    original_weights_path = source_caffemodel
    mean_r, mean_g, mean_b = transform_param_mean_value_r, transform_param_mean_value_g, transform_param_mean_value_b
    std = transform_param_scale
    ni, ci, hi, wi = input_shape
    scale_h, scale_w = hi, wi
    used_images_num=batch_size
    config = convert_int8_template.format(original_models_path,
                                          save_model_path,
                                          images_folder_path,
                                          used_images_num,
                                          original_weights_path,
                                          mean_r, mean_g, mean_b,
                                          std,
                                          scale_h, scale_w,
                                          scale_h, scale_w)
    return config


def excute(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE)
    ret, _ = p.communicate()
    #TODO
    if p.returncode != 0:
        # print ("cmd:{} ret:{} code:{} ".format(cmd, ret, p.returncode))
        sys.exit(-1)
        return False
    return True

def get_positon_normal(source_prototxt, source_caffemodel, target_prototxt,input_shape):

    config=gen_convert_int8_config(source_prototxt, source_caffemodel, target_prototxt, input_shape)
    #print(config)

    ini_file_path="convert_quantized.ini"
    with open(ini_file_path,"w+") as f:
        f.write(config)

    cmdline = "{}/build/tools/generate_quantized_pt{} -ini_file {} -mode {} 2>&1 > /dev/null".format(
        caffe_dir, debug_suffix, ini_file_path,channel_quantization_flag)
    # print(cmdline)
    return excute(cmdline)


def genoff(source_prototxt, source_caffemodel, target_prefix):
    batchsize = 1
    core_number = 1
    if source_prototxt[3] == '2':
        batchsize = 16
        core_number = 1
    if source_prototxt[3] == '3':
        batchsize = 16
        core_number = 1
    if core_version in ["1H8"]:
        cmdline = "{}/build/tools/caffe{} genoff -model {} -weights {} -mname  {} -mcore {}".format(
            caffe_dir, debug_suffix, source_prototxt, source_caffemodel, target_prefix, core_version)
    else:
        cmdline = "{}/build/tools/caffe{} genoff -model {} -weights {} -mname  {} -mcore {} -batchsize {} -core_number {}".format(
            caffe_dir, debug_suffix, source_prototxt, source_caffemodel, target_prefix, core_version, batchsize, core_number)
    print(cmdline)
    return excute(cmdline)


def get_input_shape(net_param):
    shape = []
    shape.append(1)
    for i in range(len(net_param.layer)):
        layer = net_param.layer[i]
        if layer.type not in ["ImageData"]:
            continue
        if layer.image_data_param.is_color == True:
            shape.append(3)
        else:
            shape.append(1)
        shape.append(layer.image_data_param.new_height)
        shape.append(layer.image_data_param.new_width)
        return shape

    shape = []
    if net_param.input_shape and len(net_param.input_shape) > 0:
        for i in net_param.input_shape[0].dim:
            shape.append(int(i))
        return shape

    shape = []
    if net_param.input_dim:
        for i in net_param.input_dim:
            shape.append(i)
        return shape

    return None


def create_int8_prototxt_For_get_position_normal(source_prototxt_fp32, prototxt_for_get_position_normal):

    net_param = caffe_pb2.NetParameter()
    with open(source_prototxt_fp32, "r") as f:
        net_param = text_format.Merge(str(f.read()), net_param)

    input_shape = get_input_shape(net_param)
    #print(input_shape)

    new_net_param = caffe_pb2.NetParameter()
    obj = new_net_param.layer.add()
    obj.name = "data"
    obj.type = "ImageData"
    obj.top.append("data")
    obj.top.append("label")

    obj.transform_param.scale = transform_param_scale
    obj.transform_param.mean_value.append(transform_param_mean_value_r)
    obj.transform_param.mean_value.append(transform_param_mean_value_g)
    obj.transform_param.mean_value.append(transform_param_mean_value_b)

    obj.image_data_param.source = os.path.join(
        image_list_dir, image_list_filepath)
    obj.image_data_param.root_folder = image_list_dir
    obj.image_data_param.new_height = input_shape[2]
    obj.image_data_param.new_width = input_shape[3]

    if input_shape[1] == 3:
        obj.image_data_param.is_color = True
    else:
        obj.image_data_param.is_color = False

    obj.image_data_param.shuffle = False
    obj.image_data_param.batch_size = batch_size

    idx = 0
    for i in range(len(net_param.layer)):
        layer = net_param.layer[i]
        if layer.type in ["ImageData", "Data"]:
            idx += 1
            continue
        obj = new_net_param.layer.add()
        obj.CopyFrom(layer)
        if idx == 0:
            obj.bottom[0] = "data"
        idx += 1

    with open(prototxt_for_get_position_normal, "w") as f:
        f.write(str(new_net_param))


def create_deploy_prototxt(source_prototxt, deploy_prototxt,input_shape):
    net_param = caffe_pb2.NetParameter()
    with open(source_prototxt, "r") as f:
        net_param = text_format.Merge(str(f.read()), net_param)

    new_net_param = caffe_pb2.NetParameter()
    new_net_param.input.append("data")
    for i in input_shape:
        new_net_param.input_dim.append(int(i))

    idx = 0
    for i in range(len(net_param.layer)):
        layer = net_param.layer[i]
        if layer.type in ["ImageData", "Data","Input"]:
            continue
        obj = new_net_param.layer.add()
        obj.CopyFrom(layer)
        idx += 1

    with open(deploy_prototxt, "w") as f:
        f.write(str(new_net_param))

def resize_prototxt(source_prototxt, target_prototxt, input_shape):
    net_param = caffe_pb2.NetParameter()
    with open(source_prototxt, "r") as f:
        net_param = text_format.Merge(str(f.read()), net_param)

    new_net_param = caffe_pb2.NetParameter()
    new_net_param.input.append("data")
    for i in input_shape:
        new_net_param.input_dim.append(i)
    idx = 0
    for i in range(len(net_param.layer)):
        layer = net_param.layer[i]
        if layer.type in ["ImageData", "Data"]:
            idx += 1
            continue
        obj = new_net_param.layer.add()
        obj.CopyFrom(layer)
        if idx == 0:
            obj.bottom[0] = "data"
        idx += 1

    with open(target_prototxt, "w") as f:
        f.write(str(new_net_param))

def create_mtcnn_int8_models():
    input_shapes = [
        [384, 216],
        [288, 162],
        [216, 122],
        [162, 92],
        [122, 69],
        [92, 52],
        [69, 39],
        [52, 29],
        [39, 22],
        [29, 17]]

    target_models = []
    input_shape = [1, 3, 32, 32]
    for i in input_shapes:
        input_shape[3] = i[1]
        input_shape[2] = i[0]
        target_prototxt = "det1_{}x{}.prototxt".format(i[0], i[1])
        target_models.append({"prototxt": target_prototxt,
                              "caffemodel": "{}/det1.caffemodel".format(model_path),
                              "input_shape": copy.deepcopy(input_shape)})
        resize_prototxt("{}/det1.prototxt".format(model_path), target_prototxt, input_shape)

    input_shape = [16, 3, 24, 24]
    target_prototxt = "det2_16batch.prototxt"
    resize_prototxt("{}/det2.prototxt".format(model_path), target_prototxt, input_shape)
    target_models.append({"prototxt": target_prototxt,
                          "caffemodel": "{}/det2.caffemodel".format(model_path),
                          "input_shape": copy.deepcopy(input_shape)})

    input_shape = [16, 3, 48, 48]
    target_prototxt = "det3_16batch.prototxt"
    resize_prototxt("{}/det3.prototxt".format(model_path), target_prototxt, input_shape)
    target_models.append({"prototxt": target_prototxt,
                          "caffemodel": "{}/det3.caffemodel".format(model_path),
                          "input_shape": copy.deepcopy(input_shape)})
    return target_models


def create_cambricon_first_conv_int8_models(prototxt, caffemodel,input_shape):
    if core_version == "MLU100":
      if not os.path.exists(prototxt.split(".")[0]+".cambricon"):
          if not genoff(prototxt, caffemodel, prototxt.split(".")[0]):
              return False

    deploy_prototxt = "{}_deploy".format(prototxt)
    prototxt_int8 = "{}_int8.prototxt".format(prototxt.split(".")[0])

    cambricon_prefix = prototxt.split(".")[0]+"_int8"

    if os.path.exists(cambricon_prefix+".cambricon"):
        return True

    #input_shape[0]=1
    create_deploy_prototxt(prototxt,deploy_prototxt,input_shape)

    # print("convert {}->{}".format(prototxt, prototxt_int8))

    if not get_positon_normal(deploy_prototxt, caffemodel, prototxt_int8,input_shape):
        os.remove(deploy_prototxt)
        return False
    os.remove(deploy_prototxt)

    create_deploy_prototxt(prototxt_int8,prototxt_int8,input_shape)

    if not genoff(prototxt_int8, caffemodel, cambricon_prefix):
        return False
    return True

def main():
    print("Gen Cambricon Offline Models...")
    models = create_mtcnn_int8_models()
    total=len(models)
    for model in models:
        prototxt, caffemodel, input_shape = model["prototxt"], model["caffemodel"], model["input_shape"]
        # print(prototxt, caffemodel, input_shape)
        create_cambricon_first_conv_int8_models(
            prototxt, caffemodel, input_shape)


if __name__ == '__main__':
    main()
