#!/usr/bin/python

import os
import sys
import copy
import re
import ConfigParser
import subprocess
import fnmatch
import shutil
import json


cf = ConfigParser.ConfigParser()
conf_name = "onetest_mlu270.conf"
g_output_path = "output"
cmd = 'rm -rf ' + str(g_output_path)
if (os.path.isdir(g_output_path)):
  os.system(cmd)

#### test config ####
if (len(sys.argv) != 2):
    print("  Usage: python onetest.py [MLU270 | MLU220]")
    sys.exit()

conf_argv = sys.argv[1]
assert conf_argv in ['MLU270', 'mlu270', 'MLU220', 'mlu220']

if (conf_argv == "MLU270" or conf_argv == "mlu270"):
    conf_name = "onetest_mlu270.conf"
if (conf_argv == "MLU220" or conf_argv == "mlu220"):
    conf_name = "onetest_mlu220.conf"

cf.read(conf_name)
offline_force = cf.getboolean("test", "offline_force")
g_caffe_core        = cf.get("version", "core")
g_caffe_parallel    = "[(-1,-1)]"

#### caffe config ####
g_caffe_enable          = cf.getboolean("caffe", "enable")
g_caffe_strategy        = cf.get("caffe", "strategy")
g_caffe_csv_path        = cf.get("caffe", "csv_path")
g_caffe_net             = cf.get("caffe", "net")
g_caffe_accuracy        = cf.get("caffe", "accuracy")
g_caffe_sparsity        = cf.get("caffe", "sparsity")
g_caffe_batch_size      = cf.get("caffe", "batch_size")
g_caffe_mode            = cf.get("caffe", "mode")
g_caffe_bangop          = cf.get("caffe", "bangop")

g_caffe_core_number           = cf.get("caffe", "core_number")
g_caffe_batchsize             = cf.get("caffe", "batchsize")
g_caffe_images_clas     = cf.get("caffe", "images_file_clas")
g_caffe_images_ssd      = cf.get("caffe", "images_file_ssd")
g_caffe_images_ssd_vgg16=cf.get("caffe","images_file_ssd_vgg16")
g_caffe_images_ssd_mobilenetv1=cf.get("caffe","images_file_ssd_mobilenetv1")
g_caffe_images_ssd_mobilenetv2=cf.get("caffe","images_file_ssd_mobilenetv2")
g_caffe_images_fastrcnn = cf.get("caffe", "images_file_fastrcnn")
g_caffe_images_mtcnn   = cf.get("caffe", "images_file_mtcnn")
g_caffe_images_yolov2   = cf.get("caffe", "images_file_yolov2")
g_caffe_images_yolov3   = cf.get("caffe", "images_file_yolov3")
g_caffe_images_rfcn   = cf.get("caffe", "images_file_rfcn")
g_caffe_images_segnet   = cf.get("caffe", "images_file_segnet")
g_caffe_label_clas      =  cf.get("caffe", "label_file_clas")
g_caffe_label_ssd       =  cf.get("caffe", "label_file_ssd")
g_caffe_label_ssd_vgg16=cf.get("caffe","label_file_ssd_vgg16")
g_caffe_label_ssd_mobilenetv1=cf.get("caffe","label_file_ssd_mobilenetv1")
g_caffe_label_ssd_mobilenetv2=cf.get("caffe","label_file_ssd_mobilenetv2")
g_caffe_label_yolov2    =  cf.get("caffe", "label_file_yolov2")
g_caffe_label_yolov3    =  cf.get("caffe", "label_file_yolov3")
g_caffe_label_segnet    =  cf.get("caffe", "label_file_segnet")
g_caffe_model_list_mtcnn    =  cf.get("caffe", "model_file_mtcnn")

g_caffe_bin_dir = "../../build/examples/"
g_caffe_clas_online_bin         =   g_caffe_bin_dir + "clas_online_multicore/clas_online_multicore.bin"
g_caffe_clas_offline_bin        =   g_caffe_bin_dir + "clas_offline_multicore/clas_offline_multicore.bin"
g_caffe_clas_debug_bin          =   g_caffe_bin_dir + "clas_online_singlecore/clas_online_singlecore.bin"
g_caffe_ssd_online_bin          =   g_caffe_bin_dir + "ssd/ssd_online_singlecore.bin"
g_caffe_ssd_offline_bin         =   g_caffe_bin_dir + "ssd/ssd_offline_multicore.bin"
g_caffe_fasterrcnn_online_bin   =   g_caffe_bin_dir + "faster-rcnn/faster-rcnn_online_multicore.bin"
g_caffe_fasterrcnn_offline_bin  =   g_caffe_bin_dir + "faster-rcnn/faster-rcnn_offline_multicore.bin"
g_caffe_mtcnn_offline_bin       =   g_caffe_bin_dir + "mtcnn/mtcnn-offline_multicore.bin"
g_caffe_yolov2_online_bin       =   g_caffe_bin_dir + "yolo_v2/yolov2_online_singlecore.bin"
g_caffe_yolov2_offline_bin      =   g_caffe_bin_dir + "yolo_v2/yolov2_offline_multicore.bin"
g_caffe_yolov3_online_bin       =   g_caffe_bin_dir + "yolo_v3/yolov3_online_singlecore.bin"
g_caffe_yolov3_offline_bin      =   g_caffe_bin_dir + "yolo_v3/yolov3_offline_multicore.bin"
g_caffe_rfcn_online_bin       =   g_caffe_bin_dir + "rfcn/rfcn_online_singlecore.bin"
g_caffe_rfcn_offline_bin      =   g_caffe_bin_dir + "rfcn/rfcn_offline_multicore.bin"
g_caffe_segnet_offline_bin      =   g_caffe_bin_dir + "segnet/segnet_offline_multicore.bin"
g_caffe_mean_file = ""
g_caffe_voc_path            = "../../../../../datasets/VOC2012/Annotations"
g_caffe_coco_path          = "../../../../../datasets/COCO"
g_caffe_fddb_path            = "../../../../../datasets/FDDB"



g_caffe_meanap_bin          = "../meanAP_VOC.py"
g_caffe_meanap_coco_bin     = "../meanAP_COCO.py"
g_caffe_meanap_fddb_bin     = "../meanAP_FDDB.py"
g_caffe_meanIOU_bin         = "../compute_mIOU.py"
g_caffe_bdir_yolov3         = "../../examples/yolo_v3/bbox_anchor"
g_caffe_yolov2_label_map    = "../../examples/yolo_v2/label_map.txt"

def str_to_list(s):
    l = s.strip("[]").split(',')
    return map(lambda x: x.strip(), l)

def str_to_dicts(key, s):
    ns = str_to_list(s)
    r = []
    for i in ns:
        d = {}
        d[key] = i
        r.append(d)
    return r

def str_to_dicts_parallel(key, s):
    s = s.replace(' ', '')
    pairs = s.split("),")
    pairs = map(lambda x : x.strip('[()]'), pairs)
    r = []
    for i in pairs:
        d = {}
        d[key] = i
        r.append(d)
    return r

def extend_list(a1, a2):
    r = []
    for d1 in a1:
        for d2 in a2:
            d = copy.deepcopy(d1)
            d.update(d2)
            r.append(d)
    return r

def parse_field(line):
    line = line.strip()
    fields = line.split(',')
    d = {}
    for i in range(len(fields)):
        d[fields[i]] = i
    return d


################################################
##################   caffe    ##################
################################################
def caffe_test():
    if g_caffe_strategy == "comb":
        caffe_test_comb()
    elif  g_caffe_strategy == "serial":
        caffe_test_serial()
    else:
        assert False

def caffe_test_comb():
    combs = caffe_gen_comb()
    repo = caffe_gen_repo()
    l = len(combs)
    for i in range(len(combs)):
        case = caffe_find_case(combs[i], repo)
        if case:
            top1, top5, fps, hwfps = caffe_test_one(combs[i], case)
            bmtop1, bmtop5, bmfps = caffe_get_benchmark(case)
        else:
            assert False, "caffe can't find mode in csv: %s" % (str(combs[i]))
        caffe_screen(combs[i], i+1, l, top1, top5, fps, bmtop1, bmtop5, bmfps, hwfps)

def caffe_get_benchmark(case):
    bmtop1 = case["top1"]
    bmtop5 = case["top5"]
    bmfps = case["fps"]
    return bmtop1, bmtop5, bmfps

clas_print = False
detect_print = False
simple_clas_print = False
simple_detect_print = False
simple_segnet_print = False
def caffe_screen(c, i, l, top1, top5, fps, bmtop1, bmtop5, bmfps, hwfps):
    global clas_print, detect_print, simple_clas_print, simple_detect_print, simple_segnet_print
    detect_list = ["ssd", "ssd_vgg16", "ssd_mobilenetv1", "ssd_mobilenetv2", "yolov2", "yolov3", "faster-rcnn", "rfcn", "mtcnn"]
    segnet_list = ["segnet"]
    if (c["net"] in segnet_list) and (not simple_segnet_print):
      detect_print = False
      clas_print = False
      simple_detect_print = False
      simple_clas_print = False
      simple_segnet_print = True
      print ""
      print "---------------------------------------------------------------------------------------------------------------------------------------"
      print "                    |                 net|    accu|   spar|   dev|     mode|batchsize|core_number|               mIOU|                  fps|"
      print "---------------------------------------------------------------------------------------------------------------------------------------"
    elif (c["net"] not in detect_list) and (not simple_clas_print):
      detect_print = False
      clas_print = False
      simple_detect_print = False
      simple_clas_print = True
      simple_segnet_print = False
      print ""
      print "---------------------------------------------------------------------------------------------------------------------------------------"
      print "                    |                 net|    accu|   spar|   dev|     mode|batchsize|core_number|        top1|        top5|           fps|"
      print "---------------------------------------------------------------------------------------------------------------------------------------"
    elif (c["net"] in detect_list) and (not simple_detect_print):
      detect_print = False
      clas_print = False
      simple_detect_print = True
      simple_clas_print = False
      simple_segnet_print = False
      print ""
      print "---------------------------------------------------------------------------------------------------------------------------------------"
      print "                    |                 net|    accu|   spar|   dev|     mode|batchsize|core_number|               mAP|                  fps|"
      print "---------------------------------------------------------------------------------------------------------------------------------------"
    is_clas = 0
    progress = "%s/%s" % (i, l)
    top1_info = "%s" % (top1)
    top5_info = "%s" % (top5)
    fps_info = "%s" % (hwfps)
    hwfps_info = "%s" % (hwfps)
    if c["mode"] == "cpu":
	dev = "cpu"
    else:
	dev = "mlu"
    if c["net"] in segnet_list:
	print "[ CAFFE %+8s] : |%+20s|%+8s|%+7s|%+6s|%+9s|%+9s|%+11s|%+18s|%+21s|" \
	    % (progress, c["net"], c["accuracy"], c["sparsity"], dev, c["mode"], c["batchsize"], c["core_number"], \
	       top1_info, fps_info)
    elif c["net"] not in detect_list:
	print "[ CAFFE %+8s] : |%+20s|%+8s|%+7s|%+6s|%+9s|%+9s|%+11s|%+12s|%+12s|%+14s|" \
	    % (progress, c["net"], c["accuracy"], c["sparsity"], dev, c["mode"], c["batchsize"], c["core_number"], \
	       top1_info, top5_info, fps_info)
        is_clas = 1
    else:
	print "[ CAFFE %+8s] : |%+20s|%+8s|%+7s|%+6s|%+9s|%+9s|%+11s|%+18s|%+21s|" \
	    % (progress, c["net"], c["accuracy"], c["sparsity"], dev, c["mode"], c["batchsize"], c["core_number"], \
	       top1_info, fps_info)

    #caffe_simple_compile_output_json(c['net'], c['accuracy'], c['sparsity'], c['mode'], c['simple_compile'], c['batchsize'], c['core_number'], \
    caffe_simple_compile_output_json(c['net'], c['accuracy'], c['sparsity'], c['mode'], c['batchsize'], c['core_number'], \
	top1, top5, fps, hwfps_info, is_clas)



#def caffe_simple_compile_output_json(net, accuracy, sparsity, model, simple_compile, batchsize, core_number, top1, top5, fps, hwfps):
def caffe_simple_compile_output_json(net, accuracy, sparsity, model, batchsize, core_number, top1, top5, fps, hwfps, is_clas):
    res = {}
    summary = {}
    res['Summary'] = summary
    accu = {}
    summary['accuracy'] = accu
    #res['net'] = net
    #res['dataType'] = accuracy
    #res['sparsity'] = sparsity
    #res['model'] = model
    #res['batchsize'] = batchsize
    #res['core_number'] = core_number
    if (is_clas == 0):
        accu['top1']=str(-1)
        accu['top5']=str(-1)
        accu['meanAp']=str(top1)
    else:
        accu['top1'] = str(top1)
        accu['top5'] = str(top5)
        accu['meanAp'] = str(-1)

    perf = {}
    summary['performance']=perf
    perf['latency'] = str(round(float((1000000)/float(hwfps))*float(batchsize),2))
    perf['throughput'] = str(hwfps)
    js_file= "./output/"+ net +".json"
    with open(js_file, 'w') as f:
        json.dump(res, f, indent = 4)

def caffe_find_case(c, repo):
    for i in range(len(repo)):
        if c["net"] == repo[i]["net"] \
            and c["accuracy"] == repo[i]["accuracy"] \
            and c["sparsity"] == repo[i]["sparsity"] \
            and c["batch_size"] == repo[i]["batch_size"]:
                return repo[i]
    # assert False, "no case %s, %s, %s" % (c["net"], c["accuracy"], c["sparsity"])
    # assert False, "no case"


def caffe_gen_comb():
    nets = str_to_dicts("net", g_caffe_net)
    accuracys = str_to_dicts("accuracy", g_caffe_accuracy)
    sparsitys = str_to_dicts("sparsity", g_caffe_sparsity)
    modes = str_to_dicts("mode", g_caffe_mode)
    bangop = str_to_dicts("bangop", g_caffe_bangop)
    batch_sizes = str_to_dicts("batch_size", g_caffe_batch_size)
    batchsizes = str_to_dicts("batchsize", g_caffe_batchsize)
    core_numbers = str_to_dicts("core_number", g_caffe_core_number)
    parallels = str_to_dicts_parallel("parallel", g_caffe_parallel)
    r1 = extend_list(nets, accuracys)
    r2 = extend_list(r1, sparsitys)
    r3 = extend_list(r2, modes)
    r4 = extend_list(r3, bangop)
    r5 = extend_list(r4, batch_sizes)
    r6 = extend_list(r5, batchsizes)
    r7 = extend_list(r6, core_numbers)
    r8 = extend_list(r7, parallels)
    return r8

def caffe_gen_repo():
    r = []
    with open(g_caffe_csv_path) as f:
        content = f.readlines()
        field_map = parse_field(content[0])
        for line in content[1:]:
            d = caffe_gen_repo_one(line, field_map)
            r.append(d)
    return r


def caffe_gen_repo_one(line, field_map):
    d = {}
    datas = line.strip().split(',')
    # d["name"] = datas[field_map["name"]]
    d["net"] = datas[field_map["net"]]
    d["accuracy"] = datas[field_map["accuracy"]]
    d["sparsity"] = datas[field_map["sparsity"]]
    d["batch_size"] = datas[field_map["batch_size"]]
    d["company"] = datas[field_map["company"]]
    d["version"] = datas[field_map["version"]]
    d["top1"] = datas[field_map["top1"]]
    d["top5"] = datas[field_map["top5"]]
    d["fps"] = datas[field_map["fps"]]
    d["proto_file"] = datas[field_map["proto_file"]]
    d["weights_file"] = datas[field_map["weights_file"]]
    return d

def caffe_test_one(comb, case):
    global g_caffe_images_clas
    global g_caffe_label_clas
    g_caffe_images_clas = cf.get("caffe", "images_file_clas")
    g_caffe_label_clas =  cf.get("caffe", "label_file_clas")
    if comb["net"] in ["ssd","ssd_vgg16","ssd_mobilenetv1","ssd_mobilenetv2"] :
        return caffe_test_ssd(comb, case)
    elif comb["net"] == "yolov2":
        return caffe_test_yolov2(comb, case)
    elif comb["net"] == "yolov3":
        return caffe_test_yolov3(comb, case)
    elif comb["net"] == "faster-rcnn":
        return caffe_test_fasterrcnn(comb, case)
    elif comb["net"] == "mtcnn":
        return caffe_test_mtcnn(comb, case)
    elif comb["net"] == "rfcn":
        return caffe_test_rfcn(comb, case)
    elif comb["net"] == "segnet":
        return caffe_test_segnet(comb, case)
    elif comb["net"] == "inception-v3":
        g_caffe_images_clas = cf.get("caffe", "images_file_clas_2015")
        g_caffe_label_clas =  cf.get("caffe", "label_file_clas_2015")
        return caffe_test_clas(comb, case)
    else:
        return caffe_test_clas(comb, case)

def caffe_test_clas(comb, case):
    if comb["mode"] in ["cpu", "normal"]:
        return caffe_test_clas_debug(comb, case)
    elif comb["mode"] == "fusion":
        return caffe_test_clas_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_clas_offline(comb, case)
    else:
        assert False,"caffe test mode is wrong"

def caffe_gen_case_path(file):
    i = g_caffe_csv_path.rfind('/')
    return g_caffe_csv_path[:i+1] + file

def shell_call(cmd, logfile, append=False):
    m = 'a' if append else 'w'
    with open(logfile, m) as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=log, shell=True)
        p.wait()

def caffe_test_clas_debug(comb, case):
    cmd = g_caffe_clas_debug_bin
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_clas
    label = g_caffe_label_clas
    core = g_caffe_core
    mean = "NULL"
    if comb["mode"] == "cpu":
        mode = "CPU"
    elif comb["mode"] == "normal":
        mode = "MLU"
    elif comb["mode"] == "fusion":
        mode = "MFUS"
    else:
        assert False, "can't run debug mode %s" % (comb["comb"], )

    shell = "%s -model %s -weights %s %s -labels %s -images %s -mcore %s -mmode %s" %(cmd, model, weights, mean, label, image, core, mode)

    # os.system(shell)
    logfile = g_output_path+"/log"
    shell_call(shell, logfile)
    return caffe_parse_result(logfile)


def caffe_test_clas_online(comb, case):
    cmd = g_caffe_clas_online_bin
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    core = g_caffe_core
    image = g_caffe_images_clas
    label = g_caffe_label_clas
    scale = "0.017" if comb["net"] == "mobilenet" else "1"
    int8_param = " -point INT8 " if comb["accuracy"] == "int8" else ""

    shell = "%s -model %s -weights %s -images %s -labels %s \
                -batchsize %s -core_number %s -mcore %s -simple_compile 1" \
            % (cmd, model, weights, image, label, batchsize, core_number, core)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile)
    return caffe_parse_result(logfile)


def caffe_test_clas_offline(comb, case):
    cmd = g_caffe_clas_offline_bin
    net = comb["net"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_clas
    label = g_caffe_label_clas
    core = g_caffe_core
    #scale = "0.017" if comb["net"] == "mobilenet" else "1"
    #int8_param = " -point INT8 " if comb["accuracy"] == "int8" else ""

    logfile = g_output_path+"/log"
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    gen_shell = "../../build/tools/caffe genoff   \
		 -model %s -weights %s -mname %s  \
		 -mcore %s \
		 -batchsize %s -core_number %s -simple_compile 1"   \
		%(model, weights, net, core, batchsize, core_number)
    #print gen_shell
    shell_call(gen_shell, logfile)
    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))
    shell = "%s -offlinemodel %s -images %s -labels %s" \
	    % (cmd, cambricon, image, label)
    shell_call(shell, logfile)
    return caffe_parse_result(logfile)


def caffe_test_ssd(comb, case):
    if comb["mode"] in ["cpu", "normal", "fusion"]:
        return caffe_test_ssd_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_ssd_offline(comb, case)
    else:
        assert False, "caffe test ssd: unsupport mode %s" % (comb["mode"],)

def caffe_test_ssd_online(comb, case):
    assert False, "caffe test ssd: unsupport online simple compile"


def caffe_test_ssd_offline(comb, case):
    cmd = g_caffe_ssd_offline_bin
    meanAp = g_caffe_meanap_bin
    net = comb["net"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_ssd
    label = g_caffe_label_ssd
    core = g_caffe_core
    bangop = comb["bangop"]
    voc_path = g_caffe_voc_path

    logfile = g_output_path+"/log"
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    gen_shell = "../../build/tools/caffe genoff   \
		 -model %s -weights %s -mname %s  \
		 -mcore %s\
		 -batchsize %s -core_number %s -simple_compile 1  \
		 -Bangop %s" \
		%(model, weights, net, core, batchsize, core_number, bangop)
    shell_call(gen_shell, logfile)

    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))

    shell = "%s -offlinemodel %s -images %s -labelmapfile %s -confidencethreshold 0.6 \
	        -dump 1 -Bangop %s" \
	    % (cmd, cambricon, image, label, bangop)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile)
    caffe_move_txt()
    caffe_move_jpg()

    shell = "python %s %s %s %s" \
            % (meanAp, image, g_output_path, voc_path)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    top1, top5, fps, hwfps = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps

def caffe_move_txt():
    os.system("rm -rf %s/*.txt" % (g_output_path, ))
    [ shutil.move(name, g_output_path) for name in os.listdir(os.curdir) if fnmatch.fnmatch(name,'*.txt') ]

def caffe_move_jpg():
    os.system("rm -rf %s/*.jpg" % (g_output_path, ))
    [ shutil.move(name, g_output_path) for name in os.listdir(os.curdir) if fnmatch.fnmatch(name,'*.jpg') ]

def caffe_move_png():
    os.system("rm -rf %s/*.png" % (g_output_path, ))
    [ shutil.move(name, g_output_path) for name in os.listdir(os.curdir) if fnmatch.fnmatch(name,'*.png') ]

def caffe_test_fasterrcnn(comb, case):
    if comb["mode"] in ["normal", "fusion"]:
        return caffe_test_fasterrcnn_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_fasterrcnn_offline(comb, case)
    else:
        assert False, "caffe unsupport mode %s" % (comb["mode"],)

def caffe_test_fasterrcnn_online(comb, case):
    assert False, "caffe test faster-rcnn: unsupport online simple compile"



def caffe_test_fasterrcnn_offline(comb, case):
    cmd = g_caffe_fasterrcnn_offline_bin
    meanAp = g_caffe_meanap_bin
    net = comb["net"]
    bangop = comb["bangop"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    core = g_caffe_core
    image = g_caffe_images_fastrcnn
    #int8_mode = "1" if comb["accuracy"] == "int8" else "0"
    voc_path = g_caffe_voc_path


    logfile = g_output_path+"/log"
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    gen_shell = "../../build/tools/caffe genoff   \
		 -model %s -weights %s -mname %s  \
		 -mcore %s -Bangop %s         \
		 -batchsize %s\
		 -core_number %s -simple_compile 1" \
		%(model, weights, net, core, bangop, batchsize, core_number)
    shell_call(gen_shell, logfile)
    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))

    shell = "%s -offlinemodel %s -images %s \
	       -dump 1 -Bangop %s" \
	    % (cmd, cambricon, image, bangop)
    shell_call(shell, logfile)
    caffe_move_txt()
    caffe_move_jpg()

    shell = "python %s %s %s %s" \
            % (meanAp, image, g_output_path, voc_path)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    top1, top5, fps, hwfps, = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps

def caffe_test_mtcnn(comb, case):
    if comb["mode"] == "offline":
        return caffe_test_mtcnn_offline(comb, case)
    else:
        assert False, "caffe test mtcnn: unsupport mode %s" % (comb["mode"],)


def caffe_test_mtcnn_offline(comb, case):
    comb["bangop"] = "*"
    cmd = g_caffe_mtcnn_offline_bin
    meanAp = g_caffe_meanap_fddb_bin
    model_list = g_caffe_model_list_mtcnn
    net = comb["net"]
    model_path = caffe_gen_case_path(case["proto_file"])
    image_list = g_caffe_images_mtcnn
    int8_mode = "1" if comb["accuracy"] == "int8" else "0"
    thread = comb["parallel"]
    comb["parallel"] = "{},*".format(thread)
    fddb_path = g_caffe_fddb_path

    logfile = g_output_path+"/log"
    caffe_dir = os.path.join(os.getcwd(), '../../')
    gen_shell = "../../examples/mtcnn/gen_models.py \
                    --file-list %s --model-path %s \
                    --caffe-dir %s --core-version %s" \
                %(image_list, model_path, caffe_dir, g_caffe_core)
    shell_call(gen_shell, logfile)

    model_list += comb["accuracy"]
    shell = "%s -images %s -models %s \
                -threads 16 -int8 %s" \
            % (cmd, image_list, model_list, int8_mode)
    shell_call(shell, logfile)

    result_txt = os.path.join(os.getcwd(), "mtcnn.txt")
    output_jpg = os.path.join(os.getcwd(), "mtcnn_roc.png")
    groundtruth_file = os.path.join(fddb_path, "ellipseList.txt")
    shell = "python %s %s %s %s" \
            % (meanAp, result_txt, groundtruth_file, output_jpg)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    # clean the result files
    caffe_move_jpg()
    caffe_move_txt()
    os.system("mv *.cambricon* %s" % (g_output_path))
    os.system("mv *.prototxt %s" % (g_output_path))
    os.system("mv convert_quantized.ini %s" % (g_output_path))

    top1, top5, fps, hwfps, = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps

def caffe_test_yolov2(comb, case):
    if comb["mode"] in ["normal", "fusion"]:
        return caffe_test_yolov2_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_yolov2_offline(comb, case)
    else:
        assert False, "caffe unsupport mode %s" % (comb["mode"],)

def caffe_test_yolov2_online(comb, case):
    assert False, "caffe test yolov2: unsupport online simple compile"


def caffe_test_yolov2_offline(comb, case):
    cmd = g_caffe_yolov2_offline_bin
    meanAp = g_caffe_meanap_bin
    net = comb["net"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_yolov2
    voc_path = g_caffe_voc_path
    label_map = g_caffe_label_yolov2
    core = g_caffe_core
    bangop = comb["bangop"]

    logfile = g_output_path+"/log"
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    gen_shell = "../../build/tools/caffe genoff   \
		 -model %s -weights %s -mname %s  \
		 -mcore %s -Bangop %s \
		 -batchsize %s -core_number %s -simple_compile 1" \
		%(model, weights, net, core, bangop, batchsize, core_number)
    shell_call(gen_shell, logfile)
    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))

    shell = "%s -offlinemodel %s -images %s -labels %s \
	        -Bangop %s -dump 1 " \
	    % (cmd, cambricon, image, label_map, bangop)
    shell_call(shell, logfile)
    caffe_move_txt()
    caffe_move_jpg()

    shell = "python %s %s %s %s" \
            % (g_caffe_meanap_bin, image, g_output_path, voc_path)
    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    top1, top5, fps, hwfps = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps


def caffe_test_yolov3(comb, case):
    if comb["mode"] in ["normal", "fusion"]:
        return caffe_test_yolov3_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_yolov3_offline(comb, case)
    else:
        assert False, "caffe unsupport mode %s" % (comb["mode"],)

def caffe_test_yolov3_online(comb, case):
    assert False, "caffe test yolov3: unsupport online simple compile"


def caffe_test_yolov3_offline(comb, case):
    cmd = g_caffe_yolov3_offline_bin
    meanAp = g_caffe_meanap_coco_bin
    net = comb["net"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_yolov3
    #int8_mode = "1" if comb["accuracy"] == "int8" else "0"
    coco_path = g_caffe_coco_path
    label_map = g_caffe_label_yolov3
    bdir = g_caffe_bdir_yolov3
    core = g_caffe_core

    logfile = g_output_path+"/log"
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    gen_shell = "../../build/tools/caffe genoff   \
		 -model %s -weights %s -mname %s  \
		 -mcore %s\
		 -batchsize %s -core_number %s -simple_compile 1" \
		%(model, weights, net, core, batchsize, core_number)
    shell_call(gen_shell, logfile)
    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))

    shell = "%s -offlinemodel %s -images %s -labels %s \
	        -dump 1 -preprocess_option 4" \
	    % (cmd, cambricon, image, label_map)
    shell_call(shell, logfile)
    caffe_move_txt()
    caffe_move_jpg()

    shell = "python %s --file_list %s --result_dir %s --ann_dir %s" \
            % (meanAp, image, g_output_path, coco_path)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    top1, top5, fps, hwfps = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps

def caffe_test_segnet(comb, case):
    if comb["mode"] in ["normal", "fusion"]:
        return caffe_test_segnet_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_segnet_offline(comb, case)
    else:
        assert False, "caffe unsupport mode %s" % (comb["mode"],)

def caffe_test_segnet_online(comb, case):
    assert False, "caffe test segnet: unsupport online simple compile"


def caffe_test_segnet_offline(comb, case):
    cmd = g_caffe_segnet_offline_bin
    meanIOU = g_caffe_meanIOU_bin
    net = comb["net"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_segnet
    voc_segment_class_path = g_caffe_voc_path + '/' + g_caffe_label_segnet
    core = g_caffe_core

    logfile = g_output_path+"/log"
    batchsize = comb['batchsize']
    core_number = comb['core_number']
    gen_shell = "../../build/tools/caffe genoff   \
		 -model %s -weights %s -mname %s  \
		 -mcore %s\
		 -batchsize %s -core_number %s -simple_compile 1" \
		%(model, weights, net, core, batchsize, core_number)
    shell_call(gen_shell, logfile)
    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))

    shell = "%s -offlinemodel %s -images %s -simple_compile 1" \
	    % (cmd, cambricon, image)
    shell_call(shell, logfile)
    caffe_move_txt()
    caffe_move_png()

    shell = "python %s --pred_dir %s --gt_dir %s --file_list %s" \
            % (meanIOU, g_output_path, voc_segment_class_path, image)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    top1, top5, fps, hwfps = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps

def caffe_test_rfcn(comb, case):
    if comb["mode"] in ["normal", "fusion"]:
        return caffe_test_rfcn_online(comb, case)
    elif comb["mode"] == "offline":
        return caffe_test_rfcn_offline(comb, case)
    else:
        assert False, "caffe unsupport mode %s" % (comb["mode"],)

def caffe_test_rfcn_online(comb, case):
    assert False, "caffe test faster-rcnn: unsupport online simple compile"

def caffe_test_rfcn_offline(comb, case):
    cmd = g_caffe_rfcn_offline_bin
    meanAp = g_caffe_meanap_bin
    net = comb["net"]
    model = caffe_gen_case_path(case["proto_file"])
    weights = caffe_gen_case_path(case["weights_file"])
    image = g_caffe_images_rfcn
    core = g_caffe_core
    bangop = comb["bangop"]
    voc_path = g_caffe_voc_path

    logfile = g_output_path+"/log"
    gen_shell = "../../build/tools/caffe genoff \
                        -model %s -weights %s \
                        -mname %s -mcore %s \
                        -Bangop %s" \
                    %(model, weights, net, core, bangop)
    shell_call(gen_shell, logfile)
    cambricon = "%s/%s.cambricon" % (g_output_path, net)
    os.system("mv %s.cambricon* %s" % (net, g_output_path))

    shell = "%s -offlinemodel %s -images %s -Bangop %s " \
            % (cmd, cambricon, image, bangop)
    shell_call(shell, logfile)
    caffe_move_txt()
    caffe_move_jpg()

    shell = "python %s %s %s %s" \
            % (meanAp, image, g_output_path, voc_path)

    logfile = g_output_path+"/log"
    shell_call(shell, logfile, append=True)

    top1, top5, fps, hwfps = caffe_parse_result(logfile)
    return top1, "None", fps, hwfps

def caffe_test_serial():
    pass


def caffe_parse_result(logfile):
    top1 = 0.0
    top5 = 0.0
    fps = 0.0
    hwfps = 0.0
    total_time = 0
    image_nums = 0

    with open(logfile) as log:
        lines = log.readlines()
        for i, line in enumerate(lines):
            if "accuracy1:" in line:
                top1 = re.search(r'accuracy1: (\S+) ',line).group(1)
            if "accuracy5:" in line:
                top5 = re.search(r'accuracy5: (\S+) ',line).group(1)
                image_nums = int(re.search(r'\(\d+\/(\d+)\)',line).group(1))
            if "top1:" in line:
                top1 = re.search(r'top1: (\S+) ',line).group(1)
            if "top5:" in line:
                top5 = re.search(r'top5: (\S+) ',line).group(1)
                image_nums = int(re.search(r'\(\d+\/(\d+)\)',line).group(1))
            if "^Total execution time:" in line:
                total_time = re.search(r'time: (\S+) us',line).group(1)
            if "Classify() execution time:" in line:
                total_time = re.search(r'time: (\S+) us',line).group(1)
            # for meanAp
            if "mAP:" in line:
                top1 = float(re.search(r'mAP: (\S+)',line).group(1))
            # for meanAP_COCO
            if "Average Precision" in line and "IoU=0.50     " in line:
                top1 = float(re.search(r'.*IoU=0\.50.* ] = (\S+)',line).group(1))
            # for fasterrcnn
            if "End2end throughput:" in line:
                fps = float(re.search(r'throughput: (\S+) fps',line).group(1))
            # for ssd
            if "End2end througput fps:" in line:
                fps = float(re.search(r'End2end througput fps: (\S+)',line).group(1))
            if "End2end throughput fps:" in line:
                fps = float(re.search(r'End2end throughput fps: (\S+)',line).group(1))
            if "Hardware fps:" in line:
                hwfps = float(re.search(r'Hardware fps: (\S+)', line).group(1))
            if "throughput:" in line:
                hwfps = float(re.search(r'throughput: (\S+)', line).group(1))
            if "Throughput(fps):" in line:
                hwfps = float(re.search(r'Throughput\(fps\): (\S+)', line).group(1))
            if "Throughput:" in line:
                hwfps = float(re.search(r'Throughput: (\S+)', line).group(1))
            if "mIOU:" in line:
                top1 = float(re.search(r'mIOU:(\S+)',line).group(1))

    if fps == 0 and total_time > 0:
        fps = int(image_nums) * 1000000 / float(total_time)
    return round(float(top1), 2), round(float(top5), 2), int(fps), int(hwfps)



####################################################
def init():
    os.system("mkdir %s -p" % (g_output_path, ))

def main():
    init()
    if g_caffe_enable:
        caffe_test()



if __name__ == "__main__":
    main()
