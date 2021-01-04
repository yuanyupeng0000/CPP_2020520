import os
import sys
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import copy
import argparse
import logging

sys.path.append("../test/")

from cmpData import cmpData

ERR_MFUS_MAX = 1e-3
IMG_FILE = "file_list"

CAFFE_PY_HOME = os.getcwd()

testforwardpath = CAFFE_PY_HOME + '/../build/tools/test_forward_online '
createfakemodelpath = CAFFE_PY_HOME + '/../build/tools/create_fake_caffemodel.bin '
clasforwardpath = CAFFE_PY_HOME + '/../build/examples/clas_online_singlecore/clas_online_singlecore'

def getOptions(args=sys.argv[1:]):
  parser = argparse.ArgumentParser(description="Parses command.")
  parser.add_argument("--model", help="input prototxt.")
  parser.add_argument("--weights", help="input weights.")
  parser.add_argument("--threshold", type=float, help="error threshold")
  parser.add_argument("--outputdir", help="tmp layers data will be in outputdir.")
  parser.add_argument("--image", help="input imges.")
  parser.add_argument("--mode", help="compare by layer: layer; compare by ops: op")
  options = parser.parse_args(args)
  return options

# the 0 to i-1 layer will be write
def write_pt_id(layers, id, name, debug_info = False):
    tmplayers = layers[:id]
    print "dumping: " + name
    with open(name, "w") as f:
      if debug_info:
        f.write("debug_info: true\n")
      for layer in tmplayers:
        f.write("layer {\n")
        f.write("\n".join(["  " + line for line in str(layer).split("\n") if line != ""]))
        f.write("\n")
        f.write("}\n\n")
    f.close()

def write_pt(layers, name, debug_info = False):
  write_pt_id(layers, len(layers), name, debug_info)

def computeBaseline(pt, weight, mode, outdir):
  net = caffe_pb2.NetParameter()
  with open(pt) as f:
     s = f.read()
     txtf.Merge(s, net)
  layers = net.layer
  pt_name = "debug_pt"
  pt_name = os.path.abspath(pt_name)
  weight = os.path.abspath(weight)
  write_pt(layers, pt_name, debug_info = True)
  tmpdir = os.getcwd()
  os.chdir(outdir)
  computeResult(pt_name, weight, mode, "tmp")
  os.chdir(tmpdir)

def computeCpuResult(pt, weight, outputname):
  computeResult(pt, weight, 'CPU', outputname)

def computeResult(pt, weight, mode, outputname):
  test_out_file = outputname
  os.environ['TEST_ACCURACY'] = 'ON'
  os.environ['TEST_OUTPUT_FILE'] = test_out_file
  os.system(clasforwardpath + ' -model ' + pt + ' -weights ' + weight + ' -mmode '
            + mode + ' -mcore MLU270' + ' -images ' + IMG_FILE)

def getFakeWeight(pt, modelname):
  logging.debug(os.system(createfakemodelpath + ' ' + pt + ' ' + modelname))

def compareModelErr(pt, weight, mmode1, mmode2):
  if mmode1 == mmode2:
    print "Two modes should be different!"
    return 0
  computeResult(pt, weight, mmode1, 'mode1out')
  err_rate = compareModelErrWithBaseline(pt, weight, mmode2, 'mode1out')
  return err_rate

def compareModelErrWithBaseline(pt, weight, mmode, base_result):
  tmpout = 'mode2out'
  computeResult(pt, weight, mmode, tmpout)
  err, total, errlist = cmpData(tmpout, base_result)
  errate = err/total
  if errate > 1:
    errate = 1
  return round(errate, 5)

def compareByLayers(pt, weights):
  cpu_result = "cpu_tmp_out"
  computeCpuResult(pt, weights, cpu_result)
  errdict = dict()
  net = caffe_pb2.NetParameter()
  with open(pt) as f:
     s = f.read()
     txtf.Merge(s, net)
  layers = net.layer
  for i in range(0, len(layers)):
    layers[i].engine = 1
  for i in range(1, len(layers)):
    tmplayers = copy.deepcopy(layers)
    tmplayers[i].engine = 0
    tmppt = "testpt"
    write_pt(tmplayers, tmppt)
    err = compareModelErrWithBaseline(tmppt, weights, 'MLU', cpu_result)
    err = round(err, 5)
    errdict[tmplayers[i].name] = err
  return errdict

def compareByOptype(pt, weights):
  cpu_result = "cpu_tmp_out"
  computeCpuResult(pt, weights, cpu_result)
  errdict = dict()
  ops = []
  net = caffe_pb2.NetParameter()
  with open(pt) as f:
     s = f.read()
     txtf.Merge(s, net)
  layers = net.layer
  for i in range(1, len(layers)):
    ops.append(layers[i].type)
  ops = list(set(ops))
  for i in range(0, len(layers)):
    layers[i].engine = 1
  for op in ops:
    tmplayers = copy.deepcopy(layers)
    tmppt = "testpt"
    for i in range(1, len(tmplayers)):
      if tmplayers[i].type == op:
        tmplayers[i].engine = 0   # run on MLU
    write_pt(tmplayers, tmppt)
    err = compareModelErrWithBaseline(tmppt, weights, 'MLU', cpu_result)
    err = round(err, 5)
    errdict[op] = err
  return errdict

def compareMFUS(pt, weights):
  net = caffe_pb2.NetParameter()
  with open(pt) as f:
     s = f.read()
     txtf.Merge(s, net)
  layers = net.layer
  idx = len(layers)-1
  tmppt = "tmppt"
  passidx = 0
  failidx = idx
  errs = dict()
  errlayer = "NONE"
  while idx > passidx:
    write_pt_id(layers, idx+1, tmppt)
    err = compareModelErr(tmppt, weights, 'MFUS', 'MLU')
    err = round(err, 5)
    errs[layers[idx].name + ' index:' + str(idx)] = err
    if err < ERR_MFUS_MAX:
      passidx = idx
    else:
      failidx = idx
    idx = (passidx + failidx)/2
    if failidx - passidx == 1:
      errlayer = layers[failidx].name
      break
    if passidx == len(layers)-1:
      print "MFUS vs MLU mode err_rate: " + str(err) + " is below threshold"
      break
  return errs, errlayer

if __name__ == "__main__":
  options = getOptions()
  if options.model:
    pt = options.model
  else:
    print "input model cannot be empty!"
    exit(1)
  if options.image:
    IMG_FILE = options.image
    IMG_FILE = os.path.abspath(IMG_FILE)
  else:
    print "Error: image must not be null"
    exit(1)
  if not os.path.exists(IMG_FILE):
    print "Error: image does not exist!"
    exit(1)
  if options.weights:
    weightname = options.weights
  else:
    print "Using generated fake weight ..."
    weightname = 'tmpweight'
    getFakeWeight(pt, weightname)
  if options.threshold:
    threshold = options.threshold
    if threshold < 0:
      print "Error: threshold should be positive"
      exit(1)
    else:
      ERR_MFUS_MAX = threshold
  else:
    print "Using default threshold: " + str(ERR_MFUS_MAX)
  if options.outputdir:
    outputdir = options.outputdir
    if not os.path.exists(outputdir):
      os.mkdir(outputdir)
  else:
    print "Using Default outputdir: ./tmplayers"
    outputdir = "./tmplayers"
    if os.path.exists(outputdir):
      print("Error: outpudir exists!")
      exit(1)
    os.mkdir(outputdir)
  outputdir = os.path.abspath(outputdir)
  cpu_layers_dir = outputdir+'/cpu_layers/'
  mlu_layers_dir = outputdir+'/mlu_layers/'
  if os.path.exists(cpu_layers_dir):
    print("Error: cpudir exists!")
    exit(1)
  if os.path.exists(cpu_layers_dir):
    print("Error: mludir exists!")
    exit(1)
  os.mkdir(cpu_layers_dir)
  os.mkdir(mlu_layers_dir)
  # step1. compute CPU baseline
  computeBaseline(pt, weightname, 'CPU', cpu_layers_dir)
  print("CPU mode baseline saved to %s" % cpu_layers_dir)
  # step2. compute the err_rate between CPU and MLU
  errdict = dict()
  if options.mode and options.mode == 'layer':
      errdict = compareByLayers(pt, weightname)
  else:
    print "calculating errors by operation type ..."
    errdict = compareByOptype(pt, weightname)
  layer_err = compareModelErr(pt, weightname, 'MLU', 'CPU')
  # step3. compute MLU mode baseline
  computeBaseline(pt, weightname, 'MLU', mlu_layers_dir)
  print("MLU mode baseline saved to %s" % mlu_layers_dir)
  # step4. compute the errate between MFUS and MLU and allocate the bug layer
  errs, errlayer = compareMFUS(pt, weightname)
  # last: print the result
  print("\nThe err rate between MLU mode and CPU: %s" % layer_err)
  print "\nThe err rate compared to CPU by layers/ops:"
  for k, v in sorted(errdict.items(), key = lambda item:item[1]):
    print("%s: %s" % (k, v))
  if errlayer == 'NONE':
    print "\nMFUS is the same with MLU mode!"
  else:
    print "\nThe bug layer between MFUS and MLU is: " + errlayer
    print "\nThe err rate between MFUS and MLU:"
    for k, v in sorted(errs.items(), key = lambda item:item[1]):
      print("%s: %s" % (k, v))
