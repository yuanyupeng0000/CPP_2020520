#-*- coding: utf-8 -*-
"""
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import sys
import os
global util_layers
global del_layers
global layers

def del_layer_by_top(top):
    for index in range(len(layers)):
        if layers[index].top[0] == top:
            del layers[index]
            break

def find_layer_by_top(top):
    layer = [layer for layer in layers if layer.top[0] == top][0]
    return layer

def find_previous_layer_type(previous_layer_top):
    res = [layer for layer in layers if layer.top[0] == previous_layer_top]
    return res[0].type


def find_layers_to_del(current_layer_top):
    current_layer = find_layer_by_top(current_layer_top)
    del_layers.append(current_layer_top)
    if current_layer.type == 'PriorBox':
        pass
    elif current_layer.type == 'Permute':
        # collect input layers of Permute Layer
        util_layers.append(current_layer.bottom[0])
    else:
        # recursively delete layers
        for name in current_layer.bottom:
            if name != 'data':
                find_layers_to_del(name)

if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.pt') and not sys.argv[1].endswith('.prototxt'):
        print "Usage: python optimize_detection_pt.py origin.prototxt(or .pt)"
        exit(1)

    out_filename = '_optimized.'.join(os.path.basename(sys.argv[1]).split('.'))
    net = caffe_pb2.NetParameter()

    fn = sys.argv[1]
    # interpret origin pt
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)

    layers = net.layer
    assert layers[-1].type in ['DetectionPoseOutput','DetectionOutput']
    out_type = 'SsdDetectionPose' if layers[-1].type == 'DetectionPoseOutput' else 'SsdDetection'
    # find all priorbox param
    priorbox_params = [layer.prior_box_param for layer in layers if layer.type == 'PriorBox']
    input_layers = [layer.bottom for layer in layers if layer.type == 'PriorBox']
    # find all input layers of Priorbox
    input_layers = [name for l in input_layers for name in l if name != 'data']
    util_layers = []
    del_layers = []
    # delete all layers between bottom and Permute
    for bottom in layers[-1].bottom:
        if bottom != 'data':
            find_layers_to_del(bottom)
    for layer in del_layers:
        del_layer_by_top(layer)
    layers[-1].type = out_type
    layers[-1].ClearField('bottom')
    # add bottoms
    layers[-1].bottom.extend(util_layers+input_layers+['data'])
    # add priorbox
    layers[-1].priorbox_params.extend(priorbox_params)
    with open(out_filename,'w') as f:
        if len(net.input) != 0:
            for i in range(len(net.input)):
                f.write('input:"{}"\n'.format(net.input[i]))
                f.write('input_shape:{\n')
                f.write('\n'.join(['  ' + line for line in str(net.input_shape[i]).split('\n') if line != '']))
                f.write('\n}\n\n')
        for layer in layers:
            # format the output pt
            f.write('layer {\n')
            f.write('\n'.join(['  '+line for line in str(layer).split('\n') if line != '']))
            f.write('\n')
            f.write('}\n\n')
        print "output file:" + out_filename
