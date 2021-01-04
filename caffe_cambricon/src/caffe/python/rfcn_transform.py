#-*- coding: utf-8 -*-
"""
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#! /usr/bin/env python
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import sys
import os

def find_layer_by_top(top):
    layer = [layer for layer in layers if layer.top[0] == top][0]
    return layer

if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.pt') and not sys.argv[1].endswith('.prototxt'):
        print "Usage: python rfcn_transform.py rfcn.prototxt(or .pt)"
        exit(1)
    out_filename = '_mlu.'.join(os.path.basename(sys.argv[1]).split('.'))
    net = caffe_pb2.NetParameter()
    fn = sys.argv[1]
    # interpret origin pt
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net)
    layers = net.layer
    psroi_layers = [layer for layer in layers if layer.type == 'PSROIPooling']
    assert len(psroi_layers) == 2
    for psroi_layer in psroi_layers:
        group_size = psroi_layer.psroi_pooling_param.group_size
        output_dim = psroi_layer.psroi_pooling_param.output_dim
        while output_dim % 16 != 0:
            output_dim += 1
        psroi_layer.psroi_pooling_param.output_dim = output_dim
        bottom = psroi_layer.bottom
        for b in bottom:
            b_layer = find_layer_by_top(b)
            if b_layer.type == 'Convolution':
                b_layer.convolution_param.num_output = output_dim * (group_size ** 2)
    for layer in layers[-2:]:
        dim = layer.reshape_param.shape.dim
        new_dim = []
        for d in dim:
            if d != -1:
                while d % 16 != 0:
                    d += 1
            new_dim.append(d)
        layer.reshape_param.shape.ClearField('dim')
        layer.reshape_param.shape.dim.extend(new_dim)
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
