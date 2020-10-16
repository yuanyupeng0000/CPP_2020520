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

import sys
import numpy as np
sys.path.insert(0, "../caffe/python")
import os
import caffe

caffe.set_mode_cpu()

# caffe.set_mode_mlu()
# usage: python convert_mlu_model.py rfcn.prototxt rfcn_mlu.prototxt resnet101_rfcn_final.caffemodel
net = caffe.Net( sys.argv[1], sys.argv[3] , caffe.TEST)
net_mlu = caffe.Net( sys.argv[2], caffe.TEST)

for name in net.params.keys():

    if name == "rpn_bbox_pred":
        # 4 * 9 -> 9 * 4
        W = net.params[name][0].data
        B = net.params[name][1].data
        net_mlu.params[name][0].data[...] = W.reshape(9, 4, 512).transpose(1, 0, 2).reshape(36, 512, 1, 1)
        net_mlu.params[name][1].data[...] = B.reshape(9, 4).transpose(1, 0).reshape(36)
    # elif name == "rpn_cls_score":
    #     # 2 * 9 -> 9 * 2
    #     W = net.params[name][0].data
    #     B = net.params[name][1].data
    #     net_mlu.params[name][0].data[...] = W.reshape(9, 2, 512).transpose(1, 0, 2).reshape(18, 512, 1, 1)
    #     net_mlu.params[name][1].data[...] = B.reshape(9, 2).transpose(1, 0).reshape(18)
    elif name == "rfcn_cls":
        # 21 * 7**2 -> 32 * 7**2
        W = net.params[name][0].data
        B = net.params[name][1].data
        # convert [OUTPUT_DIM, GROUP_SIZE^2] -> [GROUP_SIZE^2, OUTPUT_DIM]
        W = W.reshape(21, 7**2, *W.shape[1:]).transpose((1, 0, 2, 3, 4)).reshape(*W.shape)
        B = B.reshape(21, 7**2).transpose((1, 0)).reshape(*B.shape)
        # convert [GROUP_SIZE^2, OUTPUT_DIM] -> [GROUP_SIZE^2, ALIGN_TO_16(OUTPUT_DIM)]
        net_mlu.params[name][0].data[...] = np.pad(W.reshape(7**2, 21, *W.shape[1:]), ((0, 0), (0, 11), (0, 0), (0, 0), (0, 0)), mode='constant').reshape(-1, *W.shape[1:])
        net_mlu.params[name][1].data[...] = np.pad(B.reshape(7**2, 21), ((0, 0), (0, 11)), mode='constant').reshape(7**2 * 32)
    elif name == "rfcn_bbox":
        # 8  * 7**2 -> 16 * 7**2
        W = net.params[name][0].data
        B = net.params[name][1].data
        # convert [OUTPUT_DIM, GROUP_SIZE^2] -> [GROUP_SIZE^2, OUTPUT_DIM]
        W = W.reshape(8, 7**2, *W.shape[1:]).transpose((1, 0, 2, 3, 4)).reshape(*W.shape)
        B = B.reshape(8, 7**2).transpose((1, 0)).reshape(*B.shape)
        # convert [GROUP_SIZE^2, OUTPUT_DIM] -> [GROUP_SIZE^2, ALIGN_TO_16(OUTPUT_DIM)]
        net_mlu.params[name][0].data[...] = np.pad(W.reshape(7**2, 8, *W.shape[1:]), ((0, 0), (0, 8), (0, 0), (0, 0), (0, 0)), mode='constant').reshape(-1, *W.shape[1:])
        net_mlu.params[name][1].data[...] = np.pad(B.reshape(7**2, 8), ((0, 0), (0, 8)), mode='constant').reshape(7**2 * 16)
    else:
        # net_mlu.params[name].data[...] = net.params[name]
        # print name
        for idx in range(len(net.params[name])):
            net_mlu.params[name][idx].data[...] = net.params[name][idx].data

net_mlu.save( '_mlu.'.join(os.path.basename(sys.argv[3]).split('.')))
