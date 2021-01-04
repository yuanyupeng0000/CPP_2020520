/*
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
*/

#ifndef INCLUDE_CAFFE_LAYERS_MLU_POOLING_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_POOLING_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief MLU acceleration of PoolingLayer
 *        Pools the input image by taking the max, average, etc. within regions.
 */
template <typename Dtype>
class MLUPoolingLayer : public PoolingLayer<Dtype> {
  public:
  explicit MLUPoolingLayer(const LayerParameter& param)
      : PoolingLayer<Dtype>(param),
      pool_param_ptr_(nullptr), pool_op_ptr_(nullptr),
      pool_addpad_op_param_(nullptr), addpad_op_ptr_(nullptr),
      pool_index_op_ptr_(nullptr), pool_index_param_ptr_(nullptr),
      pad_ht_(0), pad_hb_(0), pad_wl_(0), pad_wr_(0), has_pad_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLUPoolingLayer() { MLUDestroyOp(); }

  protected:
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();

  cnmlPoolOpParam_t pool_param_ptr_;
  cnmlBaseOp_t pool_op_ptr_;
  cnmlAddPadOpParam_t pool_addpad_op_param_;
  cnmlBaseOp_t addpad_op_ptr_;
  cnmlBaseOp_t pool_index_op_ptr_;
  cnmlPoolOpParam_t pool_index_param_ptr_;
  Blob<Dtype> addpad_;
  int top_size_;
  int pad_ht_;
  int pad_hb_;
  int pad_wl_;
  int pad_wr_;
  bool has_pad_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_POOLING_LAYER_HPP_
