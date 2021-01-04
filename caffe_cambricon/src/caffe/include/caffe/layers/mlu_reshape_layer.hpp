/*
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
*/

#ifndef INCLUDE_CAFFE_LAYERS_MLU_RESHAPE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_RESHAPE_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

/**
 * @brief MLU acceleration of ReshapeLayer
 *        ReshapeLayer Takes a Blob and change its dimensions into specified
 *  dimensions,but the content of input blob remain the same.
 */
template <typename Dtype>
class MLUReshapeLayer : public ReshapeLayer<Dtype> {
  public:
  explicit MLUReshapeLayer(const LayerParameter& param)
      : ReshapeLayer<Dtype>(param),
        reshape_op_ptr_(nullptr),
        reshape_param_(nullptr),
        transpose_d2h_param_ptr_(nullptr),
        transpose_d2h_op_ptr_(nullptr),
        transpose_h2d_param_ptr_(nullptr),
        transpose_h2d_op_ptr_(nullptr) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser) {
    fuser->fuse(transpose_d2h_op_ptr_);
    fuser->fuse(reshape_op_ptr_);
    fuser->fuse(transpose_h2d_op_ptr_);
  }
  virtual ~MLUReshapeLayer();

  protected:
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  cnmlBaseOp_t reshape_op_ptr_;
  cnmlReshapeOpParam_t reshape_param_;
  int output_shape_[4];
  cnmlNdTransposeOpParam_t transpose_d2h_param_ptr_;
  cnmlBaseOp_t transpose_d2h_op_ptr_;
  Blob<Dtype> transpose_d2h_blob_;
  cnmlNdTransposeOpParam_t transpose_h2d_param_ptr_;
  cnmlBaseOp_t transpose_h2d_op_ptr_;
  Blob<Dtype> transpose_h2d_blob_;
};
}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_RESHAPE_LAYER_HPP_
