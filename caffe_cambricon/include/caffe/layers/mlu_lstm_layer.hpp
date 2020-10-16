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

#ifndef INCLUDE_CAFFE_LAYERS_MLU_LSTM_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_MLU_LSTM_LAYER_HPP_
#ifdef USE_MLU

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN). Implemented by unrolling
 *        the LSTM computation through time.
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
 *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
 *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
 *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
 *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
 *     h_t := o_t .* \tanh[c_t]
 * In the implementation, the i, f, o, and g computations are performed as a
 * single inner product.
 *
 * Notably, this implementation lacks the "diagonal" gates, as used in the
 * LSTM architectures described by Alex Graves [3] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 */

template <typename Dtype>
class MLULSTMLayer : public LSTMLayer<Dtype> {
  public:
  explicit MLULSTMLayer(const LayerParameter& param)
      : LSTMLayer<Dtype>(param), c_reshape_param_ptr_(nullptr),
      bottom0_reshape_param_ptr_(nullptr), bottom1_reshape_param_ptr_(nullptr),
      bottom2_reshape_param_ptr_(nullptr), bottom3_reshape_param_ptr_(nullptr),
      bottom4_reshape_param_ptr_(nullptr), top0_reshape_param_ptr_(nullptr),
      top1_reshape_param_ptr_(nullptr), top2_reshape_param_ptr_(nullptr),
      cont_slice_op_(nullptr), w_xc_x_slice_op_(nullptr),
      h_concat_op_(nullptr), h_concat_op1_(nullptr),
      h_concat_op2_(nullptr), w_xc_op_(nullptr),
      w_xc_static_op_(nullptr), c_reshape_op_(nullptr),
      bottom0_reshape_op_(nullptr), bottom1_reshape_op_(nullptr),
      bottom2_reshape_op_(nullptr), bottom3_reshape_op_(nullptr),
      bottom4_reshape_op_(nullptr), top0_reshape_op_(nullptr),
      top1_reshape_op_(nullptr), top2_reshape_op_(nullptr),
      num_output_(0), channel(1), height(1), width(1) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual inline bool mfus_supported() { return true; }
  virtual void fuse(MFusion<Dtype>* fuser);
  virtual ~MLULSTMLayer();

  protected:
  virtual void MLUDestroyOp();
  virtual void MLUCreateOpBindData(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
  virtual void MLUCompileOp();
  virtual void Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  vector<shared_ptr<Blob<Dtype> > > w_xc_x_blobs_;
  vector<shared_ptr<Blob<Dtype> > > cont_blobs_;
  vector<shared_ptr<Blob<Dtype> > > h_cont_blobs_;
  vector<shared_ptr<Blob<Dtype> > > w_hc_h_blobs_;
  vector<shared_ptr<Blob<Dtype> > > gate_input_blobs_;
  vector<shared_ptr<Blob<Dtype> > > temp_input_blobs_;
  vector<shared_ptr<Blob<Dtype> > > c_blobs_;
  vector<shared_ptr<Blob<Dtype> > > h_blobs_;
  vector<shared_ptr<Blob<Dtype> > > i_f_o_g_blobs_;
  vector<shared_ptr<Blob<Dtype> > > i_blobs_;
  vector<shared_ptr<Blob<Dtype> > > f_blobs_;
  vector<shared_ptr<Blob<Dtype> > > o_blobs_;
  vector<shared_ptr<Blob<Dtype> > > g_blobs_;
  vector<shared_ptr<Blob<Dtype> > > bak_blobs_;

  vector<shared_ptr<Blob<Dtype> > > f_c_reshape_blobs_;
  vector<shared_ptr<Blob<Dtype> > > cont_reshape_blobs_;
  vector<shared_ptr<Blob<Dtype> > > h_reshape_blobs_;

  shared_ptr<Blob<Dtype> > w_xc_x_static_blob_;
  shared_ptr<Blob<Dtype> > w_xc_blob_;
  shared_ptr<Blob<Dtype> > c_reshape_blob_;

  shared_ptr<Blob<Dtype> > bottom0_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom1_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom2_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom3_reshape_blob_;
  shared_ptr<Blob<Dtype> > bottom4_reshape_blob_;

  shared_ptr<Blob<Dtype> > top0_reshape_blob_;
  shared_ptr<Blob<Dtype> > top1_reshape_blob_;
  shared_ptr<Blob<Dtype> > top2_reshape_blob_;

  // temp_blobs_ used in LSTMUnit
  vector<shared_ptr<Blob<Dtype> > > h_cont_reshape_blobs_;  // i_{t} * g_{t}
  vector<shared_ptr<Blob<Dtype> > > i_g_temp_blobs_;  // i_{t} * g_{t}
  vector<shared_ptr<Blob<Dtype> > > f_c_temp_blobs_;  // f_{t} * c_{t}
  vector<shared_ptr<Blob<Dtype> > > cont_temp_blobs_;  // cont_t * f_{t} * c_{t}
  vector<shared_ptr<Blob<Dtype> > > tanh_c_temp_blobs_;  // tanh[c_t]

  cnmlReshapeOpParam_t c_reshape_param_ptr_;

  cnmlReshapeOpParam_t bottom0_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom1_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom2_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom3_reshape_param_ptr_;
  cnmlReshapeOpParam_t bottom4_reshape_param_ptr_;

  cnmlReshapeOpParam_t top0_reshape_param_ptr_;
  cnmlReshapeOpParam_t top1_reshape_param_ptr_;
  cnmlReshapeOpParam_t top2_reshape_param_ptr_;

  vector<cnmlReshapeOpParam_t> h_cont_reshape_param_ptr_;
  vector<cnmlReshapeOpParam_t> f_c_temp_reshape_param_ptr_;
  vector<cnmlReshapeOpParam_t> cont_temp_reshape_param_ptr_;
  vector<cnmlReshapeOpParam_t> h_reshape_param_ptr_;

  cnmlBaseOp_t cont_slice_op_;
  cnmlBaseOp_t w_xc_x_slice_op_;
  cnmlBaseOp_t h_concat_op_;  // top[0]
  cnmlBaseOp_t h_concat_op1_;  // top[1]
  cnmlBaseOp_t h_concat_op2_;  // top[2]
  cnmlBaseOp_t w_xc_op_;
  cnmlBaseOp_t w_xc_static_op_;
  cnmlBaseOp_t c_reshape_op_;

  cnmlBaseOp_t bottom0_reshape_op_;
  cnmlBaseOp_t bottom1_reshape_op_;
  cnmlBaseOp_t bottom2_reshape_op_;
  cnmlBaseOp_t bottom3_reshape_op_;
  cnmlBaseOp_t bottom4_reshape_op_;

  cnmlBaseOp_t top0_reshape_op_;
  cnmlBaseOp_t top1_reshape_op_;
  cnmlBaseOp_t top2_reshape_op_;

  vector<cnmlBaseOp_t> w_hc_op_;
  vector<cnmlBaseOp_t> h_cont_reshape_op_;
  vector<cnmlBaseOp_t> f_c_temp_reshape_op_;
  vector<cnmlBaseOp_t> cont_temp_reshape_op_;
  vector<cnmlBaseOp_t> h_reshape_op_;
  vector<cnmlBaseOp_t> gate_slice_op_;  // gate_input  to i,f,o,g  slice_op

  vector<cnmlBaseOp_t> add_op_;
  vector<cnmlBaseOp_t> add_1_op_;  // w_xc_x + w_xc_x_static ,static_input true
  vector<cnmlBaseOp_t> add_2_op_;  // LSTMUnit  add  c_{t}

  vector<cnmlBaseOp_t> cyclemult_op_;  // h_{t} * cont_{t}
  vector<cnmlBaseOp_t> mult_op_;  // f_{t} * c_{t}
  vector<cnmlBaseOp_t> mult_1_op_;  // i_{t} * g_{t}
  vector<cnmlBaseOp_t> cyclemult_1_op_;  // cont_t * (temp)
  vector<cnmlBaseOp_t> mult_2_op_;  // o_t * tanh{c_t}
  vector<cnmlBaseOp_t> sigmoid_i_op_;
  vector<cnmlBaseOp_t> sigmoid_f_op_;
  vector<cnmlBaseOp_t> sigmoid_o_op_;
  vector<cnmlBaseOp_t> tanh_g_op_;
  vector<cnmlBaseOp_t> tanh_c_op_;
  int num_output_;
  int channel, height, width;
};

}  // namespace caffe
#endif  // USE_MLU
#endif  // INCLUDE_CAFFE_LAYERS_MLU_LSTM_LAYER_HPP_
