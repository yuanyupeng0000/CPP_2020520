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

#ifdef USE_MLU
#include <memory>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_lstm_layer.hpp"
#include "caffe/util/io.hpp"
namespace caffe {

template <typename Dtype>
void MLULSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  this->T_ = bottom[0]->shape(0);
  this->N_ = bottom[0]->shape(1);
  LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
            << this->T_ << " timesteps of "
            << this->N_ << " independent streams.";

  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(this->T_, bottom[1]->shape(0));
  CHECK_EQ(this->N_, bottom[1]->shape(1));

  // If expose_hidden is set, we take as input and produce as output
  // the hidden state blobs at the first and last timesteps.
  this->expose_hidden_ = this->layer_param_.recurrent_param().expose_hidden();

  // Get (recurrent) input/output names.
  vector<string> output_names;
  this->OutputBlobNames(&output_names);
  vector<string> recur_input_names;
  this->RecurrentInputBlobNames(&recur_input_names);
  vector<string> recur_output_names;
  this->RecurrentOutputBlobNames(&recur_output_names);
  const int num_recur_blobs = recur_input_names.size();
  CHECK_EQ(num_recur_blobs, recur_output_names.size());

  // If provided, bottom[2] is a static input to the recurrent net.
  const int num_hidden_exposed = this->expose_hidden_ * num_recur_blobs;
  this->static_input_ = (bottom.size() > 2 + num_hidden_exposed);
  if (this->static_input_) {
    CHECK_GE(bottom[2]->num_axes(), 1);
    CHECK_EQ(this->N_, bottom[2]->shape(0));
  }
  CHECK_EQ(top.size() - this->expose_hidden_*2, 1);

  // num_output
  num_output_ = this->layer_param_.recurrent_param().num_output();

  w_hc_op_.resize(this->T_, nullptr);
  h_cont_reshape_op_.resize(this->T_, nullptr);
  h_cont_reshape_param_ptr_.resize(this->T_, nullptr);

  add_op_.resize(this->T_, nullptr);
  add_1_op_.resize(this->T_, nullptr);
  add_2_op_.resize(this->T_, nullptr);

  cyclemult_op_.resize(this->T_, nullptr);
  add_op_.resize(this->T_, nullptr);
  mult_op_.resize(this->T_, nullptr);
  mult_1_op_.resize(this->T_, nullptr);
  cyclemult_1_op_.resize(this->T_, nullptr);
  mult_2_op_.resize(this->T_, nullptr);

  sigmoid_i_op_.resize(this->T_, nullptr);
  sigmoid_f_op_.resize(this->T_, nullptr);
  sigmoid_o_op_.resize(this->T_, nullptr);

  tanh_g_op_.resize(this->T_, nullptr);
  tanh_c_op_.resize(this->T_, nullptr);

  gate_slice_op_.resize(this->T_, nullptr);

  f_c_temp_reshape_op_.resize(this->T_, nullptr);
  f_c_temp_reshape_param_ptr_.resize(this->T_, nullptr);

  cont_temp_reshape_op_.resize(this->T_, nullptr);
  cont_temp_reshape_param_ptr_.resize(this->T_, nullptr);

  if (this->expose_hidden_) {
    h_reshape_op_.resize(this->T_ - 1, nullptr);
    h_reshape_param_ptr_.resize(this->T_ - 1, nullptr);
  } else {
    h_reshape_op_.resize(this->T_ - 1, nullptr);
    h_reshape_param_ptr_.resize(this->T_ - 1, nullptr);
  }

  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
      this->layer_param_.blobs_dtype(0).type() : DT_FLOAT16;

  vector<int> input_shape1 = bottom[0]->shape();
  if (input_shape1.size() > 2)
    channel = input_shape1[2];
  if (input_shape1.size() > 3)
    height = input_shape1[3];
  if (input_shape1.size() > 4)
    width = input_shape1[4];

  // set weights and bias
  this->blobs_.clear();
  if (this->static_input_)
    this->blobs_.resize(4);
  else
    this->blobs_.resize(3);

  // w_xc
  vector<int> weight_shape(4, 1);
  weight_shape[0] = 4 * num_output_;
  weight_shape[1] = channel;
  weight_shape[2] = height;
  weight_shape[3] = width;

  this->blobs_[0].reset(new Blob<Dtype>(weight_shape,
                                       cpu_dtype,
                                       mlu_dtype,
                                       CNML_FILTER));
  // b_c
  BaseDataType mlu_dtype1 = bottom[0]->mlu_type();
  vector<int> bias_shape(4, 1);
  bias_shape[1] = num_output_ * 4;
  this->blobs_[1].reset(new Blob<Dtype>(bias_shape,
                                       cpu_dtype,
                                       mlu_dtype1,
                                       CNML_CONST));
  vector<int> w_hc_shape(4, 1);
  w_hc_shape[0] = num_output_ * 4;
  w_hc_shape[1] = num_output_;
  // w_xc_static
  if (this->static_input_) {
    vector<int> w_xc_static_shape = bottom[2]->shape();
    w_xc_static_shape[0] = num_output_ * 4;
    this->blobs_[2].reset(new Blob<Dtype>(w_xc_static_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
    // w_hc
    this->blobs_[3].reset(new Blob<Dtype>(w_hc_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
  } else {
    // w_hc
    this->blobs_[2].reset(new Blob<Dtype>(w_hc_shape,
                                        cpu_dtype,
                                        mlu_dtype,
                                        CNML_FILTER));
  }

  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.recurrent_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
      this->layer_param_.recurrent_param().bias_filler()));
  bias_filler->Fill(this->blobs_[1].get());
  if (this->static_input_) {
    weight_filler->Fill(this->blobs_[2].get());
    weight_filler->Fill(this->blobs_[3].get());
  } else {
    weight_filler->Fill(this->blobs_[2].get());
  }

  // int8 position
  if (this->static_input_) {
    if (this->layer_param_.blobs_dtype_size() > 2 &&
      (this->layer_param_.blobs_dtype(0).position_size() ||
      this->layer_param_.blobs_dtype(0).scale_size()) &&
      (this->layer_param_.blobs_dtype(1).position_size() ||
      this->layer_param_.blobs_dtype(1).scale_size()) &&
      (this->layer_param_.blobs_dtype(2).position_size() ||
       this->layer_param_.blobs_dtype(2).scale_size())) {
      if (this->layer_param_.blobs_dtype(0).position_size() &&
          this->layer_param_.blobs_dtype(1).position_size() &&
          this->layer_param_.blobs_dtype(2).position_size()) {
        this->blobs_[0]->set_mlu_position(
            this->layer_param_.blobs_dtype(0).position(0));
        this->blobs_[2]->set_mlu_position(
            this->layer_param_.blobs_dtype(1).position(0));
        this->blobs_[3]->set_mlu_position(
            this->layer_param_.blobs_dtype(2).position(0));
      }
      if (this->layer_param_.blobs_dtype(0).scale_size() &&
          this->layer_param_.blobs_dtype(1).scale_size() &&
          this->layer_param_.blobs_dtype(2).scale_size()) {
          this->blobs_[0]->set_mlu_scale(
              this->layer_param_.blobs_dtype(0).scale(0));
          this->blobs_[2]->set_mlu_scale(
              this->layer_param_.blobs_dtype(1).scale(0));
          this->blobs_[3]->set_mlu_scale(
              this->layer_param_.blobs_dtype(2).scale(0));
      }
    }
  } else {
    if (this->layer_param_.blobs_dtype_size() > 1 &&
      (this->layer_param_.blobs_dtype(0).position_size() ||
       this->layer_param_.blobs_dtype(0).scale_size()) &&
      (this->layer_param_.blobs_dtype(1).position_size() ||
       this->layer_param_.blobs_dtype(1).scale_size())) {
      if (this->layer_param_.blobs_dtype(0).position_size() &&
          this->layer_param_.blobs_dtype(1).position_size()) {
        this->blobs_[0]->set_mlu_position(
            this->layer_param_.blobs_dtype(0).position(0));
        this->blobs_[2]->set_mlu_position(
            this->layer_param_.blobs_dtype(1).position(0));
      }
      if (this->layer_param_.blobs_dtype(0).scale_size() &&
          this->layer_param_.blobs_dtype(1).scale_size()) {
          this->blobs_[0]->set_mlu_scale(
              this->layer_param_.blobs_dtype(0).scale(0));
          this->blobs_[2]->set_mlu_scale(
              this->layer_param_.blobs_dtype(1).scale(0));
      }
    }
  }
}

template <typename Dtype>
void MLULSTMLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  BaseDataType bak_mlu_dtype = this->layer_param_.blobs_dtype_size() > 0 ?
      this->layer_param_.blobs_dtype(0).type() : DT_FLOAT16;
  // bottom0 reshape
  vector<int> bottom_shape = {this->T_ * this->N_, channel, height, width};
  bottom0_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  if (this->layer_param_.bottom_mlu_dtype_size() > 0 &&
      (this->layer_param_.bottom_mlu_dtype(0).position_size() ||
       this->layer_param_.bottom_mlu_dtype(0).scale_size())) {
    if (this->layer_param_.bottom_mlu_dtype(0).position_size()) {
      bottom0_reshape_blob_->set_mlu_position(
          this->layer_param_.bottom_mlu_dtype(0).position(0));
    }
    if (this->layer_param_.bottom_mlu_dtype(0).scale_size()) {
      bottom0_reshape_blob_->set_mlu_scale(
          this->layer_param_.bottom_mlu_dtype(0).scale(0));
    }
  }

  // bottom1 reshape
  vector<int> bottom_shape1 = {this->T_, this->N_, 1, 1};
  bottom1_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_shape1, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // bottom2 reshape
  if (this->static_input_) {
    vector<int> static_shape(4, 1);
    static_shape[0] = this->N_;
    static_shape[1] = bottom[2]->shape(1);
    bottom2_reshape_blob_.reset(new Blob<Dtype>(
                   static_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    if (this->layer_param_.bottom_mlu_dtype_size() > 1 &&
        (this->layer_param_.bottom_mlu_dtype(1).position_size()||
         this->layer_param_.bottom_mlu_dtype(1).scale_size())) {
      if (this->layer_param_.bottom_mlu_dtype(1).position_size()) {
        bottom2_reshape_blob_->set_mlu_position(
            this->layer_param_.bottom_mlu_dtype(1).position(0));
      }
      if (this->layer_param_.bottom_mlu_dtype(1).scale_size()) {
        bottom2_reshape_blob_->set_mlu_scale(
            this->layer_param_.bottom_mlu_dtype(1).scale(0));
      }
    }
  }
  // bottom3, bottom4 reshape
  if (this->expose_hidden_) {
    vector<int> bottom_hidden_shape(4, 1);
    bottom_hidden_shape[1] = this->N_;
    bottom_hidden_shape[2] = num_output_;
    bottom3_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
    bottom4_reshape_blob_.reset(new Blob<Dtype>(
                   bottom_hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // top[0]
  vector<int> top_shape(3, 1);
  top_shape[0] =  this->T_;
  top_shape[1] =  this->N_;
  top_shape[2] = num_output_;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);

  // top[1], top[2]
  if (this->expose_hidden_) {
    vector<int> hidden_shape(3, 1);
    hidden_shape[1] = this->N_;
    hidden_shape[2] = num_output_;
    top[1]->Reshape(hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
    top[2]->Reshape(hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
  }

  // bak_blobs
  vector<int> w_hc_shape(4, 1);
  w_hc_shape[0] = num_output_ * 4;
  w_hc_shape[1] = num_output_;
  bak_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    bak_blobs_[i].reset(new Blob<Dtype>(
                w_hc_shape, cpu_dtype, bak_mlu_dtype, CNML_FILTER));
  }

  // w_xc_blob_ MLP
  vector<int> w_xc_shape(4, 1);
  w_xc_shape[0] = this->T_ * this->N_;
  w_xc_shape[1] = num_output_ * 4;
  w_xc_blob_.reset(new Blob<Dtype>(
                   w_xc_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // w_xc_x_blobs  Slice
  w_xc_x_blobs_.clear();
  w_xc_x_blobs_.resize(this->T_);
  vector<int> w_xc_x_shape(4, 1);
  w_xc_x_shape[0] = this->N_;
  w_xc_x_shape[1] = num_output_ * 4;
  for (int i = 0; i < this->T_; i++) {
    w_xc_x_blobs_[i].reset(new Blob<Dtype>(
                   w_xc_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // gate_input_blobs_ Eltwise
  // shape: 2, 72
  gate_input_blobs_.clear();
  gate_input_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    gate_input_blobs_[i].reset(new Blob<Dtype>(
                   w_xc_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // temp_input_blobs_ Eltwise,  if static_input_ is true
  // shape: 2, 72
  temp_input_blobs_.clear();
  temp_input_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    temp_input_blobs_[i].reset(new Blob<Dtype>(
                    w_xc_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // w_xc_x_static MLP
  if (this->static_input_)
    w_xc_x_static_blob_.reset(new Blob<Dtype>(
                    w_xc_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // w_hc_h_blobs MLP
  w_hc_h_blobs_.clear();
  w_hc_h_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    w_hc_h_blobs_[i].reset(new Blob<Dtype>(
                    w_xc_x_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_cont_reshape_blobs_
  h_cont_reshape_blobs_.clear();
  h_cont_reshape_blobs_.resize(this->T_);
  vector<int> cont_reshape_shape(4, 1);
  cont_reshape_shape[0] = this->N_;
  cont_reshape_shape[1] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    h_cont_reshape_blobs_[i].reset(new Blob<Dtype>(
                    cont_reshape_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  if (this->static_input_) {
    for (int i = 0; i < this->T_; i++) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 2 + i &&
          (this->layer_param_.bottom_mlu_dtype(2 + i).position_size() ||
           this->layer_param_.bottom_mlu_dtype(2 + i).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(2 + i).position_size()) {
          h_cont_reshape_blobs_[i]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(2 + i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(2 + i).scale_size()) {
          h_cont_reshape_blobs_[i]->set_mlu_scale(
            this->layer_param_.bottom_mlu_dtype(2 + i).scale(0));
        }
      }
    }
  } else {
    for (int i = 0; i < this->T_; i++) {
      if (this->layer_param_.bottom_mlu_dtype_size() > 1 + i &&
          (this->layer_param_.bottom_mlu_dtype(1 + i).position_size() ||
           this->layer_param_.bottom_mlu_dtype(1 + i).scale_size())) {
        if (this->layer_param_.bottom_mlu_dtype(1 + i).position_size()) {
          h_cont_reshape_blobs_[i]->set_mlu_position(
              this->layer_param_.bottom_mlu_dtype(1 + i).position(0));
        }
        if (this->layer_param_.bottom_mlu_dtype(1 + i).scale_size()) {
          h_cont_reshape_blobs_[i]->set_mlu_scale(
            this->layer_param_.bottom_mlu_dtype(1 + i).scale(0));
        }
      }
    }
  }

  // c_reshape_blob_
  if (this->expose_hidden_) {
    c_reshape_blob_.reset(new Blob<Dtype>(
                    cont_reshape_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // cont_blobs Slice
  cont_blobs_.clear();
  cont_blobs_.resize(this->T_);
  vector<int> cont_shape(4, 1);
  cont_shape[1] = this->N_;
  for (int i = 0; i < this->T_; i++) {
    cont_blobs_[i].reset(new Blob<Dtype>(
                   cont_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_cont_blobs  Scale
  h_cont_blobs_.clear();
  h_cont_blobs_.resize(this->T_);
  vector<int> h_cont_shape(4, 1);
  h_cont_shape[1] = this->N_;
  h_cont_shape[2] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    h_cont_blobs_[i].reset(new Blob<Dtype>(
                      h_cont_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // c_blobs  LSTMUnit c_{t}
  c_blobs_.clear();
  c_blobs_.resize(this->T_+1);
  vector<int> c_h_shape(4, 1);
  c_h_shape[0] = this->N_;
  c_h_shape[1] = num_output_;
  c_blobs_[0].reset(new Blob<Dtype>(
               c_h_shape, cpu_dtype, mlu_dtype, CNML_CONST));
  for (int i = 1; i < this->T_ + 1; i++) {
    c_blobs_[i].reset(new Blob<Dtype>(
                 c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
  // h_blobs  LSTMUnit h_{t}
  h_blobs_.clear();
  h_blobs_.resize(this->T_+1);
  h_blobs_[0].reset(new Blob<Dtype>(
               h_cont_shape, cpu_dtype, mlu_dtype, CNML_CONST));
  for (int i = 1; i < this->T_ + 1; i++) {
    h_blobs_[i].reset(new Blob<Dtype>(
                 c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
  // i_f_o_g_blobs
  i_f_o_g_blobs_.clear();
  i_f_o_g_blobs_.resize(this->T_ * 4);
  for (int i = 0; i < this->T_ * 4; i++) {
    i_f_o_g_blobs_[i].reset(new Blob<Dtype>(
                   c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // i_blobs
  i_blobs_.clear();
  i_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    i_blobs_[i].reset(new Blob<Dtype>(
                c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // f_blobs
  f_blobs_.clear();
  f_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    f_blobs_[i].reset(new Blob<Dtype>(
                c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // o_blobs
  o_blobs_.clear();
  o_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    o_blobs_[i].reset(new Blob<Dtype>(
                 c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // g_blobs
  g_blobs_.clear();
  g_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    g_blobs_[i].reset(new Blob<Dtype>(
                 c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // temp blobs_ used in LSTMUnit
  // i_g_temp_blobs_
  i_g_temp_blobs_.clear();
  i_g_temp_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    i_g_temp_blobs_[i].reset(new Blob<Dtype>(
            c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // f_c_temp_blobs_
  f_c_temp_blobs_.clear();
  f_c_temp_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    f_c_temp_blobs_[i].reset(new Blob<Dtype>(
          c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // f_c_reshape_blobs_
  // reshape  2,18,1,1, to 1,2,18,1
  f_c_reshape_blobs_.clear();
  f_c_reshape_blobs_.resize(this->T_);
  vector<int> f_c_reshape_shape(4, 1);
  f_c_reshape_shape[1] = this->N_;
  f_c_reshape_shape[2] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    f_c_reshape_blobs_[i].reset(new Blob<Dtype>(
                f_c_reshape_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // h_reshape_blobs_
  // reshape  2,18,1,1, to 1,2,18,1
  h_reshape_blobs_.clear();
  h_reshape_blobs_.resize(this->T_-1);
  for (int i = 0; i < this->T_-1; i++) {
    h_reshape_blobs_[i].reset(new Blob<Dtype>(
                   f_c_reshape_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // cont_temp_blobs_
  cont_temp_blobs_.clear();
  cont_temp_blobs_.resize(this->T_);
  vector<int> cont_temp_shape(4, 1);
  cont_temp_shape[1] = this->N_;
  cont_temp_shape[2] = num_output_;
  for (int i = 0; i < this->T_; i++) {
    cont_temp_blobs_[i].reset(new Blob<Dtype>(
                          cont_temp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // cont_reshape_blobs_
  // reshape 1,2,18,1 to 2,18,1,1
  cont_reshape_blobs_.clear();
  cont_reshape_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    cont_reshape_blobs_[i].reset(new Blob<Dtype>(
                          c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }

  // tanh_c_temp_blobs_
  tanh_c_temp_blobs_.clear();
  tanh_c_temp_blobs_.resize(this->T_);
  for (int i = 0; i < this->T_; i++) {
    tanh_c_temp_blobs_[i].reset(new Blob<Dtype>(
                          c_h_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
  // top0_reshape_blob_
  vector<int> top_temp_shape(4, 1);
  top_temp_shape[0] = this->T_ * this->N_;
  top_temp_shape[1] = num_output_;
  top0_reshape_blob_.reset(new Blob<Dtype>(
                        top_temp_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

  // top1_reshape_blob_, top2_reshape_blob_
  if (this->expose_hidden_) {
    vector<int> top_hidden_shape(4, 1);
    top_hidden_shape[0] = this->N_;
    top_hidden_shape[1] = num_output_;
    top1_reshape_blob_.reset(new Blob<Dtype>(
                        top_hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));

    top2_reshape_blob_.reset(new Blob<Dtype>(
                        top_hidden_shape, cpu_dtype, mlu_dtype, CNML_TENSOR));
  }
}

template <typename Dtype>
void MLULSTMLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int bottom0_dim[4];
  bottom0_dim[0] = bottom0_reshape_blob_->shape(0);
  bottom0_dim[1] = bottom0_reshape_blob_->shape(1);
  bottom0_dim[2] = bottom0_reshape_blob_->shape(2);
  bottom0_dim[3] = bottom0_reshape_blob_->shape(3);
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom0_reshape_param_ptr_,
                                     bottom0_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&bottom0_reshape_op_,
                         bottom0_reshape_param_ptr_,
                         bottom[0]->mlu_tensor(),
                         bottom0_reshape_blob_->mlu_tensor()));

  int bottom1_dim[4];
  bottom1_dim[0] = bottom1_reshape_blob_->shape(0);
  bottom1_dim[1] = bottom1_reshape_blob_->shape(1);
  bottom1_dim[2] = bottom1_reshape_blob_->shape(2);
  bottom1_dim[3] = bottom1_reshape_blob_->shape(3);
  MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom1_reshape_param_ptr_,
                                     bottom1_dim,
                                     4));
  MLU_CHECK(cnmlCreateReshapeOp(&bottom1_reshape_op_,
                         bottom1_reshape_param_ptr_,
                         bottom[1]->mlu_tensor(),
                         bottom1_reshape_blob_->mlu_tensor()));

  if (this->static_input_) {
    int bottom2_dim[4];
    bottom2_dim[0] = bottom2_reshape_blob_->shape(0);
    bottom2_dim[1] = bottom2_reshape_blob_->shape(1);
    bottom2_dim[2] = bottom2_reshape_blob_->shape(2);
    bottom2_dim[3] = bottom2_reshape_blob_->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom2_reshape_param_ptr_,
                                       bottom2_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&bottom2_reshape_op_,
                           bottom2_reshape_param_ptr_,
                           bottom[2]->mlu_tensor(),
                           bottom2_reshape_blob_->mlu_tensor()));
  }
  if (this->expose_hidden_) {
    int bottom3_dim[4];
    bottom3_dim[0] = bottom3_reshape_blob_->shape(0);
    bottom3_dim[1] = bottom3_reshape_blob_->shape(1);
    bottom3_dim[2] = bottom3_reshape_blob_->shape(2);
    bottom3_dim[3] = bottom3_reshape_blob_->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom3_reshape_param_ptr_,
                                       bottom3_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&bottom3_reshape_op_,
                           bottom3_reshape_param_ptr_,
                           bottom[3]->mlu_tensor(),
                           bottom3_reshape_blob_->mlu_tensor()));
    int bottom4_dim[4];
    bottom4_dim[0] = bottom4_reshape_blob_->shape(0);
    bottom4_dim[1] = bottom4_reshape_blob_->shape(1);
    bottom4_dim[2] = bottom4_reshape_blob_->shape(2);
    bottom4_dim[3] = bottom4_reshape_blob_->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&bottom4_reshape_param_ptr_,
                                       bottom4_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&bottom4_reshape_op_,
                           bottom4_reshape_param_ptr_,
                           bottom[4]->mlu_tensor(),
                           bottom4_reshape_blob_->mlu_tensor()));
  }


  // slice cont_{t}
  cnmlTensor_t cont_tensors[this->T_];  // NOLINT
  for (int i = 0; i < this->T_; i++) {
    cont_tensors[i] = cont_blobs_[i]->mlu_tensor();
  }
  cnmlTensor_t bottom_cont_tensor = bottom1_reshape_blob_->mlu_tensor();
  MLU_CHECK(cnmlCreateSplitOp(&cont_slice_op_,
                             0,
                             &bottom_cont_tensor,
                             1,
                             cont_tensors,
                             this->T_));

  // W_xc_x = W_xc * x + b_c
  MLU_CHECK(cnmlCreateMlpOp(&w_xc_op_,
                           bottom0_reshape_blob_->mlu_tensor(),
                           w_xc_blob_->mlu_tensor(),
                           this->blobs_[0]->mlu_tensor(),
                           this->blobs_[1]->mlu_tensor()));

  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[0]->mlu_tensor(),
                           this->blobs_[0]->sync_data(),
                           false));
  MLU_CHECK(cnmlBindConstData_V2(this->blobs_[1]->mlu_tensor(),
                           this->blobs_[1]->sync_data(),
                           false));

  // slice w_xc_x_{t}
  cnmlTensor_t w_xc_x_tensors[this->T_];  // NOLINT
  for (int i = 0; i < this->T_; i++) {
    w_xc_x_tensors[i] = w_xc_x_blobs_[i]->mlu_tensor();
  }
  cnmlTensor_t w_xc_tensor = w_xc_blob_->mlu_tensor();
  MLU_CHECK(cnmlCreateNdSplitOp(&w_xc_x_slice_op_,
                             0,
                             &w_xc_tensor,
                             1,
                             w_xc_x_tensors,
                             this->T_));

  if (this->static_input_) {
    // W_xc_x_static = W_xc_static * x_static
    // w_xc_x_static = blobs[2]* bottom[2]
    MLU_CHECK(cnmlCreateMlpOp(&w_xc_static_op_,
                             bottom2_reshape_blob_->mlu_tensor(),
                             w_xc_x_static_blob_->mlu_tensor(),
                             this->blobs_[2]->mlu_tensor(),
                             nullptr));
    MLU_CHECK(cnmlBindConstData_V2(this->blobs_[2]->mlu_tensor(),
                                   this->blobs_[2]->sync_data(),
                                   false));
  }

  // h_conted_{t-1} := cont_t * h_{t-1}
  for (int i = 0; i < this->T_; i++) {
    if (i == 0) {
      if (this->expose_hidden_) {
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               bottom4_reshape_blob_->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_cont_blobs_[i]->mlu_tensor()));
      } else {
        caffe_set(h_blobs_[0]->count(), Dtype(0),
                h_blobs_[0]->mutable_cpu_data());
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               h_blobs_[i]->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_cont_blobs_[i]->mlu_tensor()));
        MLU_CHECK(cnmlBindConstData_V2(h_blobs_[i]->mlu_tensor(),
                               h_blobs_[i]->sync_data(),
                               false));
      }
    } else {
      if (this->expose_hidden_) {
        int h_reshape_dim[4];
        h_reshape_dim[0] = h_reshape_blobs_[i-1]->shape(0);
        h_reshape_dim[1] = h_reshape_blobs_[i-1]->shape(1);
        h_reshape_dim[2] = h_reshape_blobs_[i-1]->shape(2);
        h_reshape_dim[3] = h_reshape_blobs_[i-1]->shape(3);
        MLU_CHECK(cnmlCreateNdReshapeOpParam(&h_reshape_param_ptr_[i-1],
                                           h_reshape_dim,
                                           4));
        MLU_CHECK(cnmlCreateReshapeOp(&h_reshape_op_[i-1],
                               h_reshape_param_ptr_[i-1],
                               h_blobs_[i]->mlu_tensor(),
                               h_reshape_blobs_[i-1]->mlu_tensor()));
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               h_reshape_blobs_[i-1]->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_cont_blobs_[i]->mlu_tensor()));
      } else {
        int h_reshape_dim[4];
        h_reshape_dim[0] = h_reshape_blobs_[i-1]->shape(0);
        h_reshape_dim[1] = h_reshape_blobs_[i-1]->shape(1);
        h_reshape_dim[2] = h_reshape_blobs_[i-1]->shape(2);
        h_reshape_dim[3] = h_reshape_blobs_[i-1]->shape(3);
        MLU_CHECK(cnmlCreateNdReshapeOpParam(&h_reshape_param_ptr_[i-1],
                                           h_reshape_dim,
                                           4));
        MLU_CHECK(cnmlCreateReshapeOp(&h_reshape_op_[i-1],
                               h_reshape_param_ptr_[i-1],
                               h_blobs_[i]->mlu_tensor(),
                               h_reshape_blobs_[i-1]->mlu_tensor()));
        MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_op_[i],
                               h_reshape_blobs_[i-1]->mlu_tensor(),
                               cont_blobs_[i]->mlu_tensor(),
                               h_cont_blobs_[i]->mlu_tensor()));
      }
    }
    int h_cont_reshape_dim[4];
    h_cont_reshape_dim[0] = h_cont_reshape_blobs_[i]->shape(0);
    h_cont_reshape_dim[1] = h_cont_reshape_blobs_[i]->shape(1);
    h_cont_reshape_dim[2] = h_cont_reshape_blobs_[i]->shape(2);
    h_cont_reshape_dim[3] = h_cont_reshape_blobs_[i]->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&h_cont_reshape_param_ptr_[i],
                                       h_cont_reshape_dim,
                                       4));


    MLU_CHECK(cnmlCreateReshapeOp(&h_cont_reshape_op_[i],
                           h_cont_reshape_param_ptr_[i],
                           h_cont_blobs_[i]->mlu_tensor(),
                           h_cont_reshape_blobs_[i]->mlu_tensor()));


    if (this->static_input_) {
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[3]->count(); q++) {
          bak_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[3]->mutable_cpu_data()[q];
        }
        if (this->blobs_[3]->mlu_type() == DT_INT8) {
            if (this->layer_param_.blobs_dtype(2).position_size())
              bak_blobs_[j]->set_mlu_position(
                    this->layer_param_.blobs_dtype(2).position(0));
            if (this->layer_param_.blobs_dtype(2).scale_size())
              bak_blobs_[j]->set_mlu_scale(
                    this->layer_param_.blobs_dtype(2).scale(0));
        }
      }
    } else {
      for (int j = 0; j < this->T_; j++) {
        for (int q = 0; q < this->blobs_[2]->count(); q++) {
          bak_blobs_[j]->mutable_cpu_data()[q] =
              this->blobs_[2]->mutable_cpu_data()[q];
        }
        if (this->blobs_[2]->mlu_type() == DT_INT8) {
            if (this->layer_param_.blobs_dtype(1).position_size())
              bak_blobs_[j]->set_mlu_position(
                    this->layer_param_.blobs_dtype(1).position(0));
            if (this->layer_param_.blobs_dtype(1).scale_size())
              bak_blobs_[j]->set_mlu_scale(
                    this->layer_param_.blobs_dtype(1).scale(0));
        }
      }
    }
    // W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    if (this->static_input_) {
      MLU_CHECK(cnmlCreateMlpOp(&w_hc_op_[i],
                               h_cont_reshape_blobs_[i]->mlu_tensor(),
                               w_hc_h_blobs_[i]->mlu_tensor(),
                               bak_blobs_[i]->mlu_tensor(),
                               nullptr));

      MLU_CHECK(cnmlBindConstData_V2(bak_blobs_[i]->mlu_tensor(),
                               bak_blobs_[i]->sync_data(),
                               false));

    } else {
    // W_hc_h_{t-1} := W_hc * h_conted_{t-1}
      MLU_CHECK(cnmlCreateMlpOp(&w_hc_op_[i],
                               h_cont_reshape_blobs_[i]->mlu_tensor(),
                               w_hc_h_blobs_[i]->mlu_tensor(),
                               bak_blobs_[i]->mlu_tensor(),
                               nullptr));


      MLU_CHECK(cnmlBindConstData_V2(bak_blobs_[i]->mlu_tensor(),
                               bak_blobs_[i]->sync_data(),
                               false));
    }
    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    if (!this->static_input_) {
      MLU_CHECK(cnmlCreateAddOp(&add_op_[i],
                               w_hc_h_blobs_[i]->mlu_tensor(),
                               w_xc_x_blobs_[i]->mlu_tensor(),
                               gate_input_blobs_[i]->mlu_tensor()));
    } else {
      MLU_CHECK(cnmlCreateAddOp(&add_op_[i],
                               w_hc_h_blobs_[i]->mlu_tensor(),
                               w_xc_x_blobs_[i]->mlu_tensor(),
                               temp_input_blobs_[i]->mlu_tensor()));

      MLU_CHECK(cnmlCreateAddOp(&add_1_op_[i],
                               temp_input_blobs_[i]->mlu_tensor(),
                               w_xc_x_static_blob_->mlu_tensor(),
                               gate_input_blobs_[i]->mlu_tensor()));
    }
    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]

    cnmlTensor_t gate_output_tensors[4];
    for (int j = 0; j < 4; j++) {
      gate_output_tensors[j] = i_f_o_g_blobs_[j + i * 4]->mlu_tensor();
    }
    cnmlTensor_t gate_input_tensor = gate_input_blobs_[i]->mlu_tensor();
    MLU_CHECK(cnmlCreateNdSplitOp(&gate_slice_op_[i],
                               1,
                               &gate_input_tensor,
                               1,
                               gate_output_tensors,
                               4));

    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    // i_t := \sigmoid[i_t']
    MLU_CHECK(cnmlCreateActiveOp(&sigmoid_i_op_[i],
                                cnmlActiveFunction_t::CNML_ACTIVE_SIGMOID,
                                i_f_o_g_blobs_[i * 4]->mlu_tensor(),
                                i_blobs_[i]->mlu_tensor()));

    // f_t := \sigmoid[f_t']
    MLU_CHECK(cnmlCreateActiveOp(&sigmoid_f_op_[i],
                                cnmlActiveFunction_t::CNML_ACTIVE_SIGMOID,
                                i_f_o_g_blobs_[i * 4 + 1]->mlu_tensor(),
                                f_blobs_[i]->mlu_tensor()));

    // o_t := \sigmoid[o_t']
    MLU_CHECK(cnmlCreateActiveOp(&sigmoid_o_op_[i],
                                cnmlActiveFunction_t::CNML_ACTIVE_SIGMOID,
                                i_f_o_g_blobs_[i * 4 + 2]->mlu_tensor(),
                                o_blobs_[i]->mlu_tensor()));

    // g_t := \tanh[g_t']
    MLU_CHECK(cnmlCreateActiveOp(&tanh_g_op_[i],
                                cnmlActiveFunction_t::CNML_ACTIVE_TANH,
                                i_f_o_g_blobs_[i * 4 + 3]->mlu_tensor(),
                                g_blobs_[i]->mlu_tensor()));

    // c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    MLU_CHECK(cnmlCreateMultOp(&mult_1_op_[i],
                              i_blobs_[i]->mlu_tensor(),
                              g_blobs_[i]->mlu_tensor(),
                              i_g_temp_blobs_[i]->mlu_tensor()));
    if (i == 0) {
      if (this->expose_hidden_) {
        int c_reshape_dim[4];
        c_reshape_dim[0] = c_reshape_blob_->shape(0);
        c_reshape_dim[1] = c_reshape_blob_->shape(1);
        c_reshape_dim[2] = c_reshape_blob_->shape(2);
        c_reshape_dim[3] = c_reshape_blob_->shape(3);
        MLU_CHECK(cnmlCreateNdReshapeOpParam(&c_reshape_param_ptr_,
                                           c_reshape_dim,
                                           4));

        MLU_CHECK(cnmlCreateReshapeOp(&c_reshape_op_,
                               c_reshape_param_ptr_,
                               bottom3_reshape_blob_->mlu_tensor(),
                               c_reshape_blob_->mlu_tensor()));

        MLU_CHECK(cnmlCreateMultOp(&mult_op_[i],
                               f_blobs_[i]->mlu_tensor(),
                               c_reshape_blob_->mlu_tensor(),
                               f_c_temp_blobs_[i]->mlu_tensor()));
      } else {
        caffe_set(c_blobs_[0]->count(), Dtype(0),
                c_blobs_[0]->mutable_cpu_data());
        MLU_CHECK(cnmlCreateMultOp(&mult_op_[i],
                               f_blobs_[i]->mlu_tensor(),
                               c_blobs_[i]->mlu_tensor(),
                               f_c_temp_blobs_[i]->mlu_tensor()));
        MLU_CHECK(cnmlBindConstData_V2(c_blobs_[i]->mlu_tensor(),
                               c_blobs_[i]->sync_data(),
                               false));
      }
    } else {
      MLU_CHECK(cnmlCreateMultOp(&mult_op_[i],
                               f_blobs_[i]->mlu_tensor(),
                               c_blobs_[i]->mlu_tensor(),
                               f_c_temp_blobs_[i]->mlu_tensor()));
    }

    int f_c_reshape_dim[4];
    f_c_reshape_dim[0] = f_c_reshape_blobs_[i]->shape(0);
    f_c_reshape_dim[1] = f_c_reshape_blobs_[i]->shape(1);
    f_c_reshape_dim[2] = f_c_reshape_blobs_[i]->shape(2);
    f_c_reshape_dim[3] = f_c_reshape_blobs_[i]->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&f_c_temp_reshape_param_ptr_[i],
                                       f_c_reshape_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&f_c_temp_reshape_op_[i],
                           f_c_temp_reshape_param_ptr_[i],
                           f_c_temp_blobs_[i]->mlu_tensor(),
                           f_c_reshape_blobs_[i]->mlu_tensor()));

    MLU_CHECK(cnmlCreateCycleMultOp(&cyclemult_1_op_[i],
                              f_c_reshape_blobs_[i]->mlu_tensor(),
                              cont_blobs_[i]->mlu_tensor(),
                              cont_temp_blobs_[i]->mlu_tensor()));

    int cont_temp_reshape_dim[4];
    cont_temp_reshape_dim[0] = cont_reshape_blobs_[i]->shape(0);
    cont_temp_reshape_dim[1] = cont_reshape_blobs_[i]->shape(1);
    cont_temp_reshape_dim[2] = cont_reshape_blobs_[i]->shape(2);
    cont_temp_reshape_dim[3] = cont_reshape_blobs_[i]->shape(3);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&cont_temp_reshape_param_ptr_[i],
                                       cont_temp_reshape_dim,
                                       4));
    MLU_CHECK(cnmlCreateReshapeOp(&cont_temp_reshape_op_[i],
                           cont_temp_reshape_param_ptr_[i],
                           cont_temp_blobs_[i]->mlu_tensor(),
                           cont_reshape_blobs_[i]->mlu_tensor()));

    MLU_CHECK(cnmlCreateAddOp(&add_2_op_[i],
                             cont_reshape_blobs_[i]->mlu_tensor(),
                             i_g_temp_blobs_[i]->mlu_tensor(),
                             c_blobs_[i+1]->mlu_tensor()));

    // h_t := o_t .* \tanh[c_t]
    MLU_CHECK(cnmlCreateActiveOp(&tanh_c_op_[i],
                              cnmlActiveFunction_t::CNML_ACTIVE_TANH,
                              c_blobs_[i+1]->mlu_tensor(),
                              tanh_c_temp_blobs_[i]->mlu_tensor()));

    MLU_CHECK(cnmlCreateMultOp(&mult_2_op_[i],
                              o_blobs_[i]->mlu_tensor(),
                              tanh_c_temp_blobs_[i]->mlu_tensor(),
                              h_blobs_[i+1]->mlu_tensor()));
  }

  // top[0] data   Concat
  cnmlTensor_t h_inputs_tensor[this->T_];   // NOLINT
  for (int i = 0; i < this->T_; i++) {
    h_inputs_tensor[i] = h_blobs_[i+1]->mlu_tensor();
  }

  cnmlTensor_t h_outputs_tensor = top0_reshape_blob_->mlu_tensor();
  MLU_CHECK(cnmlCreateNdConcatOp(&h_concat_op_,
                              0,
                              h_inputs_tensor,
                              this->T_,
                              &h_outputs_tensor,
                              1));
    int top0_shape_dim[3];
    top0_shape_dim[0] = top[0]->shape(0);
    top0_shape_dim[1] = top[0]->shape(1);
    top0_shape_dim[2] = top[0]->shape(2);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&top0_reshape_param_ptr_,
                                        top0_shape_dim,
                                        3));

  MLU_CHECK(cnmlCreateReshapeOp(&top0_reshape_op_,
                            top0_reshape_param_ptr_,
                            top0_reshape_blob_->mlu_tensor(),
                            top[0]->mlu_tensor()));


  if (this->expose_hidden_) {
    // top[1]
    MLU_CHECK(cnmlCreateDeviceMemcpyOp(&h_concat_op1_,
                                      h_blobs_[this->T_]->mlu_tensor(),
                                      top1_reshape_blob_->mlu_tensor()));
    int top1_shape_dim[3];
    top1_shape_dim[0] = top[1]->shape(0);
    top1_shape_dim[1] = top[1]->shape(1);
    top1_shape_dim[2] = top[1]->shape(2);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&top1_reshape_param_ptr_,
                                        top1_shape_dim,
                                        3));

    MLU_CHECK(cnmlCreateReshapeOp(&top1_reshape_op_,
                               top1_reshape_param_ptr_,
                               top1_reshape_blob_->mlu_tensor(),
                               top[1]->mlu_tensor()));

    // top[2]
    MLU_CHECK(cnmlCreateDeviceMemcpyOp(&h_concat_op2_,
                                      c_blobs_[this->T_]->mlu_tensor(),
                                      top2_reshape_blob_->mlu_tensor()));

    int top2_shape_dim[3];
    top2_shape_dim[0] = top[2]->shape(0);
    top2_shape_dim[1] = top[2]->shape(1);
    top2_shape_dim[2] = top[2]->shape(2);
    MLU_CHECK(cnmlCreateNdReshapeOpParam(&top2_reshape_param_ptr_,
                                        top2_shape_dim,
                                        3));
    MLU_CHECK(cnmlCreateReshapeOp(&top2_reshape_op_,
                               top2_reshape_param_ptr_,
                               top2_reshape_blob_->mlu_tensor(),
                               top[2]->mlu_tensor()));
  }
}  // MLUCreateOpBindData

template <typename Dtype>
void MLULSTMLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(bottom0_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(bottom1_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->static_input_) {
    MLU_CHECK(cnmlCompileBaseOp(bottom2_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  if (this->expose_hidden_) {
    MLU_CHECK(cnmlCompileBaseOp(bottom3_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(bottom4_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(cont_slice_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(w_xc_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(w_xc_x_slice_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->static_input_) {
    MLU_CHECK(cnmlCompileBaseOp(w_xc_static_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  for (int i = 0; i < this->T_; i++) {
    if (i != 0)
      MLU_CHECK(cnmlCompileBaseOp(h_reshape_op_[i-1],
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(cyclemult_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(h_cont_reshape_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(w_hc_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(add_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    if (this->static_input_)
      MLU_CHECK(cnmlCompileBaseOp(add_1_op_[i],
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(gate_slice_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(sigmoid_i_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(sigmoid_f_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(sigmoid_o_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(tanh_g_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(mult_1_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    if (i == 0 && this->expose_hidden_)
      MLU_CHECK(cnmlCompileBaseOp(c_reshape_op_,
                                  Caffe::rt_core(),
                                  Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(mult_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(f_c_temp_reshape_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(cyclemult_1_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(cont_temp_reshape_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(add_2_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(tanh_c_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(mult_2_op_[i],
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
  MLU_CHECK(cnmlCompileBaseOp(h_concat_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  MLU_CHECK(cnmlCompileBaseOp(top0_reshape_op_,
                              Caffe::rt_core(),
                              Caffe::core_number()));
  if (this->expose_hidden_) {
    MLU_CHECK(cnmlCompileBaseOp(h_concat_op1_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(top1_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(h_concat_op2_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
    MLU_CHECK(cnmlCompileBaseOp(top2_reshape_op_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
}  // MLUCompileOp

template <typename Dtype>
void MLULSTMLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(bottom0_reshape_op_);
  fuser->fuse(bottom1_reshape_op_);
  if (this->static_input_) {
    fuser->fuse(bottom2_reshape_op_);
  }
  if (this->expose_hidden_) {
    fuser->fuse(bottom3_reshape_op_);
    fuser->fuse(bottom4_reshape_op_);
  }

  fuser->fuse(cont_slice_op_);
  fuser->fuse(w_xc_op_);
  fuser->fuse(w_xc_x_slice_op_);
  if (this->static_input_) {
    fuser->fuse(w_xc_static_op_);
  }
  for (int i = 0; i < this->T_; i++) {
    if (i != 0)
      fuser->fuse(h_reshape_op_[i-1]);
    fuser->fuse(cyclemult_op_[i]);
    fuser->fuse(h_cont_reshape_op_[i]);
    fuser->fuse(w_hc_op_[i]);
    fuser->fuse(add_op_[i]);
    if (this->static_input_)
      fuser->fuse(add_1_op_[i]);
    fuser->fuse(gate_slice_op_[i]);
    fuser->fuse(sigmoid_i_op_[i]);
    fuser->fuse(sigmoid_f_op_[i]);
    fuser->fuse(sigmoid_o_op_[i]);
    fuser->fuse(tanh_g_op_[i]);
    fuser->fuse(mult_1_op_[i]);
    if (i == 0 && this->expose_hidden_)
      fuser->fuse(c_reshape_op_);
    fuser->fuse(mult_op_[i]);
    fuser->fuse(f_c_temp_reshape_op_[i]);
    fuser->fuse(cyclemult_1_op_[i]);
    fuser->fuse(cont_temp_reshape_op_[i]);
    fuser->fuse(add_2_op_[i]);
    fuser->fuse(tanh_c_op_[i]);
    fuser->fuse(mult_2_op_[i]);
  }
  fuser->fuse(h_concat_op_);
  fuser->fuse(top0_reshape_op_);
  if (this->expose_hidden_) {
    fuser->fuse(h_concat_op1_);
    fuser->fuse(top1_reshape_op_);
    fuser->fuse(h_concat_op2_);
    fuser->fuse(top2_reshape_op_);
  }
}  // fuse

template <typename Dtype>
void MLULSTMLayer<Dtype>::MLUDestroyOp() {
  if (bottom0_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom0_reshape_op_));
    bottom0_reshape_op_ = nullptr;
  }
  if (bottom0_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom0_reshape_param_ptr_));
     bottom0_reshape_param_ptr_ = nullptr;
  }
  if (bottom1_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom1_reshape_op_));
    bottom1_reshape_op_ = nullptr;
  }
  if (bottom1_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom1_reshape_param_ptr_));
     bottom1_reshape_param_ptr_ = nullptr;
  }
  if (bottom2_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom2_reshape_op_));
    bottom2_reshape_op_ = nullptr;
  }
  if (bottom2_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom2_reshape_param_ptr_));
     bottom2_reshape_param_ptr_ = nullptr;
  }
  if (bottom3_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom3_reshape_op_));
    bottom3_reshape_op_ = nullptr;
  }
  if (bottom3_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom3_reshape_param_ptr_));
     bottom3_reshape_param_ptr_ = nullptr;
  }
  if (bottom4_reshape_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&bottom4_reshape_op_));
    bottom4_reshape_op_ = nullptr;
  }
  if (bottom4_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&bottom4_reshape_param_ptr_));
     bottom4_reshape_param_ptr_ = nullptr;
  }
  if (cont_slice_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&cont_slice_op_));
    cont_slice_op_ = nullptr;
  }
  if (w_xc_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&w_xc_op_));
    w_xc_op_ = nullptr;
  }
  if (w_xc_x_slice_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&w_xc_x_slice_op_));
    w_xc_x_slice_op_ = nullptr;
  }
  if (w_xc_static_op_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&w_xc_static_op_));
    w_xc_static_op_ = nullptr;
  }
  for (int i = 0; i < cyclemult_op_.size(); i++) {
    if (cyclemult_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&cyclemult_op_[i]));
      cyclemult_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_reshape_op_.size(); i++) {
    if (h_reshape_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_reshape_op_[i]));
      h_reshape_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_reshape_param_ptr_.size(); i++) {
    if (h_reshape_param_ptr_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&h_reshape_param_ptr_[i]));
      h_reshape_param_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_cont_reshape_op_.size(); i++) {
    if (h_cont_reshape_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&h_cont_reshape_op_[i]));
      h_cont_reshape_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < h_cont_reshape_param_ptr_.size(); i++) {
    if (h_cont_reshape_param_ptr_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&h_cont_reshape_param_ptr_[i]));
      h_cont_reshape_param_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < w_hc_op_.size(); i++) {
    if (w_hc_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&w_hc_op_[i]));
      w_hc_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < add_op_.size(); i++) {
    if (add_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&add_op_[i]);
      add_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < add_1_op_.size(); i++) {
    if (add_1_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&add_1_op_[i]);
      add_1_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < gate_slice_op_.size(); i++) {
    if (gate_slice_op_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyBaseOp(&gate_slice_op_[i]));
      gate_slice_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < sigmoid_i_op_.size(); i++) {
    if (sigmoid_i_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&sigmoid_i_op_[i]);
      sigmoid_i_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < sigmoid_f_op_.size(); i++) {
    if (sigmoid_f_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&sigmoid_f_op_[i]);
      sigmoid_f_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < sigmoid_o_op_.size(); i++) {
    if (sigmoid_o_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&sigmoid_o_op_[i]);
      sigmoid_o_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < tanh_g_op_.size(); i++) {
    if (tanh_g_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&tanh_g_op_[i]);
      tanh_g_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < mult_1_op_.size(); i++) {
    if (mult_1_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&mult_1_op_[i]);
      mult_1_op_[i] = nullptr;
    }
  }
  if (c_reshape_op_ != nullptr) {
    cnmlDestroyBaseOp(&c_reshape_op_);
    c_reshape_op_ = nullptr;
  }
  if (c_reshape_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&c_reshape_param_ptr_));
    c_reshape_param_ptr_ = nullptr;
  }
  for (int i = 0; i < mult_op_.size(); i++) {
    if (mult_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&mult_op_[i]);
      mult_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < f_c_temp_reshape_op_.size(); i++) {
    if (f_c_temp_reshape_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&f_c_temp_reshape_op_[i]);
      f_c_temp_reshape_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < f_c_temp_reshape_param_ptr_.size(); i++) {
    if (f_c_temp_reshape_param_ptr_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&f_c_temp_reshape_param_ptr_[i]));
      f_c_temp_reshape_param_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < cyclemult_1_op_.size(); i++) {
    if (cyclemult_1_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&cyclemult_1_op_[i]);
      cyclemult_1_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < cont_temp_reshape_op_.size(); i++) {
    if (cont_temp_reshape_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&cont_temp_reshape_op_[i]);
      cont_temp_reshape_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < cont_temp_reshape_param_ptr_.size(); i++) {
    if (cont_temp_reshape_param_ptr_[i] != nullptr) {
      MLU_CHECK(cnmlDestroyReshapeOpParam(&cont_temp_reshape_param_ptr_[i]));
      cont_temp_reshape_param_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < add_2_op_.size(); i++) {
    if (add_2_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&add_2_op_[i]);
      add_2_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < tanh_c_op_.size(); i++) {
    if (tanh_c_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&tanh_c_op_[i]);
      tanh_c_op_[i] = nullptr;
    }
  }
  for (int i = 0; i < mult_2_op_.size(); i++) {
    if (mult_2_op_[i] != nullptr) {
      cnmlDestroyBaseOp(&mult_2_op_[i]);
      mult_2_op_[i] = nullptr;
    }
  }
  if (h_concat_op_ != nullptr) {
    cnmlDestroyBaseOp(&h_concat_op_);
    h_concat_op_ = nullptr;
  }
  if (h_concat_op1_ != nullptr) {
    cnmlDestroyBaseOp(&h_concat_op1_);
    h_concat_op1_ = nullptr;
  }
  if (h_concat_op2_ != nullptr) {
    cnmlDestroyBaseOp(&h_concat_op2_);
    h_concat_op2_ = nullptr;
  }
  if (top0_reshape_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyReshapeOpParam(&top0_reshape_param_ptr_));
    top0_reshape_param_ptr_ = nullptr;
  }
  if (top0_reshape_op_ != nullptr) {
     cnmlDestroyBaseOp(&top0_reshape_op_);
     top0_reshape_op_ = nullptr;
  }
  if (top1_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&top1_reshape_param_ptr_));
     top1_reshape_param_ptr_ = nullptr;
  }
  if (top1_reshape_op_ != nullptr) {
     cnmlDestroyBaseOp(&top1_reshape_op_);
     top1_reshape_op_ = nullptr;
  }
  if (top2_reshape_param_ptr_ != nullptr) {
     MLU_CHECK(cnmlDestroyReshapeOpParam(&top2_reshape_param_ptr_));
     top2_reshape_param_ptr_ = nullptr;
  }
  if (top2_reshape_op_ != nullptr) {
     cnmlDestroyBaseOp(&top2_reshape_op_);
     top2_reshape_op_ = nullptr;
  }
}

template <typename Dtype>
MLULSTMLayer<Dtype>::~MLULSTMLayer() {
  MLUDestroyOp();
}

template <typename Dtype>
void MLULSTMLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // reshape bottom[0]_ to bottom0_reshape_blob_
  // 3, 2, 16 to 6, 16, 1, 1
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom0_reshape_op_,
          bottom[0]->mutable_mlu_data(),
          bottom0_reshape_blob_->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));

  MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom1_reshape_op_,
          bottom[1]->mutable_mlu_data(),
          bottom1_reshape_blob_->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));

  if (this->static_input_) {
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom2_reshape_op_,
            bottom[2]->mutable_mlu_data(),
            bottom2_reshape_blob_->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
  }
  if (this->expose_hidden_) {
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom3_reshape_op_,
            bottom[3]->mutable_mlu_data(),
            bottom3_reshape_blob_->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeReshapeOpForward_V3(bottom4_reshape_op_,
            bottom[4]->mutable_mlu_data(),
            bottom4_reshape_blob_->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
  }


  // Slice cont_{t}
  void* cont_input_ptrs = bottom1_reshape_blob_->mutable_mlu_data();
  void* cont_output_ptrs[this->T_];
  for (int i = 0; i < this->T_; i++) {
    cont_output_ptrs[i] = cont_blobs_[i]->mutable_mlu_data();
  }
  MLU_CHECK(cnmlComputeSplitOpForward_V3(cont_slice_op_,
                  &cont_input_ptrs,
                  1,
                  cont_output_ptrs,
                  this->T_,
                  Caffe::forward_param(), Caffe::queue()));

  //     W_xc_x = W_xc * x + b_c
  MLU_CHECK(cnmlComputeMlpOpForward_V3(w_xc_op_,
        bottom0_reshape_blob_->mutable_mlu_data(),
        w_xc_blob_->mutable_mlu_data(),
        Caffe::forward_param(), Caffe::queue()));

  // Slice w_xc_x_{t}
  void* w_xc_x_input_ptrs = w_xc_blob_->mutable_mlu_data();
  void* w_xc_x_output_ptrs[this->T_];
  for (int i = 0; i < this->T_; i++) {
    w_xc_x_output_ptrs[i] = w_xc_x_blobs_[i]->mutable_mlu_data();
  }
  MLU_CHECK(cnmlComputeSplitOpForward_V3(w_xc_x_slice_op_,
                   &w_xc_x_input_ptrs,
                   1,
                   w_xc_x_output_ptrs,
                   this->T_,
                   Caffe::forward_param(), Caffe::queue()));

  //     W_xc_x_static = W_xc_static * x_static
  if (this->static_input_) {
    MLU_CHECK(cnmlComputeMlpOpForward_V3(w_xc_static_op_,
          bottom2_reshape_blob_->mutable_mlu_data(),
          w_xc_x_static_blob_->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));
  }

  //     h_conted_{t-1} := cont_t * h_{t-1}
  for (int i = 0; i < this->T_; i++) {
    if (i == 0) {
      if (this->expose_hidden_) {
        MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
              bottom4_reshape_blob_->mutable_mlu_data(),
              cont_blobs_[i]->mutable_mlu_data(),
              h_cont_blobs_[i]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
      } else {
        MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
              nullptr,
              cont_blobs_[i]->mutable_mlu_data(),
              h_cont_blobs_[i]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
      }
    } else {
      // reshape h_blobs_ to h_reshape_blobs_
      // 2,18,1,1, to 1,2,18,1
      MLU_CHECK(cnmlComputeReshapeOpForward_V3(h_reshape_op_[i-1],
              h_blobs_[i]->mutable_mlu_data(),
              h_reshape_blobs_[i-1]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
      MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_op_[i],
              h_reshape_blobs_[i-1]->mutable_mlu_data(),
              cont_blobs_[i]->mutable_mlu_data(),
              h_cont_blobs_[i]->mutable_mlu_data(),
              Caffe::forward_param(), Caffe::queue()));
    }

    // reshape h_cont_blobs to h_cont_reshape_blobs_
    // 1,2,18,1 to 2,18,1,1
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(h_cont_reshape_op_[i],
            h_cont_blobs_[i]->mutable_mlu_data(),
            h_cont_reshape_blobs_[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));

    //   W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    MLU_CHECK(cnmlComputeMlpOpForward_V3(w_hc_op_[i],
          h_cont_reshape_blobs_[i]->mutable_mlu_data(),
          w_hc_h_blobs_[i]->mutable_mlu_data(),
          Caffe::forward_param(), Caffe::queue()));

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c

    if (!this->static_input_) {
      MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_[i],
            w_hc_h_blobs_[i]->mutable_mlu_data(),
            w_xc_x_blobs_[i]->mutable_mlu_data(),
            gate_input_blobs_[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
    } else {
      MLU_CHECK(cnmlComputeAddOpForward_V3(add_op_[i],
            w_hc_h_blobs_[i]->mutable_mlu_data(),
            w_xc_x_blobs_[i]->mutable_mlu_data(),
            temp_input_blobs_[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));

      MLU_CHECK(cnmlComputeAddOpForward_V3(add_1_op_[i],
            temp_input_blobs_[i]->mutable_mlu_data(),
            w_xc_x_static_blob_->mutable_mlu_data(),
            gate_input_blobs_[i]->mutable_mlu_data(),
            Caffe::forward_param(), Caffe::queue()));
    }

    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]

    // slice  gate_input_blobs_  to i, f, o, g
    void* gate_input_ptrs = gate_input_blobs_[i]->mutable_mlu_data();
    void* gate_output_ptrs[4];
    for (int j = 0; j < 4; j++) {
      gate_output_ptrs[j] = i_f_o_g_blobs_[j + i * 4]->mutable_mlu_data();
    }
    MLU_CHECK(cnmlComputeSplitOpForward_V3(gate_slice_op_[i],
                       &gate_input_ptrs,
                       1,
                       gate_output_ptrs,
                       4,
                       Caffe::forward_param(), Caffe::queue()));

    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]

    // i_t := \sigmoid[i_t']
    MLU_CHECK(cnmlComputeActiveOpForward_V3(sigmoid_i_op_[i],
                        i_f_o_g_blobs_[i * 4]->mutable_mlu_data(),
                        i_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));

    // f_t := \sigmoid[f_t']
    MLU_CHECK(cnmlComputeActiveOpForward_V3(sigmoid_f_op_[i],
                        i_f_o_g_blobs_[i * 4 + 1]->mutable_mlu_data(),
                        f_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));

    // o_t := \sigmoid[o_t']
    MLU_CHECK(cnmlComputeActiveOpForward_V3(sigmoid_o_op_[i],
                        i_f_o_g_blobs_[i * 4 + 2]->mutable_mlu_data(),
                        o_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));

    // g_t := \tanh[g_t']
    MLU_CHECK(cnmlComputeActiveOpForward_V3(tanh_g_op_[i],
                        i_f_o_g_blobs_[i * 4 + 3]->mutable_mlu_data(),
                        g_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));

    // c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    MLU_CHECK(cnmlComputeMultOpForward_V3(mult_1_op_[i],
                        i_blobs_[i]->mutable_mlu_data(),
                        g_blobs_[i]->mutable_mlu_data(),
                        i_g_temp_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));

    if (i == 0) {
      if (this->expose_hidden_) {
        // bottom[bottom.size()-2] to c_reshape_blob_
        // 1,2,18,1  to 2,18,1,1
        MLU_CHECK(cnmlComputeReshapeOpForward_V3(c_reshape_op_,
                bottom3_reshape_blob_->mutable_mlu_data(),
                c_reshape_blob_->mutable_mlu_data(),
                Caffe::forward_param(), Caffe::queue()));

        MLU_CHECK(cnmlComputeMultOpForward_V3(mult_op_[i],
                        f_blobs_[i]->mutable_mlu_data(),
                        c_reshape_blob_->mutable_mlu_data(),
                        f_c_temp_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));

      } else {
        MLU_CHECK(cnmlComputeMultOpForward_V3(mult_op_[i],
                        f_blobs_[i]->mutable_mlu_data(),
                        // c_blobs_[i]->mutable_mlu_data(),
                        nullptr,
                        f_c_temp_blobs_[i]->mutable_mlu_data(),
                        Caffe::forward_param(), Caffe::queue()));
      }
    } else {
      MLU_CHECK(cnmlComputeMultOpForward_V3(mult_op_[i],
                         f_blobs_[i]->mutable_mlu_data(),
                         c_blobs_[i]->mutable_mlu_data(),
                         f_c_temp_blobs_[i]->mutable_mlu_data(),
                         Caffe::forward_param(), Caffe::queue()));
    }

    // reshape f_c_temp_blobs to f_c_reshape_blobs_
    // 2,18,1,1  to 1,2,18,1
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(f_c_temp_reshape_op_[i],
                            f_c_temp_blobs_[i]->mutable_mlu_data(),
                            f_c_reshape_blobs_[i]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeCycleMultOpForward_V3(cyclemult_1_op_[i],
                            f_c_reshape_blobs_[i]->mutable_mlu_data(),
                            cont_blobs_[i]->mutable_mlu_data(),
                            cont_temp_blobs_[i]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));

    // reshape cont_temp_blobs to cont_reshape_blobs_
    // 1,2,18,1 to 2,18,1,1
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(cont_temp_reshape_op_[i],
                            cont_temp_blobs_[i]->mutable_mlu_data(),
                            cont_reshape_blobs_[i]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeAddOpForward_V3(add_2_op_[i],
                            cont_reshape_blobs_[i]->mutable_mlu_data(),
                            i_g_temp_blobs_[i]->mutable_mlu_data(),
                            c_blobs_[i+1]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));

    // h_t := o_t .* \tanh[c_t]
    MLU_CHECK(cnmlComputeActiveOpForward_V3(tanh_c_op_[i],
                            c_blobs_[i+1]->mutable_mlu_data(),
                            tanh_c_temp_blobs_[i]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeMultOpForward_V3(mult_2_op_[i],
                            o_blobs_[i]->mutable_mlu_data(),
                            tanh_c_temp_blobs_[i]->mutable_mlu_data(),
                            h_blobs_[i+1]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));
  }
  // top[0]
  void* mlutensor_input_ptrs[h_blobs_.size()-1];
  for (int i = 0; i < this->T_; i++) {
    mlutensor_input_ptrs[i] = h_blobs_[i+1]->mutable_mlu_data();
  }
  void* mlutensor_output_ptrs = top0_reshape_blob_->mutable_mlu_data();

  MLU_CHECK(cnmlComputeConcatOpForward_V3(h_concat_op_,
                            mlutensor_input_ptrs,
                            this->T_,
                            &mlutensor_output_ptrs,
                            1,
                            Caffe::forward_param(), Caffe::queue()));
  // reshape top0_reshape_blob_ to top[0]
  // 6,18,1,1 to 3,2,18,1
  MLU_CHECK(cnmlComputeReshapeOpForward_V3(top0_reshape_op_,
                            top0_reshape_blob_->mutable_mlu_data(),
                            top[0]->mutable_mlu_data(),
                            Caffe::forward_param(), Caffe::queue()));


  if (this->expose_hidden_) {
    MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(h_concat_op1_,
        h_blobs_[this->T_]->mutable_mlu_data(),
        top1_reshape_blob_->mutable_mlu_data(),
        Caffe::forward_param(), Caffe::queue()));

    // reshape top1_reshape_blob_ to top[1]
    // 2,18,1,1 to 1,2,18
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(top1_reshape_op_,
        top1_reshape_blob_->mutable_mlu_data(),
        top[1]->mutable_mlu_data(),
        Caffe::forward_param(), Caffe::queue()));

    MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(h_concat_op2_,
        c_blobs_[this->T_]->mutable_mlu_data(),
        top2_reshape_blob_->mutable_mlu_data(),
        Caffe::forward_param(), Caffe::queue()));

    // reshape top2_reshape_blob_ to top[2]
    // 2,18,1,1 to 1,2,18
    MLU_CHECK(cnmlComputeReshapeOpForward_V3(top2_reshape_op_,
        top2_reshape_blob_->mutable_mlu_data(),
        top[2]->mutable_mlu_data(),
        Caffe::forward_param(), Caffe::queue()));
  }
}  // Forward_mlu

INSTANTIATE_CLASS(MLULSTMLayer);
}  // namespace caffe
#endif
