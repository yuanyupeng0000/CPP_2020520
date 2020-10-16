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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  use_pad_same_ = conv_param.same_mode();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
          conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0)
                           ? kDefaultStride
                           : conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_i).
  vector<int> pad_shape(std::max(num_spatial_axes_ * 2, 1), 1);
  pad_shape[0] = std::max(num_spatial_axes_ * 2, 1);
  pad_.Reshape(pad_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
    pad_data[2] = conv_param.pad_h();
    pad_data[3] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_ ||
          num_pad_dims == num_spatial_axes_ *2)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; " << num_spatial_axes_
        << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < pad_shape.size(); ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad
                        : conv_param.pad((num_pad_dims == 1) ? 0 : i % num_pad_dims);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] =
        (num_dilation_dims == 0)
            ? kDefaultDilation
            : conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    // no padding means pad_data should be 0, eg num_spatial_axes_ = 2, check
    // height(pad_data[0], pad_data[2]) and width (pad_data[1], pad_data[3]) be 0
    is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1 &&
        pad_data[i] == 0 && (conv_param.pad_size() == num_spatial_axes_ *2 ?
        pad_data[i + num_spatial_axes_] == 0 : 1);
    if (!is_1x1_) {
      break;
    }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  if (this->layer_param_.type() == "ConvolutionDepthwise") {
    group_ = bottom[0]->channels();
  }
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
                 << weight_shaped_blob.shape_string() << "; instead, shape was "
                 << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
                 << bias_shaped_blob.shape_string() << "; instead, shape was "
                 << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

#ifdef USE_MLU
  SetMean(bottom, top);
  SetStd(bottom, top);
#endif
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::SetupColBuf() {
  // Below code is moved here from LayerSetUp. On MLU, the count of
  // col_buffer_ increases rapidly when it is depthwise. For 1080p input,it
  // exceeds blob count limit of 2G.

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
}

#ifdef USE_MLU
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::SetStd(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter convolution_param = this->layer_param().convolution_param();
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.has_top_mlu_dtype() ?
               this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  bool scale = this->layer_param().convolution_param().has_scale();
  bool has_std = convolution_param.std_size();
  if (!((has_std || scale) || this->conv_first_)) {
    return;
  }
  vector<int> std_shape(4, 1);
  std_shape[1] = bottom[0]->channels();
  this->std_.Reshape(std_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  caffe_set(this->std_.count(), Dtype(1), this->std_.mutable_cpu_data());
  if (convolution_param.std_size() >= 1) {
    this->conv_first_ = true;
    // std value shape is 1,c,1,1 according to cnml
    Dtype* data = this->std_.mutable_cpu_data();
    if (convolution_param.std_size() == 1) {
      for (int i = 0; i < bottom[0]->channels(); i++) {
        CHECK_NE(this->layer_param().convolution_param().std(0), 0)
            << "std should not equal zero.";
        data[i] = 1. / this->layer_param().convolution_param().std(0);
      }
    } else if (convolution_param.std_size() > 1) {
      for (int i = 0; i < bottom[0]->channels(); i++) {
        if (i < convolution_param.std_size()) {
          CHECK_NE(this->layer_param().convolution_param().std(i), 0)
              << "std should not equal zero.";
          data[i] = 1. / this->layer_param().convolution_param().std(i);
        } else {
          data[i] = 1.;
        }
      }
    }
  } else if (scale) {
    this->conv_first_ = true;
    Dtype* data = this->std_.mutable_cpu_data();
    for (int i = 0; i < bottom[0]->channels(); i++) {
      CHECK_NE(this->layer_param().convolution_param().scale(), 0)
          << "std should not equal zero.";
      data[i] = 1. / this->layer_param().convolution_param().scale();
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::SetMean(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter convolution_param = this->layer_param().convolution_param();

  this->conv_first_ = true;
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = this->layer_param_.has_top_mlu_dtype() ?
               this->layer_param_.top_mlu_dtype(): DT_FLOAT16;
  vector<int> mean_shape(4, 1);
  mean_shape[1] = bottom[0]->channels();
  this->mean_.Reshape(mean_shape, cpu_dtype, mlu_dtype, CNML_CONST);
  Dtype* data = this->mean_.mutable_cpu_data();
  caffe_set(this->mean_.count(), Dtype(0), data);
  if (!(convolution_param.has_mean_file() ||
        convolution_param.mean_value_size())) {
    conv_first_ = false;
    return;
  }
  if (convolution_param.has_mean_file()) {
    string mean_file = convolution_param.mean_file();
    BlobProto blob_proto;
    Blob<Dtype> mean_file_blob;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    mean_file_blob.FromProto(blob_proto);
    vector<Dtype> mean_vals(bottom[0]->channels(), 0.);
    /**
    * mean file value are (B, G, R)
    * mean_vals value are (R, G, B, A)
    * read from mean file and convert to (R, G, B, A)
    */
    if (bottom[0]->channels() == 4) {
      for (int i = 0; i < mean_file_blob.channels(); i++) {
        for (int j = 0; j < mean_file_blob.height() * mean_file_blob.width(); j++) {
          mean_vals[i] += mean_file_blob.cpu_data()[(mean_file_blob.channels() - i - 1) *
            mean_file_blob.height() * mean_file_blob.width() + j];
        }
        mean_vals[i] /= (mean_file_blob.height() * mean_file_blob.width());
      }
    } else {
      for (int i = 0; i < mean_file_blob.channels(); i++) {
        for (int j = 0; j < mean_file_blob.height() * mean_file_blob.width(); j++) {
          mean_vals[i] += mean_file_blob.cpu_data()[i * mean_file_blob.height() *
                mean_file_blob.width() + j];
        }
        mean_vals[i] /= (mean_file_blob.height() * mean_file_blob.width());
      }
    }
    caffe_copy(bottom[0]->channels(), mean_vals.data(), data);
  } else if (convolution_param.mean_value_size()) {
    if (convolution_param.mean_value_size() > 0) {
      for (int i = 0; i < bottom[0]->channels(); i++) {
        if (i < convolution_param.mean_value_size()) {
          data[i] = this->layer_param().convolution_param().mean_value(i);
        } else {
          data[i] = 0.;
        }
      }
    }
  }
}
#endif

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
              bias_multiplier_.mutable_cpu_data());
  }
  // if use tf_pad, override the pad or pad_h/pad_w params
  if (use_pad_same_) {
    int* pad_data = pad_.mutable_cpu_data();
    int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
    int* stride_data = stride_.mutable_cpu_data();
    int pad_all_height = (top[0]->height() - 1)*stride_data[0]
        + kernel_shape_data[0] - bottom[0]->height();
    int pad_top = static_cast<int>(std::floor(pad_all_height / 2.0));
    int pad_bottom = pad_all_height - pad_top;
    int pad_all_width = (top[0]->width() - 1)*stride_data[1]
        + kernel_shape_data[1] - bottom[0]->width();
    int pad_left = static_cast<int>(std::floor(pad_all_width / 2.0));
    int pad_right = pad_all_width - pad_left;
    pad_data[0] = pad_top;
    pad_data[1] = pad_left;
    pad_data[2] = pad_bottom;
    pad_data[3] = pad_right;
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
                                                   const Dtype* weights,
                                                   Dtype* output,
                                                   bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          conv_out_channels_ / group_, conv_out_spatial_dim_,
                          kernel_dim_, (Dtype)1., weights + weight_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype)0.,
                          output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
                                                   const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype)1., bias,
                        bias_multiplier_.cpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
                                                    const Dtype* weights,
                                                    Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_,
        conv_out_channels_ / group_, (Dtype)1., weights + weight_offset_ * g,
        output + output_offset_ * g, (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
                                                  const Dtype* output,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_,
        conv_out_spatial_dim_, (Dtype)1., output + output_offset_ * g,
        col_buff + col_offset_ * g, (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
                                                    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input,
                        bias_multiplier_.cpu_data(), 1., bias);
}

#ifdef USE_CUDA

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
                                                   const Dtype* weights,
                                                   Dtype* output,
                                                   bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          conv_out_channels_ / group_, conv_out_spatial_dim_,
                          kernel_dim_, (Dtype)1., weights + weight_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype)0.,
                          output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
                                                   const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype)1., bias,
                        bias_multiplier_.gpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
                                                    const Dtype* weights,
                                                    Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_,
        conv_out_channels_ / group_, (Dtype)1., weights + weight_offset_ * g,
        output + output_offset_ * g, (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
                                                  const Dtype* output,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_,
        conv_out_spatial_dim_, (Dtype)1., output + output_offset_ * g,
        col_buff + col_offset_ * g, (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
                                                    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input,
                        bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
