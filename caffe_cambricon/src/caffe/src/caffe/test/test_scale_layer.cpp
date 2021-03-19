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

#include <algorithm>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_scale_layer.hpp"
#include "caffe/layers/scale_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ScaleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  ScaleLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_eltwise_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_broadcast_0_(new Blob<Dtype>()),
        blob_bottom_broadcast_1_(new Blob<Dtype>()),
        blob_bottom_broadcast_2_(new Blob<Dtype>()),
        blob_bottom_scale_(new Blob<Dtype>(vector<int>())),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    vector<int> broadcast_shape(2);
    broadcast_shape[0] = 2;
    broadcast_shape[1] = 3;
    this->blob_bottom_broadcast_0_->Reshape(broadcast_shape);
    broadcast_shape[0] = 3;
    broadcast_shape[1] = 4;
    this->blob_bottom_broadcast_1_->Reshape(broadcast_shape);
    broadcast_shape[0] = 4;
    broadcast_shape[1] = 5;
    this->blob_bottom_broadcast_2_->Reshape(broadcast_shape);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_eltwise_);
    filler.Fill(this->blob_bottom_broadcast_0_);
    filler.Fill(this->blob_bottom_broadcast_1_);
    filler.Fill(this->blob_bottom_broadcast_2_);
    filler.Fill(this->blob_bottom_scale_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ScaleLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_eltwise_;
    delete blob_bottom_broadcast_0_;
    delete blob_bottom_broadcast_1_;
    delete blob_bottom_broadcast_2_;
    delete blob_bottom_scale_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_eltwise_;
  Blob<Dtype>* const blob_bottom_broadcast_0_;
  Blob<Dtype>* const blob_bottom_broadcast_1_;
  Blob<Dtype>* const blob_bottom_broadcast_2_;
  Blob<Dtype>* const blob_bottom_scale_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScaleLayerTest, TestDtypesAndDevices);

TYPED_TEST(ScaleLayerTest, TestForwardEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_eltwise_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardEltwiseInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_bottom_->cpu_data();
  const int count = this->blob_bottom_->count();
  const Dtype* in_data_a = orig_bottom.cpu_data();
  const Dtype* in_data_b = this->blob_bottom_eltwise_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestBackwardEltwiseInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  Blob<Dtype> top_diff(this->blob_bottom_->shape());
  FillerParameter filler_param;
  filler_param.set_type("gaussian");
  filler_param.set_std(1);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&top_diff);
  vector<bool> propagate_down(2, true);
  // Run forward + backward without in-place computation;
  // save resulting bottom diffs.
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_top_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const bool kReshape = true;
  const bool kCopyDiff = true;
  Blob<Dtype> orig_bottom_diff;
  orig_bottom_diff.CopyFrom(*this->blob_bottom_, kCopyDiff, kReshape);
  Blob<Dtype> orig_scale_diff;
  orig_scale_diff.CopyFrom(*this->blob_bottom_eltwise_, kCopyDiff, kReshape);
  // Rerun forward + backward with in-place computation;
  // check that resulting bottom diffs are the same.
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_bottom_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(orig_bottom_diff.cpu_diff()[i],
                this->blob_bottom_->cpu_diff()[i], 1e-5);
  }
  for (int i = 0; i < this->blob_bottom_eltwise_->count(); ++i) {
    EXPECT_NEAR(orig_scale_diff.cpu_diff()[i],
                this->blob_bottom_eltwise_->cpu_diff()[i], 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardEltwiseWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScaleParameter* scale_param = layer_param.mutable_scale_param();
  scale_param->set_axis(0);
  scale_param->set_num_axes(-1);
  scale_param->mutable_filler()->set_type("gaussian");
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardBroadcastBegin) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_0_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                          this->blob_bottom_broadcast_0_->data_at(n, c, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardBroadcastMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(1);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                          this->blob_bottom_broadcast_1_->data_at(c, h, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardBroadcastMiddleInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(1);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_bottom_->data_at(n, c, h, w),
                      orig_bottom.data_at(n, c, h, w) *
                          this->blob_bottom_broadcast_1_->data_at(c, h, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleLayerTest, TestBackwardBroadcastMiddleInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> orig_bottom(this->blob_bottom_->shape());
  orig_bottom.CopyFrom(*this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(1);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  Blob<Dtype> top_diff(this->blob_bottom_->shape());
  FillerParameter filler_param;
  filler_param.set_type("gaussian");
  filler_param.set_std(1);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(&top_diff);
  vector<bool> propagate_down(2, true);
  // Run forward + backward without in-place computation;
  // save resulting bottom diffs.
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_top_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const bool kReshape = true;
  const bool kCopyDiff = true;
  Blob<Dtype> orig_bottom_diff;
  orig_bottom_diff.CopyFrom(*this->blob_bottom_, kCopyDiff, kReshape);
  Blob<Dtype> orig_scale_diff;
  orig_scale_diff.CopyFrom(*this->blob_bottom_broadcast_1_, kCopyDiff,
                           kReshape);
  // Rerun forward + backward with in-place computation;
  // check that resulting bottom diffs are the same.
  this->blob_top_vec_[0] = this->blob_bottom_;  // in-place computation
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_copy(top_diff.count(), top_diff.cpu_data(),
             this->blob_bottom_->mutable_cpu_diff());
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(orig_bottom_diff.cpu_diff()[i],
                this->blob_bottom_->cpu_diff()[i], 1e-5);
  }
  for (int i = 0; i < this->blob_bottom_broadcast_1_->count(); ++i) {
    EXPECT_NEAR(orig_scale_diff.cpu_diff()[i],
                this->blob_bottom_broadcast_1_->cpu_diff()[i], 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardBroadcastMiddleWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScaleParameter* scale_param = layer_param.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(2);
  scale_param->mutable_filler()->set_type("gaussian");
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                          layer->blobs()[0]->data_at(c, h, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardBroadcastMiddleWithParamAndBias) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScaleParameter* scale_param = layer_param.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(2);
  scale_param->mutable_filler()->set_type("gaussian");
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                              layer->blobs()[0]->data_at(c, h, 0, 0) +
                          layer->blobs()[1]->data_at(c, h, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardBroadcastEnd) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_2_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(2);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          EXPECT_NEAR(this->blob_top_->data_at(n, c, h, w),
                      this->blob_bottom_->data_at(n, c, h, w) *
                          this->blob_bottom_broadcast_2_->data_at(h, w, 0, 0),
                      1e-5);
        }
      }
    }
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardScale) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  const Dtype scale = *this->blob_bottom_scale_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data[i] * scale, 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestForwardScaleAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(2);
  shared_ptr<ScaleLayer<Dtype>> layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  const Dtype scale = *this->blob_bottom_scale_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data[i] * scale, 1e-5);
  }
}

TYPED_TEST(ScaleLayerTest, TestGradientEltwise) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                               this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientEltwiseWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScaleParameter* scale_param = layer_param.mutable_scale_param();
  scale_param->set_axis(0);
  scale_param->set_num_axes(-1);
  scale_param->mutable_filler()->set_type("gaussian");
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientBroadcastBegin) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_0_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientBroadcastMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(1);
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientBroadcastMiddleWithParam) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_1_);
  LayerParameter layer_param;
  ScaleParameter* scale_param = layer_param.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(2);
  scale_param->mutable_filler()->set_type("gaussian");
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientBroadcastEnd) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_broadcast_2_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(2);
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientScale) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientScaleAndBias) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  ScaleParameter* scale_param = layer_param.mutable_scale_param();
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

TYPED_TEST(ScaleLayerTest, TestGradientScaleAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scale_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(2);
  ScaleLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_);
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUScaleLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUScaleLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 32, 2, 2)),
        blob_bottom1_(new Blob<Dtype>(2, 32, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUScaleLayerTest() {
    delete blob_bottom_;
    delete blob_bottom1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUScaleLayerTest, TestMLUDevices);

TYPED_TEST(MLUScaleLayerTest, TestForwardAlpah1Batch) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(-1);
  scale_param->mutable_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(1, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 1);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 3e-2);
    err_sum += std::abs(data[i] - in_data_a[i] * in_data_b[i]);
    sum += std::abs(data[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer->get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MLUScaleLayerTest, TestForwardAlpah1BatchOpt) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(1);
  scale_param->mutable_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(1, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 0);  //  0 means no reshape
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[(i % 128) / 4], 3e-2);
    err_sum += std::abs(data[i] - in_data_a[i] * in_data_b[(i % 128) / 4]);
    sum += std::abs(data[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer->get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MLUScaleLayerTest, TestForwardAlpahBeta1BatchOpt) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(1);
  scale_param->mutable_filler()->set_type("gaussian");
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(1, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 0);  //  0 means no reshape
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = layer->blobs()[0]->cpu_data();
  const Dtype* in_data_b = layer->blobs()[1]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    const int j = (i % 128) / 4;
    const Dtype res = in_data_s[i] * in_data_a[j] + in_data_b[j];
    EXPECT_NEAR(data[i], res, 3e-2);
    err_sum += std::abs(data[i] - res);
    sum += std::abs(data[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer->get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MLUScaleLayerTest, TestForwardAlphaBeta2BatchOpt) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(1);
  scale_param->mutable_filler()->set_type("gaussian");
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(2, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 0);  //  0 means no reshape
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = layer->blobs()[0]->cpu_data();
  const Dtype* in_data_b = layer->blobs()[1]->cpu_data();
  for (int i = 0; i < count; ++i) {
    const int j = (i % 128) / 4;
    const Dtype res = in_data_s[i] * in_data_a[j] + in_data_b[j];
    EXPECT_NEAR(data[i], res, 3e-2);
  }
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    const int j = (i % 128) / 4;
    const Dtype res = in_data_s[i] * in_data_a[j] + in_data_b[j];
    EXPECT_NEAR(data[i], res, 3e-2);
    err_sum += std::abs(data[i] - res);
    sum += std::abs(data[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer->get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MLUScaleLayerTest, TestForwardAlpha2Input) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(0);
  //  scale_param->set_num_axes(-1);
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(2, 32, 2 , 2);

  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom1_);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 1);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom1_->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_s[i] * in_data_a[i], 3e-2);
    err_sum += std::abs(data[i] - in_data_s[i] * in_data_a[i]);
    sum += std::abs(data[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer->get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer->get_event_time());
}

TYPED_TEST(MLUScaleLayerTest, TestForwardAlphaBeta2Input) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(0);
  //  scale_param->set_num_axes(-1);
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(2, 32, 2, 2);

  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom1_);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 1);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom1_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_s[i] * in_data_a[i] + in_data_b[i], 3e-2);
    err_sum += std::abs(data[i] - in_data_s[i] * in_data_a[i] + in_data_b[i]);
    sum += std::abs(in_data_s[i] * in_data_a[i] + in_data_b[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer->get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(layer->get_event_time());
}

template <typename TypeParam>
class MFUSScaleLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSScaleLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 32, 2, 2)),
        blob_bottom1_(new Blob<Dtype>(2, 32, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSScaleLayerTest() {
    delete blob_bottom_;
    delete blob_bottom1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSScaleLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSScaleLayerTest, TestForwardAlpah1Batch) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(-1);
  scale_param->mutable_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(1, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_TRUE(layer->mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 1);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 3e-2);
    err_sum += std::abs(data[i] - in_data_a[i] * in_data_b[i]);
    sum += std::abs(data[i]);
  }
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSScaleLayerTest, TestForwardAlpah1BatchOpt) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(1);
  scale_param->mutable_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(1, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_TRUE(layer->mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 0);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[(i % 128) / 4], 3e-2);
    err_sum += std::abs(data[i] - in_data_a[i] * in_data_b[(i % 128) / 4]);
    sum += std::abs(data[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
}

TYPED_TEST(MFUSScaleLayerTest, TestForwardAlpahBeta1BatchOpt) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(1);
  scale_param->mutable_filler()->set_type("gaussian");
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(1, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_TRUE(layer->mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 0);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = layer->blobs()[0]->cpu_data();
  const Dtype* in_data_b = layer->blobs()[1]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    const int j = (i % 128) / 4;
    const Dtype res = in_data_s[i] * in_data_a[j] + in_data_b[j];
    EXPECT_NEAR(data[i], res, 3e-2);
    err_sum += std::abs(data[i] - res);
    sum += std::abs(data[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
}

TYPED_TEST(MFUSScaleLayerTest, TestForwardAlphaBeta2BatchOpt) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(1);
  scale_param->set_num_axes(1);
  scale_param->mutable_filler()->set_type("gaussian");
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(2, 32, 2, 2);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_TRUE(layer->mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 0);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = layer->blobs()[0]->cpu_data();
  const Dtype* in_data_b = layer->blobs()[1]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    const int j = (i % 128) / 4;
    const Dtype res = in_data_s[i] * in_data_a[j] + in_data_b[j];
    EXPECT_NEAR(data[i], res, 3e-2);
    err_sum += std::abs(data[i] - res);
    sum += std::abs(data[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
}

TYPED_TEST(MFUSScaleLayerTest, TestForwardAlpha2Input) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(0);
  //  scale_param->set_num_axes(-1);
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(2, 32, 2 , 2);

  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom1_);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_TRUE(layer->mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 1);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom1_->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_s[i] * in_data_a[i], 3e-2);
    err_sum += std::abs(data[i] - in_data_s[i] * in_data_a[i]);
    sum += std::abs(data[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
}

TYPED_TEST(MFUSScaleLayerTest, TestForwardAlphaBeta2Input) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layerparam;
  ScaleParameter* scale_param = layerparam.mutable_scale_param();
  scale_param->set_axis(0);
  //  scale_param->set_num_axes(-1);
  scale_param->set_bias_term(true);
  scale_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<MLUScaleLayer<Dtype>> layer(new MLUScaleLayer<Dtype>(layerparam));

  this->blob_bottom_->Reshape(2, 32, 2, 2);

  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom1_);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(this->blob_bottom1_);

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  ASSERT_TRUE(layer->mfus_supported());

  MFusion<Dtype> fuser;
  fuser.reset();
  fuser.addInputs(this->blob_bottom_vec_);
  fuser.addOutputs(this->blob_top_vec_);
  layer->Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(layer->need_reshape(), 1);
  layer->fuse(&fuser);
  fuser.compile();
  fuser.forward();

  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_s = this->blob_bottom_->cpu_data();
  const Dtype* in_data_a = this->blob_bottom1_->cpu_data();
  const Dtype* in_data_b = layer->blobs()[0]->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_s[i] * in_data_a[i] + in_data_b[i], 3e-2);
    err_sum += std::abs(data[i] - in_data_s[i] * in_data_a[i] + in_data_b[i]);
    sum += std::abs(in_data_s[i] * in_data_a[i] + in_data_b[i]);
  }
  EVENT_TIME(fuser.get_event_time());
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str() << "\t"
    << "bottom2:" << this->blob_bottom1_->shape_string().c_str();
  BOTTOM(stream);
}
#endif

}  // namespace caffe