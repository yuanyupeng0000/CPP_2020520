/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_tanh_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

double tanh_naive(double x) {
  if (x < -40) {
    // avoid negative overflow
    return -1;
  } else if (x > 40) {
    // avoid positive overflow
    return 1;
  } else {
    // exact expression for tanh, which is unstable for large x
    double exp2x = exp(2 * x);
    return (exp2x - 1.0) / (exp2x + 1.0);
  }
}

template <typename TypeParam>
class TanHLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  TanHLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TanHLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward(Dtype filler_std) {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    TanHLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = tanh_naive(bottom_data[i]);
      Dtype precision = std::max(Dtype(std::abs(expected_value * Dtype(1e-4))),
                                 min_precision);
      EXPECT_NEAR(expected_value, top_data[i], precision);
    }
  }

  void TestBackward(Dtype filler_std) {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    TanHLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
                                 this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TanHLayerTest, TestDtypesAndDevices);

TYPED_TEST(TanHLayerTest, TestTanH) { this->TestForward(1.0); }

TYPED_TEST(TanHLayerTest, TestTanHOverflow) {
  // this will fail if tanh overflow is not properly handled
  this->TestForward(10000.0);
}

TYPED_TEST(TanHLayerTest, TestTanHGradient) { this->TestBackward(1.0); }

#ifdef USE_MLU
template <typename TypeParam>
class MLUTanHLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUTanHLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUTanHLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward(Dtype filler_std) {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    MLUTanHLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-3;
    float err_sum = 0, sum = 0;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = tanh_naive(bottom_data[i]);
      Dtype precision = std::max(Dtype(std::abs(expected_value * Dtype(1e-2))),
                                 min_precision);
      EXPECT_NEAR(expected_value, top_data[i], precision);
      err_sum += std::abs(top_data[i] - expected_value);
      sum += std::abs(expected_value);
    }
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(err_sum/sum);
    EVENT_TIME(layer.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
TYPED_TEST_CASE(MLUTanHLayerTest, TestMLUDevices);

TYPED_TEST(MLUTanHLayerTest, TestTanH) { this->TestForward(1.0); }

TYPED_TEST(MLUTanHLayerTest, TestTanHOverflow) {
  // this will fail if tanh overflow is not properly handled
  this->TestForward(10000.0);
}

template <typename TypeParam>
class MFUSTanHLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSTanHLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSTanHLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward(Dtype filler_std) {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    MLUTanHLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_TRUE(layer.mfus_supported());

    MFusion<Dtype> fuser;
    fuser.reset();
    fuser.addInputs(this->blob_bottom_vec_);
    fuser.addOutputs(this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.fuse(&fuser);
    fuser.compile();
    fuser.forward();
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-3;
    float err_sum = 0, sum = 0;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = tanh_naive(bottom_data[i]);
      Dtype precision = std::max(Dtype(std::abs(expected_value * Dtype(1e-2))),
                                 min_precision);
      EXPECT_NEAR(expected_value, top_data[i], precision);
      err_sum += std::abs(top_data[i] - expected_value);
      sum += expected_value;
    }
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    ERR_RATE(err_sum/sum);
    EVENT_TIME(fuser.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
TYPED_TEST_CASE(MFUSTanHLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSTanHLayerTest, TestTanH) { this->TestForward(1.0); }

TYPED_TEST(MFUSTanHLayerTest, TestTanHOverflow) {
  // this will fail if tanh overflow is not properly handled
  this->TestForward(10000.0);
}
#endif

}  // namespace caffe
