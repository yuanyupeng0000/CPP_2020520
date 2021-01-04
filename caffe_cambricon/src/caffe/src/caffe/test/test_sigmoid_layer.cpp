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

#include <cmath>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_sigmoid_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid_alt(Dtype x) {
  return (1 / (1 + exp(-x)));
}

template <typename TypeParam>
class SigmoidLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  SigmoidLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SigmoidLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();

    LayerParameter layer_param;
    SigmoidLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); i++) {
      const Dtype top_expected = sigmoid_alt(bottom_data[i]);
      EXPECT_NEAR(top_data[i], top_expected, 5e-5);
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SigmoidLayerTest, TestDtypesAndDevices);

TYPED_TEST(SigmoidLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SigmoidLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(SigmoidLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(9, 23, 7, 8);
  this->TestForward();
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUSigmoidLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUSigmoidLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUSigmoidLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();

    LayerParameter layer_param;
    MLUSigmoidLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // check value
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    float err_sum = 0, sum = 0;
    for (int i = 0; i < blob_bottom_->count(); i++) {
      const Dtype top_expected = sigmoid_alt(bottom_data[i]);
      EXPECT_NEAR(top_data[i], top_expected, 20e-3);
      err_sum += std::abs(top_data[i] - top_expected);
      sum += std::abs(top_data[i]);
    }
    EXPECT_LE(err_sum / sum, 3e-3);
    ERR_RATE(err_sum / sum);
    EVENT_TIME(layer.get_event_time());
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    EVENT_TIME(layer.get_event_time());
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUSigmoidLayerTest, TestMLUDevices);

TYPED_TEST(MLUSigmoidLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUSigmoidLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MLUSigmoidLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(MLUSigmoidLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(7, 15, 11, 13);
  this->TestForward();
}

template <typename TypeParam>
class MFUSSigmoidLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSSigmoidLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 7, 11)),
        blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSSigmoidLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void TestForward() {
    SetUp();

    LayerParameter layer_param;
    MLUSigmoidLayer<Dtype> layer(layer_param);
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

    // check value
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    float err_sum = 0, sum = 0;
    for (int i = 0; i < blob_bottom_->count(); i++) {
      const Dtype top_expected = sigmoid_alt(bottom_data[i]);
      EXPECT_NEAR(top_data[i], top_expected, 20e-3);
      err_sum += std::abs(top_data[i] - top_expected);
      sum += std::abs(top_data[i]);
    }
    EXPECT_LE(err_sum / sum, 3e-3);
    ERR_RATE(err_sum / sum);
    EVENT_TIME(layer.get_event_time());
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str();
    BOTTOM(stream);
    EVENT_TIME(fuser.get_event_time());
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSSigmoidLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSSigmoidLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUSigmoidLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
}

TYPED_TEST(MFUSSigmoidLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(MFUSSigmoidLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(7, 15, 11, 13);
  this->TestForward();
}
#endif

}  // namespace caffe
