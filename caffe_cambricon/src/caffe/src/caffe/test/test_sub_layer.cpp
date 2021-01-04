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

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_sub_layer.hpp"
#include "caffe/layers/sub_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
float  caffe_sub(const Blob<Dtype>* bottom, const Blob<Dtype>* bottom_sub,
  const Blob<Dtype>* top_blob, float errRate) {
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* bottom_data_sub = bottom_sub->cpu_data();
  const Dtype* top_data = top_blob->cpu_data();
  float err_sum = 0, sum = 0;
  for (int i = 0; i < bottom->count(); i++) {
    Dtype top = bottom_data[i] - bottom_data_sub[i];
    EXPECT_NEAR(top_data[i], top, errRate);
    err_sum += std::abs(top_data[i]-top);
    sum+=std::abs(top);
  }
  EXPECT_LE(err_sum/sum, 5e-4);
  return err_sum/sum;
}

template <typename TypeParam>
class SubLayerTest : public CPUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    SubLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_bottom_sub_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      filler.Fill(this->blob_bottom_);
      filler.Fill(this->blob_bottom_sub_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_sub_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~SubLayerTest() {
      delete blob_bottom_;
      delete blob_bottom_sub_;
      delete blob_top_;
    }
    virtual void TestForward() {
      SetUp();
      LayerParameter layer_param;
      SubLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      caffe_sub(this->blob_bottom_, this->blob_bottom_sub_,
          this->blob_top_, 1e-5);
    }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_sub_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SubLayerTest, TestDtypesAndDevices);

TYPED_TEST(SubLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SubLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(SubLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(SubLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 32, 2, 2);
  this->blob_bottom_sub_->Reshape(1, 32, 2, 2);
  this->TestForward();
}

TYPED_TEST(SubLayerTest, TestForward1Dim) {
  vector<int> shape;
  shape.push_back(51);
  this->blob_bottom_->Reshape(shape);
  this->blob_bottom_sub_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(SubLayerTest, TestForward2Dim) {
  vector<int> shape;
  shape.push_back(51);
  shape.push_back(31);
  this->blob_bottom_->Reshape(shape);
  this->blob_bottom_sub_->Reshape(shape);
  this->TestForward();
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUSubLayerTest : public MLUDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    MLUSubLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_bottom_sub_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()) {}
    void SetUp() {
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      filler.Fill(this->blob_bottom_sub_);
      this->blob_bottom_vec_.clear();
      this->blob_top_vec_.clear();
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_sub_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~MLUSubLayerTest() {
      delete blob_bottom_;
      delete blob_bottom_sub_;
      delete blob_top_;
    }
    virtual void TestForward() {
      SetUp();
      LayerParameter layer_param;
      MLUSubLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      float rate = caffe_sub(this->blob_bottom_, this->blob_bottom_sub_,
                this->blob_top_, 5e-3);
      ERR_RATE(rate);
      std::ostringstream stream;
      stream << "bottom1:" << blob_bottom_->shape_string().c_str() << "\t"
        << "bottom2:" << blob_bottom_sub_->shape_string().c_str();
      BOTTOM(stream);
      EVENT_TIME(layer.get_event_time());
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_bottom_sub_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUSubLayerTest, TestMLUDevices);

TYPED_TEST(MLUSubLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MLUSubLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 8);
  EXPECT_EQ(this->blob_top_->width(), 10);
  OUTPUT("bottom1", this->blob_bottom_->shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_sub_->shape_string().c_str());
}

TYPED_TEST(MLUSubLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MLUSubLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 32, 2, 2);
  this->blob_bottom_sub_->Reshape(1, 32, 2, 2);
  this->TestForward();
}

TYPED_TEST(MLUSubLayerTest, TestForward1Dim) {
  vector<int> shape;
  shape.push_back(51);
  this->blob_bottom_->Reshape(shape);
  this->blob_bottom_sub_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(MLUSubLayerTest, TestForward2Dim) {
  vector<int> shape;
  shape.push_back(51);
  shape.push_back(31);
  this->blob_bottom_->Reshape(shape);
  this->blob_bottom_sub_->Reshape(shape);
  this->TestForward();
}


template <typename TypeParam>
class MFUSSubLayerTest : public MFUSDeviceTest<TypeParam> {
typedef typename TypeParam::Dtype Dtype;

  protected:
    MFUSSubLayerTest()
       : blob_bottom_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_bottom_sub_(new Blob<Dtype>(3, 5, 8, 10)),
         blob_top_(new Blob<Dtype>()) {}
  void SetUp() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_sub_);
    this->blob_bottom_vec_.clear();
    this->blob_top_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_sub_);
    this->blob_top_vec_.push_back(this->blob_top_);
  }
  virtual ~MFUSSubLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_sub_;
    delete blob_top_;
  }
  virtual void TestForward() {
    LayerParameter layer_param;
    MLUSubLayer<Dtype> layer(layer_param);
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
    float rate = caffe_sub(this->blob_bottom_, this->blob_bottom_sub_,
              this->blob_top_, 5e-3);
    ERR_RATE(rate);
    EVENT_TIME(fuser.get_event_time());
    std::ostringstream stream;
    stream << "bottom1:" << blob_bottom_->shape_string().c_str() << "\t"
           << "bottom1:" << blob_bottom_sub_->shape_string().c_str();
    BOTTOM(stream);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_sub_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSSubLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSSubLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MFUSSubLayerTest, TestForward1Batch) {
  this->blob_bottom_->Reshape(1, 32, 2, 2);
  this->blob_bottom_sub_->Reshape(1, 32, 2, 2);
  this->TestForward();
}

TYPED_TEST(MFUSSubLayerTest, TestForward1Dim) {
  vector<int> shape;
  shape.push_back(51);
  this->blob_bottom_->Reshape(shape);
  this->blob_bottom_sub_->Reshape(shape);
  this->TestForward();
}

TYPED_TEST(MFUSSubLayerTest, TestForward2Dim) {
  vector<int> shape;
  shape.push_back(51);
  shape.push_back(31);
  this->blob_bottom_->Reshape(shape);
  this->blob_bottom_sub_->Reshape(shape);
  this->TestForward();
}

#endif

}  // namespace caffe
