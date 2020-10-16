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

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lstm_reshape_layer.hpp"
#include "caffe/layers/mlu_lstm_reshape_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LstmReshapeTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  LstmReshapeTest()
      : blob_bottom_(new Blob<Dtype>(1, 50, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~LstmReshapeTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LstmReshapeTest, TestDtypesAndDevices);

TYPED_TEST(LstmReshapeTest, TestOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(1);
  blob_shape->add_dim(1000);

  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1000);
}

TYPED_TEST(LstmReshapeTest, TestValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int c = 0; c < 50 * 4 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
              this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5));
  }
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(LstmReshapeTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 50);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(LstmReshapeTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);
  blob_shape->add_dim(5);
  blob_shape->add_dim(0);
  blob_shape->add_dim(50);

  // Count is 1000, thus height should be 1000 / (5*4*50) = 1.

  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 50);
}

TYPED_TEST(LstmReshapeTest, TestInferenceOfUnspecifiedWithStartAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lstm_reshape_param()->set_axis(1);
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(4);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 4);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 25);
}

TYPED_TEST(LstmReshapeTest, TestFlattenMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lstm_reshape_param()->set_axis(1);
  layer_param.mutable_lstm_reshape_param()->set_num_axes(2);
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);

  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 50 * 4);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(LstmReshapeTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_lstm_reshape_param()->mutable_shape();
  shape->add_dim(2);
  shape->add_dim(25);
  shape->add_dim(5);
  shape->add_dim(4);
  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
}

TYPED_TEST(LstmReshapeTest, TestForwardAfterReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_lstm_reshape_param()->mutable_shape();
  shape->add_dim(2);
  shape->add_dim(25);
  shape->add_dim(5);
  shape->add_dim(4);
  LstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We know the above produced the correct result from TestForward.
  // Reshape the bottom and call layer.Reshape, then try again.
  vector<int> new_bottom_shape(1, 2 * 25 * 5 * 4);
  this->blob_bottom_->Reshape(new_bottom_shape);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
  }
}

#ifdef USE_MLU

template <typename TypeParam>
class MLULstmReshapeTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLULstmReshapeTest()
      : blob_bottom_(new Blob<Dtype>(1, 50, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLULstmReshapeTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLULstmReshapeTest, TestMLUDevices);

TYPED_TEST(MLULstmReshapeTest, TestOutputSizes) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(1);
  blob_shape->add_dim(1000);

  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->count(), this->blob_bottom_->count());
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1000);
}

TYPED_TEST(MLULstmReshapeTest, TestValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int c = 0; c < 50 * 4 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
              this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5));
    err_sum += std::abs(this->blob_top_->data_at(0, c, 0, 0) -
              this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5));
    sum += this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

// Test whether setting output dimensions to 0 either explicitly or implicitly
// copies the respective dimension of the input layer.
TYPED_TEST(MLULstmReshapeTest, TestCopyDimensions) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  blob_shape->add_dim(0);
  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 50);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

// When a dimension is set to -1, we should infer its value from the other
// dimensions (including those that get copied from below).
TYPED_TEST(MLULstmReshapeTest, TestInferenceOfUnspecified) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);
  blob_shape->add_dim(5);
  blob_shape->add_dim(0);
  blob_shape->add_dim(50);

  // Count is 1000, thus height should be 1000 / (5*4*50) = 1.

  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 50);
}

TYPED_TEST(MLULstmReshapeTest, TestInferenceOfUnspecifiedWithStartAxis) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lstm_reshape_param()->set_axis(1);
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(4);
  blob_shape->add_dim(10);
  blob_shape->add_dim(-1);

  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 4);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 10);
  EXPECT_EQ(this->blob_top_->width(), 25);
}

TYPED_TEST(MLULstmReshapeTest, TestFlattenMiddle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lstm_reshape_param()->set_axis(1);
  layer_param.mutable_lstm_reshape_param()->set_num_axes(2);
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(-1);

  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 50 * 4);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

TYPED_TEST(MLULstmReshapeTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_lstm_reshape_param()->mutable_shape();
  shape->add_dim(2);
  shape->add_dim(25);
  shape->add_dim(5);
  shape->add_dim(4);
  MLULstmReshapeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
        this->blob_bottom_->cpu_data()[i]);
    sum += std::abs(this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSLstmReshapeTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSLstmReshapeTest()
      : blob_bottom_(new Blob<Dtype>(1, 50, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSLstmReshapeTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSLstmReshapeTest, TestMFUSDevices);

TYPED_TEST(MFUSLstmReshapeTest, TestValues) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* blob_shape =
      layer_param.mutable_lstm_reshape_param()->mutable_shape();
  blob_shape->add_dim(0);
  blob_shape->add_dim(-1);
  blob_shape->add_dim(1);
  blob_shape->add_dim(1);
  MLULstmReshapeLayer<Dtype> layer(layer_param);
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
  float err_sum = 0, sum = 0;
  for (int c = 0; c < 50 * 4 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
              this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5));
    err_sum += std::abs(this->blob_top_->data_at(0, c, 0, 0) -
        this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5));
    sum += this->blob_bottom_->data_at(0, c / (4 * 5), (c / 5) % 4, c % 5);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

TYPED_TEST(MFUSLstmReshapeTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BlobShape* shape = layer_param.mutable_lstm_reshape_param()->mutable_shape();
  shape->add_dim(2);
  shape->add_dim(25);
  shape->add_dim(5);
  shape->add_dim(4);
  MLULstmReshapeLayer<Dtype> layer(layer_param);
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
  float err_sum = 0, sum = 0;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i],
              this->blob_bottom_->cpu_data()[i]);
    err_sum += std::abs(this->blob_top_->cpu_data()[i] -
            this->blob_bottom_->cpu_data()[i]);
    sum += std::abs(this->blob_bottom_->cpu_data()[i]);
  }
  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
