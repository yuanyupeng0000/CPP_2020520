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

#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_rnn_layer.hpp"
#include "caffe/layers/rnn_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RNNLayerTest;
template <typename TypeParam>
class MLURNNLayerTest;

template <typename Dtype>
class GlobalBlobHelper {
  public:
    static shared_ptr<Blob<Dtype>> global_blob_ptr_;
};

template <class Dtype>
shared_ptr<Blob<Dtype>> GlobalBlobHelper<Dtype>::global_blob_ptr_;

template <typename TypeParam>
class RNNLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  RNNLayerTest() : num_output_(16) {
    blob_bottom_vec_.push_back(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_cont_);
    blob_top_vec_.push_back(&blob_top_);

    ReshapeBlobs(1, 3);

    layer_param_.mutable_recurrent_param()->set_num_output(num_output_);
    FillerParameter* weight_filler =
        layer_param_.mutable_recurrent_param()->mutable_weight_filler();
    weight_filler->set_type("constant");
    weight_filler->set_value(0.02);
    FillerParameter* bias_filler =
        layer_param_.mutable_recurrent_param()->mutable_bias_filler();
    bias_filler->set_type("constant");
    bias_filler->set_value(0.01);

    layer_param_.set_phase(TEST);
  }

  void ReshapeBlobs(int num_timesteps, int num_instances) {
    blob_bottom_.Reshape(num_timesteps, num_instances, 3, 2);
    blob_bottom_static_.Reshape(num_instances, 2, 3, 4);
    blob_bottom_h0_.Reshape(1, num_instances, 16, 1);
    vector<int> shape(2);
    shape[0] = num_timesteps;
    shape[1] = num_instances;
    blob_bottom_cont_.Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(0.0012);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_);
  }

  int num_output_;
  LayerParameter layer_param_;
  Blob<Dtype> blob_bottom_;
  Blob<Dtype> blob_bottom_cont_;
  Blob<Dtype> blob_bottom_static_;
  Blob<Dtype> blob_bottom_h0_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RNNLayerTest, TestDtypesAndDevices);

TYPED_TEST(RNNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  RNNLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->blob_bottom_.shape();
  expected_top_shape.resize(3);
  expected_top_shape[2] = this->num_output_;
  EXPECT_TRUE(this->blob_top_.shape() == expected_top_shape);
}

TYPED_TEST(RNNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  const int kNumTimesteps = 3;
  const int num = this->blob_bottom_.shape(1);
  this->ReshapeBlobs(kNumTimesteps, num);

  // Fill the cont blob with <0, 1, 1, ..., 1>,
  // indicating a sequence that begins at the first timestep
  // then continues for the rest of the sequence.
  for (int t = 0; t < kNumTimesteps; ++t) {
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_cont_.mutable_cpu_data()[t * num + n] = t > 0;
    }
  }

  // Process the full sequence in a single batch.
  FillerParameter filler_param;
  filler_param.set_type("constant");
  filler_param.set_value(0.0024);
  ConstantFiller<Dtype> sequence_filler(filler_param);
  sequence_filler.Fill(&this->blob_bottom_);
  shared_ptr<RNNLayer<Dtype> > layer(new RNNLayer<Dtype>(this->layer_param_));
  Caffe::set_random_seed(1);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Calling forward for full sequence RNN";
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  vector<int> top_shape = this->blob_top_.shape();
  GlobalBlobHelper<Dtype>::global_blob_ptr_.reset(new Blob<Dtype>(top_shape));
  caffe_copy(this->blob_top_.count(), this->blob_top_.cpu_data(),
             GlobalBlobHelper<Dtype>::global_blob_ptr_->mutable_cpu_data());

  // Copy the inputs and outputs to reuse/check them later.
  Blob<Dtype> bottom_copy(this->blob_bottom_.shape());
  bottom_copy.CopyFrom(this->blob_bottom_);
  Blob<Dtype> top_copy(this->blob_top_.shape());
  top_copy.CopyFrom(this->blob_top_);

  // Process the batch one timestep at a time;
  // check that we get the same result.
  this->ReshapeBlobs(1, num);
  layer.reset(new RNNLayer<Dtype>(this->layer_param_));
  Caffe::set_random_seed(1);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const int bottom_count = this->blob_bottom_.count();
  const int top_count = this->blob_top_.count();
  const Dtype kEpsilon = 1e-5;
  for (int t = 0; t < kNumTimesteps; ++t) {
    caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
               this->blob_bottom_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_cont_.mutable_cpu_data()[n] = t > 0;
    }
    LOG(INFO) << "Calling forward for RNN timestep " << t;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < top_count; ++i) {
      ASSERT_LT(t * top_count + i, top_copy.count());
      EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                  top_copy.cpu_data()[t * top_count + i], kEpsilon)
          << "t = " << t << "; i = " << i;
    }
  }

  // Process the batch one timestep at a time with all cont blobs set to 0.
  // Check that we get a different result, except in the first timestep.
  Caffe::set_random_seed(1);
  layer.reset(new RNNLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int t = 0; t < kNumTimesteps; ++t) {
    caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
               this->blob_bottom_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_cont_.mutable_cpu_data()[n] = 0;
    }
    LOG(INFO) << "Calling forward for RNN timestep " << t;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < top_count; ++i) {
      if (t == 0) {
        EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                    top_copy.cpu_data()[t * top_count + i], kEpsilon)
            << "t = " << t << "; i = " << i;
      } else {
        EXPECT_NE(this->blob_top_.cpu_data()[i],
                  top_copy.cpu_data()[t * top_count + i])
            << "t = " << t << "; i = " << i;
      }
    }
  }
}

TYPED_TEST(RNNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  RNNLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

TYPED_TEST(RNNLayerTest, TestGradientNonZeroCont) {
  typedef typename TypeParam::Dtype Dtype;
  RNNLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_cont_.count(); ++i) {
    this->blob_bottom_cont_.mutable_cpu_data()[i] = i > 2;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

#ifdef USE_MLU
template <typename TypeParam>
class MLURNNLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLURNNLayerTest() : num_output_(16) {
    blob_bottom_vec_.push_back(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_cont_);
    blob_top_vec_.push_back(&blob_top_);

    ReshapeBlobs(1, 3);

    layer_param_.mutable_recurrent_param()->set_num_output(num_output_);
    FillerParameter* weight_filler =
        layer_param_.mutable_recurrent_param()->mutable_weight_filler();
    weight_filler->set_type("constant");
    weight_filler->set_value(0.02);
    FillerParameter* bias_filler =
        layer_param_.mutable_recurrent_param()->mutable_bias_filler();
    bias_filler->set_type("constant");
    bias_filler->set_value(0.01);

    layer_param_.set_phase(TEST);
  }

  void ReshapeBlobs(int num_timesteps, int num_instances) {
    blob_bottom_.Reshape(num_timesteps, num_instances, 3, 2);
    blob_bottom_static_.Reshape(num_instances, 2, 3, 4);
    blob_bottom_h0_.Reshape(1, num_instances, 16, 1);
    vector<int> shape(2);
    shape[0] = num_timesteps;
    shape[1] = num_instances;
    blob_bottom_cont_.Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(0.0012);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_);
  }

  int num_output_;
  LayerParameter layer_param_;
  Blob<Dtype> blob_bottom_;
  Blob<Dtype> blob_bottom_cont_;
  Blob<Dtype> blob_bottom_static_;
  Blob<Dtype> blob_bottom_h0_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLURNNLayerTest, TestMLUDevices);

TYPED_TEST(MLURNNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  MLURNNLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->blob_bottom_.shape();
  expected_top_shape.resize(3);
  expected_top_shape[2] = this->num_output_;
  EXPECT_TRUE(this->blob_top_.shape() == expected_top_shape);
  OUTPUT("bottom1", this->blob_bottom_.shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_cont_.shape_string().c_str());
  OUTPUT("bottom3", this->blob_bottom_static_.shape_string().c_str());
  OUTPUT("bottom4", this->blob_bottom_h0_.shape_string().c_str());
}

template <typename TypeParam>
class MFUSRNNLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSRNNLayerTest() : num_output_(16) {
    blob_bottom_vec_.push_back(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_cont_);
    blob_top_vec_.push_back(&blob_top_);

    ReshapeBlobs(1, 3);

    layer_param_.mutable_recurrent_param()->set_num_output(num_output_);
    FillerParameter* weight_filler =
        layer_param_.mutable_recurrent_param()->mutable_weight_filler();
    weight_filler->set_type("constant");
    weight_filler->set_value(0.02);
    FillerParameter* bias_filler =
        layer_param_.mutable_recurrent_param()->mutable_bias_filler();
    bias_filler->set_type("constant");
    bias_filler->set_value(0.01);
    layer_param_.set_phase(TEST);
  }

  void ReshapeBlobs(int num_timesteps, int num_instances) {
    blob_bottom_.Reshape(num_timesteps, num_instances, 3, 2);
    blob_bottom_static_.Reshape(num_instances, 2, 3, 4);
    blob_bottom_h0_.Reshape(1, num_instances, 16, 1);
    vector<int> shape(2);
    shape[0] = num_timesteps;
    shape[1] = num_instances;
    blob_bottom_cont_.Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(0.0012);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_);
  }

  int num_output_;
  LayerParameter layer_param_;
  Blob<Dtype> blob_bottom_;
  Blob<Dtype> blob_bottom_cont_;
  Blob<Dtype> blob_bottom_static_;
  Blob<Dtype> blob_bottom_h0_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSRNNLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSRNNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  MLURNNLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->blob_bottom_.shape();
  expected_top_shape.resize(3);
  expected_top_shape[2] = this->num_output_;
  EXPECT_TRUE(this->blob_top_.shape() == expected_top_shape);
  OUTPUT("bottom1", this->blob_bottom_.shape_string().c_str());
  OUTPUT("bottom2", this->blob_bottom_cont_.shape_string().c_str());
  OUTPUT("bottom3", this->blob_bottom_static_.shape_string().c_str());
  OUTPUT("bottom4", this->blob_bottom_h0_.shape_string().c_str());
}

#endif

}  // namespace caffe
