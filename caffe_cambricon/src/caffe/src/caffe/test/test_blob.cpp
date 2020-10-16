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

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
  protected:
  BlobSimpleTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobSimpleTest() { delete blob_; delete blob_preshaped_; }
  Blob<Dtype>* const blob_;
  Blob<Dtype>* const blob_preshaped_;
};

TYPED_TEST_CASE(BlobSimpleTest, TestDtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_->num_axes(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
}

TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 4);
  EXPECT_EQ(this->blob_->width(), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

TYPED_TEST(BlobSimpleTest, TestRemember) {
  Blob<float> blob(2, 3, 4, 5);
  vector<int> shape = {2, 3, 2, 10};
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NCWH);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NHWC);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NHCW);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NWCH);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NWHC);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_CNHW);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_CNWH);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_CHWN);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_CWNH);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_CWHN);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_HNCW);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_HNWC);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_HCWN);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_HCNW);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_HWNC);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_WNCH);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_WNHC);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_WCHN);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_WCNH);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_WHNC);
  blob.Reshape(shape, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_WHCN);
  shape[0] = 16;
  blob.Reshape_only(shape);
  vector<int> shape2 = {2, 3, 2, 10, 6};
  Blob<float> blob2(shape2);
  blob2.Reshape(shape2, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NCDHW);
  blob2.Reshape(shape2, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_NDHWC);
  blob2.Reshape(shape2, DT_FLOAT32, DT_FLOAT32, CNML_TENSOR, CNML_DHWCN);
}

TYPED_TEST(BlobSimpleTest, TestReshapeZero) {
  vector<int> shape(2);
  shape[0] = 0;
  shape[1] = 5;
  this->blob_->Reshape(shape);
  EXPECT_EQ(this->blob_->count(), 0);
}

TYPED_TEST(BlobSimpleTest, TestToProto) {
  BlobProto blob_proto;
  // Reshape to (3 x 2).
  vector<int> shape(2);
  shape[0] = 3;
  shape[1] = 2;
  Blob<float> blob;
  blob.Reshape(shape);
  blob.ToProto(&blob_proto, 1);
  blob.ToProto(&blob_proto, 1, 0.1);
  Blob<double> blobd;
  blobd.Reshape(shape);
  blobd.ToProto(&blob_proto, 1);
  blobd.ToProto(&blob_proto, 1, 0.1);
  blobd.mutable_cpu_data();
  blobd.Update();
}

TYPED_TEST(BlobSimpleTest, TestLegacyBlobProtoShapeEquals) {
  BlobProto blob_proto;

  // Reshape to (3 x 2).
  vector<int> shape(2);
  shape[0] = 3;
  shape[1] = 2;
  this->blob_->Reshape(shape);

  // (3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

  // (3 x 2) blob != (0 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(0);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

  // (3 x 2) blob != (3 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(3);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

  // Reshape to (1 x 3 x 2).
  shape.insert(shape.begin(), 1);
  this->blob_->Reshape(shape);

  // (1 x 3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

  // Reshape to (2 x 3 x 2).
  shape[0] = 2;
  this->blob_->Reshape(shape);

  // (2 x 3 x 2) blob != (1 x 1 x 3 x 2) legacy blob
  blob_proto.set_num(1);
  blob_proto.set_channels(1);
  blob_proto.set_height(3);
  blob_proto.set_width(2);
  EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));
}

template <typename TypeParam>
class BlobMathTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  protected:
  BlobMathTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        epsilon_(1e-6) {}

  virtual ~BlobMathTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  Dtype epsilon_;
};

TYPED_TEST_CASE(BlobMathTest, TestDtypesAndDevices);

TYPED_TEST(BlobMathTest, TestSumOfSquares) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Blob should have sum of squares == 0.
  EXPECT_EQ(0, this->blob_->sumsq_data());
  EXPECT_EQ(0, this->blob_->sumsq_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  Dtype expected_sumsq = 0;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    expected_sumsq += data[i] * data[i];
  }
  // Do a mutable access on the current device,
  // so that the sumsq computation is done on that device.
  // (Otherwise, this would only check the CPU sumsq implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
#ifdef USE_MLU
  case Caffe::MLU:
  case Caffe::MFUS:
    this->blob_->mutable_mlu_data();
    break;
#endif
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
              this->epsilon_ * expected_sumsq);
  EXPECT_EQ(0, this->blob_->sumsq_diff());

  // Check sumsq_diff too.
  const Dtype kDiffScaleFactor = 7;
  caffe_cpu_scale(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
#ifdef USE_MLU
  case Caffe::MLU:
  case Caffe::MFUS:
    break;
#endif
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
              this->epsilon_ * expected_sumsq);
  const Dtype expected_sumsq_diff =
      expected_sumsq * kDiffScaleFactor * kDiffScaleFactor;
  EXPECT_NEAR(expected_sumsq_diff, this->blob_->sumsq_diff(),
              this->epsilon_ * expected_sumsq_diff);
}

TYPED_TEST(BlobMathTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Blob should have asum == 0.
  EXPECT_EQ(0, this->blob_->asum_data());
  EXPECT_EQ(0, this->blob_->asum_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  Dtype expected_asum = 0;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    expected_asum += std::fabs(data[i]);
  }
  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_asum, this->blob_->asum_data(),
              this->epsilon_ * expected_asum);
  EXPECT_EQ(0, this->blob_->asum_diff());

  // Check asum_diff too.
  const Dtype kDiffScaleFactor = 7;
  caffe_cpu_scale(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_asum, this->blob_->asum_data(),
              this->epsilon_ * expected_asum);
  const Dtype expected_diff_asum = expected_asum * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->blob_->asum_diff(),
              this->epsilon_ * expected_diff_asum);
}

TYPED_TEST(BlobMathTest, TestScaleData) {
  typedef typename TypeParam::Dtype Dtype;

  EXPECT_EQ(0, this->blob_->asum_data());
  EXPECT_EQ(0, this->blob_->asum_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  const Dtype asum_before_scale = this->blob_->asum_data();
  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Dtype kDataScaleFactor = 3;
  this->blob_->scale_data(kDataScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->blob_->asum_data(),
              this->epsilon_ * asum_before_scale * kDataScaleFactor);
  EXPECT_EQ(0, this->blob_->asum_diff());

  // Check scale_diff too.
  const Dtype kDataToDiffScaleFactor = 7;
  const Dtype* data = this->blob_->cpu_data();
  caffe_cpu_scale(this->blob_->count(), kDataToDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  const Dtype expected_asum_before_scale = asum_before_scale * kDataScaleFactor;
  EXPECT_NEAR(expected_asum_before_scale, this->blob_->asum_data(),
              this->epsilon_ * expected_asum_before_scale);
  const Dtype expected_diff_asum_before_scale =
      asum_before_scale * kDataScaleFactor * kDataToDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum_before_scale, this->blob_->asum_diff(),
              this->epsilon_ * expected_diff_asum_before_scale);
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Dtype kDiffScaleFactor = 3;
  this->blob_->scale_diff(kDiffScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->blob_->asum_data(),
              this->epsilon_ * asum_before_scale * kDataScaleFactor);
  const Dtype expected_diff_asum =
      expected_diff_asum_before_scale * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->blob_->asum_diff(),
              this->epsilon_ * expected_diff_asum);
}

}  // namespace caffe
