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

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mlu_strided_slice_layer.hpp"
#include "caffe/layers/strided_slice_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
template <typename TypeParam>
class StridedSliceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  StridedSliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~StridedSliceLayerTest() {
    delete blob_top_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(StridedSliceLayerTest, TestDtypesAndDevices);

TYPED_TEST(StridedSliceLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_stridedslice_param()->set_n_begin(0);
  layer_param.mutable_stridedslice_param()->set_c_begin(1);
  layer_param.mutable_stridedslice_param()->set_h_begin(0);
  layer_param.mutable_stridedslice_param()->set_w_begin(0);
  layer_param.mutable_stridedslice_param()->set_n_end(1);
  layer_param.mutable_stridedslice_param()->set_c_end(2);
  layer_param.mutable_stridedslice_param()->set_h_end(1);
  layer_param.mutable_stridedslice_param()->set_w_end(3);
  StridedSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
}
TYPED_TEST(StridedSliceLayerTest, TestStridedSliceForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_stridedslice_param()->set_n_begin(0);
  layer_param.mutable_stridedslice_param()->set_c_begin(1);
  layer_param.mutable_stridedslice_param()->set_h_begin(0);
  layer_param.mutable_stridedslice_param()->set_w_begin(0);
  layer_param.mutable_stridedslice_param()->set_n_end(1);
  layer_param.mutable_stridedslice_param()->set_c_end(2);
  layer_param.mutable_stridedslice_param()->set_h_end(1);
  layer_param.mutable_stridedslice_param()->set_w_end(3);
  StridedSliceLayer<Dtype> layer(layer_param);
  const StridedSliceParameter& stridedslice_param =
                                layer_param.stridedslice_param();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int ni = this->blob_bottom_vec_[0]->num();
  int ci = this->blob_bottom_vec_[0]->channels();
  int hi = this->blob_bottom_vec_[0]->height();
  int wi = this->blob_bottom_vec_[0]->width();
  int no = this->blob_top_vec_[0]->num();
  int co = this->blob_top_vec_[0]->channels();
  int ho = this->blob_top_vec_[0]->height();
  int wo = this->blob_top_vec_[0]->width();
  int nb = stridedslice_param.has_n_begin() ?
                                stridedslice_param.n_begin() : 1;
  int cb = stridedslice_param.has_c_begin() ?
                                stridedslice_param.c_begin() : 1;
  int hb = stridedslice_param.has_h_begin() ?
                                stridedslice_param.h_begin() : 1;
  int wb = stridedslice_param.has_w_begin() ?
                                stridedslice_param.w_begin() : 1;
  int ns = stridedslice_param.has_n_stride() ?
                                stridedslice_param.n_stride() : 1;
  int cs = stridedslice_param.has_c_stride() ?
                                stridedslice_param.c_stride() : 1;
  int hs = stridedslice_param.has_h_stride() ?
                                stridedslice_param.h_stride() : 1;
  int ws = stridedslice_param.has_w_stride() ?
                                stridedslice_param.w_stride() : 1;
  float err_sum = 0, sum = 0;
  auto *input = this->blob_bottom_vec_[0]->cpu_data();
  auto *output = this->blob_top_vec_[0]->cpu_data();
  for (int nn = 0; nn < no; ++nn) {
    for (int cc = 0; cc < co; ++cc) {
      for (int hh = 0; hh < ho; ++hh) {
        for (int ww = 0; ww < wo; ++ww) {
          int nni = nb + nn * ns;
          if (nni < 0)
            nni = ni + nni;
          int hhi = hb + hh * hs;
          if (hhi < 0)
            hhi = hi + hhi;
          int wwi = wb + ww * ws;
          if (wwi < 0)
            wwi = wi + wwi;
          int cci = cb + cc * cs;
          if (cci < 0)
            cci = ci + cci;
          EXPECT_NEAR(
             output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww],
             input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi], 5e-3);
          err_sum +=std::abs(
                  output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww]-
                  input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi]);
          sum += std::abs(
                  output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww]);
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom:" << this->blob_bottom_vec_[0]->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

#ifdef USE_MLU

template <typename TypeParam>
class MLUStridedSliceLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUStridedSliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~MLUStridedSliceLayerTest() {
    delete blob_top_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(MLUStridedSliceLayerTest, TestMLUDevices);

TYPED_TEST(MLUStridedSliceLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_stridedslice_param()->set_n_begin(0);
  layer_param.mutable_stridedslice_param()->set_c_begin(1);
  layer_param.mutable_stridedslice_param()->set_h_begin(0);
  layer_param.mutable_stridedslice_param()->set_w_begin(0);
  layer_param.mutable_stridedslice_param()->set_n_end(1);
  layer_param.mutable_stridedslice_param()->set_c_end(2);
  layer_param.mutable_stridedslice_param()->set_h_end(1);
  layer_param.mutable_stridedslice_param()->set_w_end(3);
  MLUStridedSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
}
TYPED_TEST(MLUStridedSliceLayerTest, TestStridedSliceForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_stridedslice_param()->set_n_begin(0);
  layer_param.mutable_stridedslice_param()->set_c_begin(1);
  layer_param.mutable_stridedslice_param()->set_h_begin(0);
  layer_param.mutable_stridedslice_param()->set_w_begin(0);
  layer_param.mutable_stridedslice_param()->set_n_end(1);
  layer_param.mutable_stridedslice_param()->set_c_end(2);
  layer_param.mutable_stridedslice_param()->set_h_end(1);
  layer_param.mutable_stridedslice_param()->set_w_end(3);
  MLUStridedSliceLayer<Dtype> layer(layer_param);
  const StridedSliceParameter& mlustridedslice_param =
                                layer_param.stridedslice_param();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int no = this->blob_top_vec_[0]->num();
  int co = this->blob_top_vec_[0]->channels();
  int ho = this->blob_top_vec_[0]->height();
  int wo = this->blob_top_vec_[0]->width();
  int ni = this->blob_bottom_vec_[0]->num();
  int ci = this->blob_bottom_vec_[0]->channels();
  int hi = this->blob_bottom_vec_[0]->height();
  int wi = this->blob_bottom_vec_[0]->width();
  int nb = mlustridedslice_param.has_n_begin() ?
                                mlustridedslice_param.n_begin() : 1;
  int cb = mlustridedslice_param.has_c_begin() ?
                                mlustridedslice_param.c_begin() : 1;
  int hb = mlustridedslice_param.has_h_begin() ?
                                mlustridedslice_param.h_begin() : 1;
  int wb = mlustridedslice_param.has_w_begin() ?
                                mlustridedslice_param.w_begin() : 1;
  int ns =  mlustridedslice_param.has_n_stride() ?
                                mlustridedslice_param.n_stride() : 1;
  int cs =  mlustridedslice_param.has_c_stride() ?
                                mlustridedslice_param.c_stride() : 1;
  int hs =  mlustridedslice_param.has_h_stride() ?
                                mlustridedslice_param.h_stride() : 1;
  int ws =  mlustridedslice_param.has_w_stride() ?
                                mlustridedslice_param.w_stride() : 1;
  float err_sum = 0, sum = 0;
  auto *input = this->blob_bottom_vec_[0]->cpu_data();
  auto *output = this->blob_top_vec_[0]->cpu_data();
  for (int nn = 0; nn < no; ++nn) {
    for (int cc = 0; cc < co; ++cc) {
      for (int hh = 0; hh < ho; ++hh) {
        for (int ww = 0; ww < wo; ++ww) {
          int nni = nb + nn * ns;
          if (nni < 0)
            nni = ni + nni;
          int hhi = hb + hh * hs;
          if (hhi < 0)
            hhi = hi + hhi;
          int wwi = wb + ww * ws;
          if (wwi < 0)
            wwi = wi + wwi;
          int cci = cb + cc * cs;
          if (cci < 0)
            cci = ci + cci;
          EXPECT_NEAR(
                  output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww],
                  input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi], 5e-3);
          err_sum +=std::abs(
                  output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww]-
                  input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi]);
          sum += std::abs(
                  output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww]);
        }
      }
    }
  }
  std::ostringstream stream;
  stream << "bottom:" << this->blob_bottom_vec_[0]->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(layer.get_event_time());
}

template <typename TypeParam>
class MFUSStridedSliceLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSStridedSliceLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }

  virtual ~MFUSStridedSliceLayerTest() {
    delete blob_top_;
    delete blob_bottom_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(MFUSStridedSliceLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSStridedSliceLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_stridedslice_param()->set_n_begin(0);
  layer_param.mutable_stridedslice_param()->set_c_begin(1);
  layer_param.mutable_stridedslice_param()->set_h_begin(0);
  layer_param.mutable_stridedslice_param()->set_w_begin(0);
  layer_param.mutable_stridedslice_param()->set_n_end(1);
  layer_param.mutable_stridedslice_param()->set_c_end(2);
  layer_param.mutable_stridedslice_param()->set_h_end(1);
  layer_param.mutable_stridedslice_param()->set_w_end(3);
  MLUStridedSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(MFUSStridedSliceLayerTest, TestStridedSliceForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_stridedslice_param()->set_n_begin(0);
  layer_param.mutable_stridedslice_param()->set_c_begin(1);
  layer_param.mutable_stridedslice_param()->set_h_begin(0);
  layer_param.mutable_stridedslice_param()->set_w_begin(0);
  layer_param.mutable_stridedslice_param()->set_n_end(1);
  layer_param.mutable_stridedslice_param()->set_c_end(2);
  layer_param.mutable_stridedslice_param()->set_h_end(1);
  layer_param.mutable_stridedslice_param()->set_w_end(3);
  MLUStridedSliceLayer<Dtype> layer(layer_param);
  const StridedSliceParameter& mlustridedslice_param =
                                layer_param.stridedslice_param();
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
  int ni = this->blob_bottom_vec_[0]->num();
  int ci = this->blob_bottom_vec_[0]->channels();
  int hi = this->blob_bottom_vec_[0]->height();
  int wi = this->blob_bottom_vec_[0]->width();
  int no = this->blob_top_vec_[0]->num();
  int co = this->blob_top_vec_[0]->channels();
  int ho = this->blob_top_vec_[0]->height();
  int wo = this->blob_top_vec_[0]->width();
  int nb = mlustridedslice_param.has_n_begin() ?
                        mlustridedslice_param.n_begin() : 1;
  int cb = mlustridedslice_param.has_c_begin() ?
                        mlustridedslice_param.c_begin() : 1;
  int hb = mlustridedslice_param.has_h_begin() ?
                        mlustridedslice_param.h_begin() : 1;
  int wb = mlustridedslice_param.has_w_begin() ?
                        mlustridedslice_param.w_begin() : 1;
  int ns =  mlustridedslice_param.has_n_stride() ?
                        mlustridedslice_param.n_stride() : 1;
  int cs =  mlustridedslice_param.has_c_stride() ?
                        mlustridedslice_param.c_stride() : 1;
  int hs =  mlustridedslice_param.has_h_stride() ?
                        mlustridedslice_param.h_stride() : 1;
  int ws =  mlustridedslice_param.has_w_stride() ?
                        mlustridedslice_param.w_stride() : 1;
  float err_sum = 0, sum = 0;
  auto *input = this->blob_bottom_vec_[0]->cpu_data();
  auto *output = this->blob_top_vec_[0]->cpu_data();
  for (int nn = 0; nn < no; ++nn) {
    for (int cc = 0; cc < co; ++cc) {
      for (int hh = 0; hh < ho; ++hh) {
        for (int ww = 0; ww < wo; ++ww) {
          int nni = nb + nn * ns;
          if (nni < 0)
            nni = ni + nni;
          int hhi = hb + hh * hs;
          if (hhi < 0)
            hhi = hi + hhi;
          int wwi = wb + ww * ws;
          if (wwi < 0)
            wwi = wi + wwi;
          int cci = cb + cc * cs;
          if (cci < 0)
            cci = ci + cci;
          EXPECT_NEAR(output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww],
              input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi], 5e-3);
          err_sum +=std::abs(
                output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww]-
                input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi]);
          sum += std::abs(
                output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww]);
        }
      }
    }
  }

  std::ostringstream stream;
  stream << "bottom1:" << this->blob_bottom_vec_[0]->shape_string().c_str();
  BOTTOM(stream);
  ERR_RATE(err_sum/sum);
  EVENT_TIME(fuser.get_event_time());
}

#endif

}  // namespace caffe
