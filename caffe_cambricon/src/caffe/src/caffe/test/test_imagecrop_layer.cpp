/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <time.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "gtest/gtest.h"
#ifdef USE_MLU
#include "caffe/layers/mlu_image_crop_layer.hpp"
#endif

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifdef USE_MLU
template <typename TypeParam>
class MLUImagecropLayerTest : public MLUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MLUImagecropLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 4, 224, 224)),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MLUImagecropLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(int crop_x, int crop_y, int crop_h, int crop_w) {
    this->SetUp();
    LayerParameter layer_param;
    ImagecropParameter* crop_param =
                         layer_param.mutable_image_crop_param();
    crop_param->set_crop_x(crop_x);
    crop_param->set_crop_y(crop_y);
    crop_param->set_crop_h(crop_h);
    crop_param->set_crop_w(crop_w);
    MLUImagecropLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* blob_crop_;
  Blob<Dtype>* blob_roi_;
  Blob<Dtype>* blob_pad_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MLUImagecropLayerTest, TestMLUDevices);

TYPED_TEST(MLUImagecropLayerTest, TestForward) {
  int channel = 4;
  int crop_x = 0;
  int crop_y = 10;
  int crop_h = 300;
  int crop_w = 200;

  // read and process real image data
  string filename = string(TEST_SOURCE_DIR()) + "dog.jpg";
  cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if (!img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
  }
  int s_col = img.cols;
  int s_row = img.rows;
  cv::Mat imgd =
      cv::Mat(cv::Size(crop_w, crop_h), img.type(), cv::Scalar::all(0));
  cv::Mat imgcv;
  if (crop_w > 0 && crop_h > 0) {
    cv::Rect roi;
    roi.x = crop_x;
    roi.y = crop_y;
    roi.width = crop_w;
    roi.height = crop_h;
    cv::Mat crop_img = img(roi);
    cv::resize(crop_img, imgcv, cv::Size(crop_w, crop_h));
  }

  this->blob_bottom_->Reshape(1, 4, s_row, s_col);
  uint8_t* src = reinterpret_cast<uint8_t*>(this->blob_bottom_->mutable_cpu_data());
  int index = 0;
  std::ofstream file0;
  for (int i = 0; i < s_row * s_col; i++) {
    for (int j = 0; j < 3; j++) {
      src[i * 4 + j] = *(reinterpret_cast<uint8_t*>(img.data) + (index++));
    }
    src[i * 4 + 3] = (uint8_t)0;
  }

  this->TestForward(crop_x, crop_y, crop_h, crop_w);
  uint8_t* dst = reinterpret_cast<uint8_t*>(this->blob_top_->mutable_cpu_data());
  float sum = 0.0, square_sum = 0.0, tmp = 0;
  float delta_sum = 0.0, delta_square_sum = 0.0;
  float delta = 0.0;
  for (int i = 0; i < crop_h; i++) {
    for (int j = 0; j < crop_w; j++) {
      for (int c = 0; c < 3; c++) {
        // the output of resize is rgb0 that means the data
        // the index is in NHWC order
        int sx = (i * crop_w + j) * channel + c;
        int idx = (i * crop_w + j) * 3 + c;
        uint8_t val = *(reinterpret_cast<uint8_t*>(dst) + sx);
        delta = static_cast<float>(abs(val - (imgcv.data)[idx]));
        delta_sum += delta;
        delta_square_sum += delta*delta;
        tmp = static_cast<float>(abs(val));
        sum += tmp;
        square_sum += tmp * tmp;
      }
    }
  }
  EXPECT_LE(delta_sum / sum, 2e-2);
}


template <typename TypeParam>
class MFUSImagecropLayerTest : public MFUSDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
  MFUSImagecropLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 4, 224, 224)),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MFUSImagecropLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void TestForward(int crop_x, int crop_y, int crop_h, int crop_w) {
    this->SetUp();
    LayerParameter layer_param;
    ImagecropParameter* crop_param =
                         layer_param.mutable_image_crop_param();
    crop_param->set_crop_x(crop_x);
    crop_param->set_crop_y(crop_y);
    crop_param->set_crop_h(crop_h);
    crop_param->set_crop_w(crop_w);
    MLUImagecropLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    ASSERT_TRUE(layer.mfus_supported());
    MFusion<Dtype> fuser;
    fuser.reset();
    fuser.addInputs(this->blob_bottom_vec_);
    fuser.addOutputs(this->blob_top_vec_);
    layer.Reshape_dispatch(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.fuse(&fuser);
    fuser.compile();
    float timeUse;
    struct timeval tpStart, tpEnd;
    gettimeofday(&tpStart, NULL);
    fuser.forward();
    gettimeofday(&tpEnd, NULL);
    timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
               tpStart.tv_usec;
    LOG(INFO) << "Fusion MLU resize execution time: " << timeUse << " us";
  }
  void Imagecrop_test(int crop_x, int crop_y,
                       int crop_h, int crop_w) {
    int channel = 4;
    // read and process real image data
    string filename = string(TEST_SOURCE_DIR()) + "dog.jpg";
    cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if (!img.data) {
      LOG(ERROR) << "Could not open or find file " << filename;
    }
    float cvTimeUse;
    struct timeval cvStart, cvEnd;
    cv::Mat imgd =
        cv::Mat(cv::Size(crop_w, crop_h), img.type(), cv::Scalar::all(0));
    cv::Mat imgcv = imgd;
    gettimeofday(&cvStart, NULL);
    if (crop_w > 0 && crop_h > 0) {
      cv::Rect roi;
      roi.x = crop_x;
      roi.y = crop_y;
      roi.width = crop_w;
      roi.height = crop_h;
      cv::Mat crop_img = img(roi);
      cv::resize(crop_img, imgcv, imgcv.size());
    } else {
      cv::resize(img, imgcv, imgcv.size());
    }
    gettimeofday(&cvEnd, NULL);
    cvTimeUse = 1000000 * (cvEnd.tv_sec - cvStart.tv_sec) + cvEnd.tv_usec -
               cvStart.tv_usec;
    LOG(INFO) << "CPU opencv resize execution time: " << cvTimeUse << " us";
    // cv::imwrite("cvimg.jpg", imgcv);
    int s_col = img.cols;
    int s_row = img.rows;
    this->blob_bottom_->Reshape(1, 4, s_row, s_col);
    uint8_t* src = reinterpret_cast<uint8_t*>(this->blob_bottom_->mutable_cpu_data());
    int index = 0;
    for (int i = 0; i < s_row * s_col; i++) {
      for (int j = 0; j < 3; j++)
        src[i * 4 + j] = *(reinterpret_cast<uint8_t*>(img.data) + (index++));
      src[i * 4 + 3] = (uint8_t)0;
    }
    this->TestForward(crop_x, crop_y, crop_h, crop_w);
    uint8_t* dst = reinterpret_cast<uint8_t*>(this->blob_top_->mutable_cpu_data());
    float sum = 0.0, square_sum = 0.0, tmp = 0;
    float delta_sum = 0.0, delta_square_sum = 0.0;
    float delta = 0.0;
    int idx = 0;
    for (int i = 0; i < crop_h; i++) {
      for (int j = 0; j < crop_w; j++) {
        for (int c = 0; c < 3; c++) {
          // the output of resize is rgb0 that means the data
          // the index is in NHWC order
          int sx = (i * crop_w + j) * channel + c;
          uint8_t val = *(reinterpret_cast<uint8_t*>(dst) + sx);
          delta = static_cast<float>(abs(val - (imgcv.data)[idx++]));
          delta_sum += delta;
          delta_square_sum += delta*delta;
          tmp = static_cast<float>(abs(val));
          sum += tmp;
          square_sum += tmp * tmp;
        }
      }
    }
    EXPECT_LE(delta_sum / sum, 2e-2);
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MFUSImagecropLayerTest, TestMFUSDevices);

TYPED_TEST(MFUSImagecropLayerTest, TestForward_200_300) {
  int crop_x = 0;
  int crop_y = 0;
  int crop_h = 200;
  int crop_w = 300;
  this->Imagecrop_test(crop_x, crop_y, crop_h, crop_w);
}
TYPED_TEST(MFUSImagecropLayerTest, TestForward_300_320) {
  int crop_x = 10;
  int crop_y = 0;
  int crop_h = 300;
  int crop_w = 200;
  this->Imagecrop_test(crop_x, crop_y, crop_h, crop_w);
}

#endif
}  // namespace caffe
