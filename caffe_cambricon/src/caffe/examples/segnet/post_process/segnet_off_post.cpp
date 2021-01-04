/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include "glog/logging.h"
#include "cnrt.h" // NOLINT
#include "segnet_off_post.hpp"
#include "post_processor.hpp"
#include "runner.hpp"
#include "off_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::vector;
using std::string;

template<typename Dtype, template <typename> class Qtype>
void SegnetOffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(infr->deviceId());

  outCpuPtrs_ = new(Dtype);
  outCpuPtrs_[0] = new float[infr->outCounts()[0]];
  syncCpuPtrs_ = new(Dtype);
  syncCpuPtrs_[0] = new char[infr->outputSizeArray()[0]];

  int dim_values[4] = {infr->outNum(), infr->outHeight(),
    infr->outWidth(), infr->outChannel()};
  int dim_order[4] = {0, 3, 1, 2};

  while (true) {
    Dtype* mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) break;  // no more work

    Timer copyout;
    TimePoint t1 = std::chrono::high_resolution_clock::now();
    CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_[0], mluOutData[0],
                          infr->outputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    cnrtDataType_t mluDtype = infr->mluOutputDtype()[0];
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
            outCpuPtrs_[0], cpuDtype,
            nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(syncCpuPtrs_[0], cpuDtype,
            outCpuPtrs_[0], 4, dim_values, dim_order));
    }
    copyout.log("copyout time ...");

    infr->pushFreeOutputData(mluOutData);
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = infr->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    infr->pushFreeInputTimeTraceData(timetrace);

    vector<string> origin_img = infr->popValidInputNames();
    getResults(origin_img);
  }
}

template<typename Dtype, template <typename> class Qtype>
void SegnetOffPostProcessor<Dtype, Qtype>::runSerial() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  if (!this->initSerialMode) {
    outCpuPtrs_ = new(Dtype);
    outCpuPtrs_[0] = new float[infr->outCounts()[0]];
    syncCpuPtrs_ = new(Dtype);
    syncCpuPtrs_[0] = new char[infr->outputSizeArray()[0]];

    this->initSerialMode = true;
  }

  Dtype* mluOutData = infr->popValidOutputData();
  int dim_values[4] = {infr->outNum(), infr->outHeight(),
      infr->outWidth(), infr->outChannel()};
  int dim_order[4] = {0, 3, 1, 2};
  TimePoint t1 = std::chrono::high_resolution_clock::now();
  CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_[0], mluOutData[0],
                        infr->outputSizeArray()[0],
                        CNRT_MEM_TRANS_DIR_DEV2HOST));
  cnrtDataType_t cpuDtype = CNRT_FLOAT32;
  cnrtDataType_t mluDtype = infr->mluOutputDtype()[0];
  if (cpuDtype != mluDtype) {
    CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
          outCpuPtrs_[0], cpuDtype,
          nullptr, 4, dim_values, dim_order));
  } else {
    CNRT_CHECK(cnrtTransDataOrder(syncCpuPtrs_[0], cpuDtype,
          outCpuPtrs_[0], 4, dim_values, dim_order));
  }
  infr->pushFreeOutputData(mluOutData);
  TimePoint t2 = std::chrono::high_resolution_clock::now();
  auto timetrace = infr->popValidOutputTimeTraceData();
  timetrace->out_start = t1;
  timetrace->out_end = t2;
  this->appendTimeTrace(*timetrace);
  infr->pushFreeInputTimeTraceData(timetrace);

  vector<string> origin_img = infr->popValidInputNames();

  getResults(origin_img);
}
template <typename Dtype, template <typename> class Qtype>
void SegnetOffPostProcessor<Dtype, Qtype>::getResults(
                                               vector<string> origin_img) {
  Runner<Dtype, Qtype> * runner = static_cast<Runner<Dtype, Qtype>*>(this->runner_);
  float* data = reinterpret_cast<float*>(outCpuPtrs_[0]);
  int c = runner->outChannel();
  int h = runner->h();
  int w = runner->w();
  int i = 0;
  unsigned char* r_data = reinterpret_cast<unsigned char*>(
     malloc(h * w * sizeof(unsigned char)));
  unsigned char* g_data = reinterpret_cast<unsigned char*>(
     malloc(h * w * sizeof(unsigned char)));
  unsigned char* b_data = reinterpret_cast<unsigned char*>(
     malloc(h * w * sizeof(unsigned char)));
  cv::Mat r(h, w, CV_8UC1, r_data);
  cv::Mat g(h, w, CV_8UC1, g_data);
  cv::Mat b(h, w, CV_8UC1, b_data);

  for (auto img_name : origin_img) {
    if (img_name != "null") {
      int pos = img_name.find_last_of('/');
      string img_num;
      string name = img_name;
      if (pos > 0 && pos < img_name.size()) {
        name = name.substr(pos + 1);
      }
      pos = name.rfind(".");
      if (pos > 0 && pos < name.size()) {
        name = name.substr(0, pos);
      }
      img_num = name;
      float* tmp_data = data + i * c * h * w;
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          float max_prob = 0.0;
          int max_index = 0;
          for (int m = 0; m < c; m++) {
              int index = m * h * w + j * w + k;
              if (tmp_data[index] > max_prob) {
                max_prob = tmp_data[index];
                max_index = m;
              }
          }
          if (max_index >= this->classRGBLabelVector.size()) {
              LOG(ERROR) << "max_index : "
                         << max_index
                         << " , is more than "
                         << this->classRGBLabelVector.size();
              max_index = 0;
          }
          r_data[j*w + k] = this->classRGBLabelVector[max_index][0];
          g_data[j*w + k] = this->classRGBLabelVector[max_index][1];
          b_data[j*w + k] = this->classRGBLabelVector[max_index][2];
        }
      }
      cv::Mat mergedImage;
      cv::Mat rgb[3] = {b, g, r};
      cv::merge(rgb, 3, mergedImage);
      cv::imwrite(FLAGS_outputdir + "/" + img_num + ".png", mergedImage);
    }
    i++;
  }
  free(r_data);
  free(g_data);
  free(b_data);
}
INSTANTIATE_OFF_CLASS(SegnetOffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
