/*
All modification made by Cambricon Corporation: © 2020 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2020, the respective contributors
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
#include <queue>
#include <string>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <iomanip>
#include "cnrt.h" // NOLINT
#include "glog/logging.h"
#include "fcn_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "on_data_provider.hpp"
#include "common_functions.hpp"
using std::vector;
using std::string;

template <typename Dtype, template <typename> class Qtype>
void FcnOnPostProcessor<Dtype, Qtype>::getResults(float* data,
                                               vector<string> origin_img) {
  Runner<Dtype, Qtype> * runner = static_cast<Runner<Dtype, Qtype>*>(this->runner_);
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
      cv::imwrite(FLAGS_outputdir + '/' + img_num + ".png", mergedImage);
    }
    i++;
  }
  free(r_data);
  free(g_data);
  free(b_data);
}

template <typename Dtype, template <typename> class Qtype>
void FcnOnPostProcessor<Dtype, Qtype>::runSerial() {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  if (!this->initSerialMode) {
    this->initSerialMode = true;
  }
  caffe::Net<float>* netBuff = runner->net();
  auto outputBlob = netBuff->output_blobs()[0];
  resultDataPtr_ = outputBlob->mutable_cpu_data();
  vector<string> origin_img = runner->popValidInputNames();
  getResults(resultDataPtr_, origin_img);
}

template <typename Dtype, template <typename> class Qtype>
void FcnOnPostProcessor<Dtype, Qtype>::runParallel() {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  ::setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());
  this->outCount_ = runner->outCounts()[0];
  this->outN_ = runner->outNum();
  caffe::Net<float>* netBuff = runner->net();
  int outputCount = netBuff->output_blobs()[0]->count();
  outputCpuPtr_ = new Dtype[outputCount];
  size_t tensorSize;
  MLU_CHECK(cnmlGetTensorSize_V2(
      netBuff->output_blobs()[0]->mlu_tensor(), &tensorSize));
  std::vector<std::future<void>> futureVector;
  while (true) {
    Dtype* outputMluPtr = runner->popValidOutputData();
    Dtype* outputSyncPtr = runner->popValidOutputSyncData();
    if (nullptr == outputMluPtr) break;  // no more work, exit
    TimePoint t1 = std::chrono::high_resolution_clock::now();

    auto outputBlob = netBuff->output_blobs()[0];
    Timer timer;
    CNRT_CHECK(cnrtMemcpy(outputSyncPtr, outputMluPtr,
          tensorSize, CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = to_cnrt_dtype(outputBlob->cpu_type());
    cnrtDataType_t mluDtype = to_cnrt_dtype(outputBlob->mlu_type());
    int dim_values[4] = {outputBlob->mlu_shape()[0], outputBlob->mlu_shape()[1],
                         outputBlob->mlu_shape()[2], outputBlob->mlu_shape()[3]};
    int dim_order[4] = {0, 3, 1, 2};
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(outputSyncPtr),
                                       mluDtype,
                                       reinterpret_cast<void*>(outputCpuPtr_),
                                       cpuDtype,
                                       nullptr,
                                       outputBlob->mlu_shape().size(),
                                       dim_values,
                                       dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(outputSyncPtr),
                                    cpuDtype,
                                    reinterpret_cast<void*>(outputCpuPtr_),
                                    outputBlob->mlu_shape().size(),
                                    dim_values,
                                    dim_order));
    }
    resultDataPtr_ = outputCpuPtr_;
    timer.log("copy out time");
    vector<string> origin_img = runner->popValidInputNames();
    getResults(resultDataPtr_, origin_img);
    runner->pushFreeOutputData(outputMluPtr);
    runner->pushFreeOutputSyncData(outputSyncPtr);
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = runner->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    runner->pushFreeInputTimeTraceData(timetrace);
  }
}
INSTANTIATE_ON_CLASS(FcnOnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
