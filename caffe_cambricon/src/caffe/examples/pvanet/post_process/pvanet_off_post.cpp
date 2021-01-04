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
#include <queue>
#include <string>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <iomanip>
#include "cnrt.h" // NOLINT
#include "glog/logging.h"
#include "pvanet_off_post.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"
#include "threadPool.h"

using std::vector;
using std::string;

template<typename Dtype, template <typename> class Qtype>
void PvanetOffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(infr->deviceId());

  this->readLabels(&this->label_to_display_name);

  outCpuPtrs_ = new Dtype[infr->outBlobNum()];
  outSyncPtrs_ = new Dtype[infr->outBlobNum()];
  for (int i = 0; i < infr->outBlobNum(); ++i) {
    outCpuPtrs_[i] = new float[infr->outCounts()[i]];
    outSyncPtrs_[i] = new char[infr->outputSizeArray()[i]];
  }

  int TASK_NUM = SimpleInterface::thread_num;
  std::vector<std::future<void>> futureVector;
  int height = infr->h();
  int width = infr->w();
  while (true) {
    Dtype* mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) break;  // no more work

    Timer copyout;
    TimePoint t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < infr->outBlobNum(); ++i) {
      cnrtDataType cpuDtype = CNRT_FLOAT32;
      cnrtDataType mluDtype = infr->mluOutputDtype()[i];
      CNRT_CHECK(cnrtMemcpy(outSyncPtrs_[i],
                            mluOutData[i],
                            infr->outputSizeArray()[i],
                            CNRT_MEM_TRANS_DIR_DEV2HOST));
      vector<int> dimOrderTmp(infr->outDimNums()[i]);
      for (int j = 0; j < infr->outDimNums()[i]; ++j) {
        dimOrderTmp[j] = j;
      }
      vector<int> dimOrder = to_cpu_shape(dimOrderTmp);
      if (cpuDtype != mluDtype) {
        CNRT_CHECK(cnrtTransOrderAndCast(outSyncPtrs_[i], mluDtype,
              outCpuPtrs_[i], cpuDtype, nullptr, infr->outDimNums()[i],
              infr->outDimValues()[i], dimOrder.data()));
      } else {
        CNRT_CHECK(cnrtTransDataOrder(outSyncPtrs_[i], mluDtype, outCpuPtrs_[i],
              infr->outDimNums()[i], infr->outDimValues()[i], dimOrder.data()));
      }
    }
    copyout.log("copyout time ...");

    infr->pushFreeOutputData(mluOutData);
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = infr->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    infr->pushFreeInputTimeTraceData(timetrace);

    Timer postProcess;
    vector<cv::Mat> imgs;
    vector<string> img_names;
    vector<vector<vector<float>>> boxes = getResults(&imgs, &img_names);

    if (FLAGS_dump) {
      Timer dumpTimer;
      const int size = imgs.size();
      if (TASK_NUM > size)
        TASK_NUM = size;
      const int delta = size / TASK_NUM;
      int from = 0;
      int to = delta;
      for (int i = 0; i < TASK_NUM; i++) {
        from = delta * i;
        if (i == TASK_NUM - 1) {
          to = size;
        } else {
          to = delta * (i + 1);
        }
        auto func = tp_->add([](const vector<cv::Mat>& imgs,
                    const vector<vector<vector<float>>>& boxes,
                    const float threshold,
                    const map<int, string>& label_to_display_name,
                    const vector<string>& img_names, const int inHeight,
                    const int inWidth, const int& from, const int& to,
                    PvanetPostProcessor<Dtype, Qtype>* object) {
          object->writeVisualizeBBox(imgs, boxes, threshold,
              label_to_display_name, img_names, inHeight, inWidth, from, to);
        }, imgs, boxes, FLAGS_confidencethreshold, this->label_to_display_name,
        img_names, height, width, from, to, this);

        futureVector.push_back(std::move(func));
      }
      dumpTimer.log("dump imgs time ...");
    }
    if (futureVector.size() > SimpleInterface::task_threshold) {
      for (int k = 0; k < futureVector.size(); k++) {
        futureVector[k].get();
      }
      futureVector.clear();
    }
    postProcess.log("post process time ...");
  }
  for (int i = 0; i < futureVector.size(); i++) {
    futureVector[i].get();
  }
}

template<typename Dtype, template <typename> class Qtype>
void PvanetOffPostProcessor<Dtype, Qtype>::runSerial() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  int height = infr->h();
  int width = infr->w();
  if (!this->initSerialMode) {
    this->readLabels(&this->label_to_display_name);

    outCpuPtrs_ = new Dtype[infr->outBlobNum()];
    outSyncPtrs_ = new Dtype[infr->outBlobNum()];
    for (int i = 0; i < infr->outBlobNum(); ++i) {
      outCpuPtrs_[i] = new float[infr->outCounts()[i]];
      outSyncPtrs_[i] = new char[infr->outputSizeArray()[i]];
    }
    this->initSerialMode = true;
  }
  TimePoint t1 = std::chrono::high_resolution_clock::now();

  Dtype* mluOutData = infr->popValidOutputData();
  for (int i = 0; i < infr->outBlobNum(); ++i) {
    cnrtDataType cpuDtype = CNRT_FLOAT32;
    cnrtDataType mluDtype = infr->mluOutputDtype()[i];
    CNRT_CHECK(cnrtMemcpy(outSyncPtrs_[i],
                          mluOutData[i],
                          infr->outputSizeArray()[i],
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    vector<int> dimOrderTmp(infr->outDimNums()[i]);
    for (int j = 0; j < infr->outDimNums()[i]; ++j) {
      dimOrderTmp[j] = j;
    }
    vector<int> dimOrder = to_cpu_shape(dimOrderTmp);
    if (cpuDtype != mluDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(outSyncPtrs_[i], mluDtype,
            outCpuPtrs_[i], cpuDtype, nullptr, infr->outDimNums()[i],
            infr->outDimValues()[i], dimOrder.data()));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(outSyncPtrs_[i], mluDtype, outCpuPtrs_[i],
            infr->outDimNums()[i], infr->outDimValues()[i], dimOrder.data()));
    }
  }
  TimePoint t2 = std::chrono::high_resolution_clock::now();
  auto timetrace = infr->popValidOutputTimeTraceData();
  timetrace->out_start = t1;
  timetrace->out_end = t2;
  this->appendTimeTrace(*timetrace);
  infr->pushFreeInputTimeTraceData(timetrace);

  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<vector<float>>> boxes = getResults(&imgs, &img_names);

  if (FLAGS_dump)
    this->writeVisualizeBBox(imgs, boxes, FLAGS_confidencethreshold,
        this->label_to_display_name, img_names, height, width, 0, imgs.size());

  infr->pushFreeOutputData(mluOutData);
}

template<typename Dtype, template <typename> class Qtype>
vector<vector<vector<float>>> PvanetOffPostProcessor<Dtype, Qtype>::getResults(
                                          vector<cv::Mat> *imgs,
                                          vector<string> *img_names) {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  float* resultData = reinterpret_cast<float*>(outCpuPtrs_[1]);
  int roiCount = infr->outCounts()[1] / infr->n();
  vector<vector<vector<float>>> boxes = this->detection_out(
                                     resultData, roiCount, infr->n());

  auto&& origin_img = infr->popValidInputNames();
  for (auto& img_name : origin_img) {
    if (img_name != "null") {
      if (FLAGS_yuv == 0) {
        cv::Mat img = cv::imread(img_name, -1);
        int pos = img_name.find_last_of('/');
        string file_name(img_name.substr(pos + 1));
        imgs->push_back(img);
        img_names->push_back("pvanet_" + file_name);
      } else {
        cv::Mat img = convertYuv2Mat(img_name, infr->w(), infr->h());
        int pos = img_name.find_last_of('/');
        string file_name(img_name.substr(pos + 1));
        imgs->push_back(img);
        img_names->push_back("pvanet_" + file_name);
      }
    }
  }
  return boxes;
}

INSTANTIATE_OFF_CLASS(PvanetOffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
