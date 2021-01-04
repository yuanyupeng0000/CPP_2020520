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
#include <functional>
#include "cnrt.h" // NOLINT
#include "glog/logging.h"
#include "yolov3_off_post.hpp"
#include "off_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::vector;
using std::string;

template<typename Dtype, template <typename> class Qtype>
void Yolov3OffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(infr->deviceId());

  this->readLabels(&this->labels);

  outCpuPtrs_ = new(Dtype);
  outCpuPtrs_[0] = new float[infr->outCounts()[0]];
  size_t outputSize = infr->outputSizeArray()[0];
  syncCpuPtrs_ = new(Dtype);
  syncCpuPtrs_[0] = new char[outputSize];

  int TASK_NUM = SimpleInterface::thread_num;
  std::vector<std::future<void>> futureVector;
  while (true) {
    Dtype* mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) {
      break;  // no more work
    }
    TimePoint t1 = std::chrono::high_resolution_clock::now();
    Timer copyout;
    CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_[0], mluOutData[0],
                          infr->outputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    cnrtDataType_t mluDtype = infr->mluOutputDtype()[0];
    int dim_values[4] = {infr->outNum(), infr->outChannel(),
                         infr->outHeight(), infr->outWidth()};
    int dim_order[4] = {0, 3, 1, 2};
    if (cpuDtype != mluDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
          outCpuPtrs_[0], cpuDtype,
          nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
          outCpuPtrs_[0], cpuDtype,
          nullptr, 4, dim_values, dim_order));
    }
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = infr->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    infr->pushFreeInputTimeTraceData(timetrace);

    copyout.log("copyout time ...");
    infr->pushFreeOutputData(mluOutData);

    Timer postProcess;

    vector<cv::Mat> imgs;
    vector<string> img_names;
    vector<vector<vector<float>>> detections = getResults(&imgs, &img_names);

    Timer dumpTimer;
    if (FLAGS_dump) {
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

        int input_dim = FLAGS_yuv ? 416 : infr->h();
        auto func = this->tp_->add([](const vector<cv::Mat>& imgs,
                    const vector<vector<vector<float>>>& detections,
                    const vector<string>& labelNameMap,
                    const vector<string>& img_names,
                    const int input_dim,
                    const int& from,
                    const int& to,
                    Yolov3PostProcessor<Dtype, Qtype>* object) {
                    object->writeVisualizeBBox(imgs,
                    detections,
                    labelNameMap,
                    img_names,
                    input_dim,
                    from,
                    to);
        }, imgs, detections, this->labels, img_names,
           input_dim, from, to, this);

        futureVector.push_back(std::move(func));
      }
    }
    dumpTimer.log("dump out time ...");
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
void Yolov3OffPostProcessor<Dtype, Qtype>::runSerial() {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);

  if (!this->initSerialMode) {
    this->readLabels(&this->labels);

    outCpuPtrs_ = new(Dtype);
    outCpuPtrs_[0] = new float[infr->outCounts()[0]];
    size_t outputSize = infr->outputSizeArray()[0];
    syncCpuPtrs_ = new (Dtype);
    syncCpuPtrs_[0] = new char[outputSize];

    this->initSerialMode = true;
  }

  Dtype* mluOutData = infr->popValidOutputData();
  TimePoint t1 = std::chrono::high_resolution_clock::now();

  CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_[0], mluOutData[0],
                        infr->outputSizeArray()[0],
                        CNRT_MEM_TRANS_DIR_DEV2HOST));
  cnrtDataType_t cpuDtype = CNRT_FLOAT32;
  cnrtDataType_t mluDtype = infr->mluOutputDtype()[0];
  int dim_values[4] = {infr->outNum(), infr->outChannel(),
                       infr->outHeight(), infr->outWidth()};
  int dim_order[4] = {0, 3, 1, 2};
  if (cpuDtype != mluDtype) {
    CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
       outCpuPtrs_[0], cpuDtype,
       nullptr, 4, dim_values, dim_order));
  } else {
    CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_[0], mluDtype,
       outCpuPtrs_[0], cpuDtype,
       nullptr, 4, dim_values, dim_order));
  }
  TimePoint t2 = std::chrono::high_resolution_clock::now();
  auto timetrace = infr->popValidOutputTimeTraceData();
  timetrace->out_start = t1;
  timetrace->out_end = t2;
  this->appendTimeTrace(*timetrace);
  infr->pushFreeInputTimeTraceData(timetrace);

  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<vector<float>>> detections = getResults(&imgs, &img_names);

  if (FLAGS_dump) {
    int input_dim = FLAGS_yuv ? 416 : infr->h();
    this->writeVisualizeBBox(imgs, detections, this->labels,
                      img_names, input_dim, 0, imgs.size());
  }
  infr->pushFreeOutputData(mluOutData);
}

template<typename Dtype, template <typename> class Qtype>
vector<vector<vector<float>>> Yolov3OffPostProcessor<Dtype, Qtype>::getResults(
                                            vector<cv::Mat> *imgs,
                                            vector<string> *img_names) {
  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  float* outputData = reinterpret_cast<float*>(outCpuPtrs_[0]);
  vector<vector<vector<float>>> detections;

  // BangOp implementation
  float max_limit = 1;
  float min_limit = 0;
  int batchSize = infr->outNum();
  int count = infr->outChannel();
  for (int i = 0; i < batchSize; i++) {
    int num_boxes = static_cast<int>(outputData[i * count]);
    vector<vector<float>> batch_box;
    for (int k = 0; k < num_boxes; k++) {
      int index = i * count + 64 + k * 7;
      vector<float> single_box;
      float bl = std::max(
          min_limit, std::min(max_limit, outputData[index + 3]));  // x1
      float br = std::max(
          min_limit, std::min(max_limit, outputData[index + 5]));  // x2
      float bt = std::max(
          min_limit, std::min(max_limit, outputData[index + 4]));  // y1
      float bb = std::max(
          min_limit, std::min(max_limit, outputData[index + 6]));  // y2
      single_box.push_back(bl);
      single_box.push_back(bt);
      single_box.push_back(br);
      single_box.push_back(bb);
      single_box.push_back(outputData[index + 2]);
      single_box.push_back(outputData[index + 1]);
      if ((br - bl) > 0 && (bb - bt) > 0) {
        batch_box.push_back(single_box);
      }
    }
    detections.push_back(batch_box);
  }

  auto&& origin_img = infr->popValidInputNames();
  for (auto&& img_name : origin_img) {
    if (img_name != "null") {
      cv::Mat img;
      if (FLAGS_yuv) {
        cv::Size size = cv::Size(infr->w(), infr->h());
        img = yuv420sp2Bgr24(convertYuv2Mat(img_name, size));
      } else {
        img = cv::imread(img_name, -1);
      }
      imgs->push_back(img);
      img_names->push_back(img_name);
    }
  }

  return detections;
}

INSTANTIATE_OFF_CLASS(Yolov3OffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
