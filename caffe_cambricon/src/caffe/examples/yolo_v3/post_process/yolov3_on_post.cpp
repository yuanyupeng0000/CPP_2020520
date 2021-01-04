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
#include "yolov3_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "on_data_provider.hpp"
#include "common_functions.hpp"
using std::vector;
using std::string;

template<typename Dtype, template<typename> class Qtype>
vector<vector<vector<float>>> Yolov3OnPostProcessor<Dtype, Qtype>::getResults(
      vector<cv::Mat> *imgs, vector<string>* img_names) {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  vector<vector<vector<float>>> detections;
  float* outputData = reinterpret_cast<float*>(resultDataPtr_);
  float max_limit = 1;
  float min_limit = 0;
  int batchSize = runner->outNum();
  int count = runner->outChannel();
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
  vector<string> origin_img = runner->popValidInputNames();
  for (auto img_name : origin_img) {
    if (img_name != "null") {
      cv::Mat img;
      if (FLAGS_yuv) {
        cv::Size size = cv::Size(runner->w(), runner->h());
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

template <typename Dtype, template <typename> class Qtype>
void Yolov3OnPostProcessor<Dtype, Qtype>::runSerial() {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  if (!this->initSerialMode) {
    this->readLabels(&this->labels);
    this->initSerialMode = true;
  }
  caffe::Net<float>* netBuff = runner->net();
  auto outputBlob = netBuff->output_blobs()[0];
  resultDataPtr_ = outputBlob->mutable_cpu_data();
  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<vector<float>>> detections = getResults(&imgs, &img_names);
  int input_dim = FLAGS_yuv ? 416 : runner->h();
  this->writeVisualizeBBox(imgs, detections, this->labels,
                          img_names, input_dim, 0, imgs.size());
}

template <typename Dtype, template <typename> class Qtype>
void Yolov3OnPostProcessor<Dtype, Qtype>::runParallel() {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  ::setupConfig(this->threadId_, runner->deviceId(), runner->deviceSize());
  this->outCount_ = runner->outCounts()[0];
  this->outN_ = runner->outNum();
  this->readLabels(&this->labels);
  caffe::Net<float>* netBuff = runner->net();
  int outputCount = netBuff->output_blobs()[0]->count();
  outputCpuPtr_ = new Dtype[outputCount];
  size_t tensorSize;
  MLU_CHECK(cnmlGetTensorSize_V2(
      netBuff->output_blobs()[0]->mlu_tensor(), &tensorSize));
  int TASK_NUM = SimpleInterface::thread_num;
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
    runner->pushFreeOutputData(outputMluPtr);
    runner->pushFreeOutputSyncData(outputSyncPtr);
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = runner->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    runner->pushFreeInputTimeTraceData(timetrace);
    Timer dumpTimer;
    if (FLAGS_dump) {
      vector<cv::Mat> imgs;
      vector<string> img_names;
      vector<vector<vector<float>>> final_boxes = getResults(&imgs, &img_names);
      int input_dim = FLAGS_yuv ? 416 : runner->h();
      const int size = imgs.size();
      if (TASK_NUM > size) TASK_NUM = size;
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
        auto func = tp_->add(
            [](const vector<cv::Mat>& imgs,
               const vector<vector<vector<float>>>& final_boxes,
               const vector<string>& labelNameMap,
               const vector<string>& img_names, const int input_dim,
               const int& from, const int& to,
               Yolov3PostProcessor<Dtype, Qtype>* object) {
               object->writeVisualizeBBox(imgs, final_boxes, labelNameMap,
                          img_names, input_dim, from, to);
            },
            imgs, final_boxes, this->labels, img_names, input_dim, from, to, this);

        futureVector.push_back(std::move(func));
      }
    }
    dumpTimer.log("dump imgs time ...");
  }
  for (int i = 0; i < futureVector.size(); i++) {
    futureVector[i].get();
  }
}
INSTANTIATE_ON_CLASS(Yolov3OnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
