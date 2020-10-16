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
#include "rfcn_on_post.hpp"
#include "runner.hpp"
#include "on_runner.hpp"
#include "command_option.hpp"
#include "on_data_provider.hpp"
#include "common_functions.hpp"
using std::vector;
using std::string;

template<typename Dtype, template<typename> class Qtype>
vector<vector<float>> RfcnOnPostProcessor<Dtype, Qtype>::getResults(
      vector<cv::Mat> *imgs, vector<string>* img_names) {
  OnRunner<Dtype, Qtype> *runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  float* roiData = reinterpret_cast<float*>(roisDataPtr_);
  float* scoreData = reinterpret_cast<float*>(scoreDataPtr_);
  float* boxData = reinterpret_cast<float*>(boxDataPtr_);
  int batch_size = runner->n();
  int roioffset = roiCount / batch_size;
  int scoreoffset = scoreCount / batch_size;
  int boxoffset = boxCount / batch_size;
  int height = runner->h();
  int width = runner->w();
  vector<vector<float>> boxes = this->detection_out(roiData ,
                                                    scoreData,
                                                    boxData,
                                                    roioffset,
                                                    scoreoffset,
                                                    boxoffset,
                                                    batch_size,
                                                    width,
                                                    height);
  vector<string> origin_img = runner->popValidInputNames();
  for (auto img_name : origin_img) {
    if (img_name != "null") {
        cv::Mat img = cv::imread(img_name, -1);
        int pos = img_name.find_last_of('/');
        string file_name(img_name.substr(pos + 1));
        imgs->push_back(img);
        img_names->push_back(file_name);
    }
  }
  return boxes;
}

template <typename Dtype, template <typename> class Qtype>
void RfcnOnPostProcessor<Dtype, Qtype>::runSerial() {
  OnRunner<Dtype, Qtype> * runner = static_cast<OnRunner<Dtype, Qtype>*>(this->runner_);
  if (!this->initSerialMode) {
    this->readLabels(&this->label_to_display_name);
    this->initSerialMode = true;
  }
  caffe::Net<float>* netBuff = runner->net();
  auto roisBlob = netBuff->blob_by_name("rois");
  auto scoreBlob = netBuff->blob_by_name("cls_prob");
  auto boxBlob = netBuff->blob_by_name("bbox_pred");
  roiCount = roisBlob->count();
  scoreCount = scoreBlob->count();
  boxCount = boxBlob->count();
  roisDataPtr_ = roisBlob->mutable_cpu_data();
  scoreDataPtr_ = scoreBlob->mutable_cpu_data();
  boxDataPtr_ = boxBlob->mutable_cpu_data();
  vector<cv::Mat> imgs;
  vector<string> img_names;
  vector<vector<float>> boxes = getResults(&imgs, &img_names);
  this->WriteVisualizeBBox_offline(imgs, boxes, this->label_to_display_name,
        img_names, runner->h(), runner->w(), 0, imgs.size());
}

INSTANTIATE_ON_CLASS(RfcnOnPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
