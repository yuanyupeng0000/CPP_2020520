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

#include "glog/logging.h"
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#include "segnet_processor.hpp"
#include "runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::pair;

template <typename Dtype, template <typename> class Qtype>
void SegnetProcessor<Dtype, Qtype>::initClassesInfo() {
  vector<string> classNameInfo = {"background", "aeroplane", "bicycle", "bird",
                                  "boat", "bottle", "bus", "car", "cat", "chair",
                                  "cow", "diningtable", "dog", "horse", "motobike",
                                  "person", "pottedplant", "sheep", "sofa", "train",
                                  "tvmonitor" };
  vector<vector<int>> rgbInfo = {{0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
                                 {0, 0, 128}, {128, 0, 128}, {0, 128, 128},
                                 {128, 128, 128}, {64, 0, 0}, {192, 0, 0},
                                 {64, 128, 0}, {192, 128, 0}, {64, 0, 128},
                                 {192, 0, 128}, {64, 128, 128}, {192, 128, 128},
                                 {0, 64, 0}, {128, 64, 0}, {0, 192, 0}, {128, 192, 0},
                                 {0, 64, 128}};
  for (int i = 0; i < 21; i++) {
    classInfoMap.insert(pair<int, string>(i, classNameInfo[i]));
  }
  for (int i = 0; i < 21; i++) {
    classRGBLabelVector.push_back(rgbInfo[i]);
  }
}
INSTANTIATE_ALL_CLASS(SegnetProcessor);
#endif
