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

#include <glog/logging.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>

DEFINE_string(data_type, "INT8", "Set the quantized data type.");

int main(int argc, char* argv[1]) {
  FLAGS_alsologtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  if (argc != 2) {
    LOG(INFO) << "Usage: " << argv[0] << " 0.in";
    return 1;
  }

  float temp;
  float max = -FLT_MAX;
  float min = FLT_MAX;
  std::ifstream in_f(argv[1]);
  while (in_f >> temp) {
    max = std::max(temp, max);
    min = std::min(temp, min);
  }
  in_f.close();

  float abs_max;
  int position = 0;
  double scale = 1;
  abs_max = std::max(-min, max);

  std::string data_type = FLAGS_data_type;
  int critical_value = std::pow(2, 7) - 1;
  if (data_type == "INT8") {
    critical_value = std::pow(2, 7) - 1;
  } else if (data_type == "INT16") {
    critical_value = std::pow(2, 15) - 1;
  } else {
    LOG(FATAL) << "The specified data type is not supported.";
  }

  if (abs_max == 0) {
    position = 0;
    scale = 1;
  } else {
    position = log2(abs_max / critical_value);
    position += position > 0 ? 1 : 0;
    scale = critical_value * pow(2, static_cast<int>(position)) / abs_max;
  }

  LOG(INFO) << "position: " << position;
  LOG(INFO) << "scale: " << scale;
  return 0;
}
