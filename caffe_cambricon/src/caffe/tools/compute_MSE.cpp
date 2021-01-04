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
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " file1 file2 ";
    return 1;
  }

  float tmp_val = 0.0;
  std::vector<float> file_A_array;
  std::vector<float> file_B_array;

  std::fstream file_A(argv[1], std::ios::in);
  while (file_A >> tmp_val) {
    file_A_array.push_back(tmp_val);
  }
  file_A.close();

  std::fstream file_B(argv[2], std::ios::in);
  while (file_B >> tmp_val) {
    file_B_array.push_back(tmp_val);
  }
  file_B.close();

  assert(file_A_array.size() == file_B_array.size());

  double sum = 0.0000001, square_sum = 0.0000001, tmp = 0.0000001;
  double  delta = 0.0, delta_sum = 0.0, delta_square_sum = 0.0;
  for (size_t i = 0; i < file_A_array.size(); i++) {
    delta = std::fabs(file_A_array[i] - file_B_array[i]);
    delta_sum += delta;
    delta_square_sum += std::pow(delta, 2);

    tmp = std::fabs(file_A_array[i]);
    sum += tmp;
    square_sum += std::pow(tmp, 2);
  }

  LOG(INFO) << "diff1: " << (delta_sum / sum) * 100 << "%;    "
            << "diff2: " << std::sqrt(delta_square_sum) / std::sqrt(square_sum) * 100
            << "%";

  return 0;
}
