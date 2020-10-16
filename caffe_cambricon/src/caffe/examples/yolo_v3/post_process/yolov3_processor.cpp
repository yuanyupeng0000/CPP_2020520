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

#include "glog/logging.h"
#include "cnrt.h" // NOLINT
#include "yolov3_processor.hpp"
#include "runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

template <typename Dtype, template <typename> class Qtype>
void Yolov3PostProcessor<Dtype, Qtype>::writeVisualizeBBox(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName, const vector<string>& imageNames,
    int input_dim, const int from, const int to) {
  // Retrieve detections.
  for (int i = from; i < to; ++i) {
    if (imageNames[i] == "null") continue;
    cv::Mat image;
    image = images[i];
    vector<vector<float>> result = detections[i];
    std::string name = imageNames[i];
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.find(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    string filename = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream fileMap(filename);
    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(images[i].cols),
        static_cast<float>(input_dim) / static_cast<float>(images[i].rows));
    for (int j = 0; j < result.size(); j++) {
      result[j][0] =
          result[j][0] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.cols) / 2.0;
      result[j][2] =
          result[j][2] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.cols) / 2.0;
      result[j][1] =
          result[j][1] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.rows) / 2.0;
      result[j][3] =
          result[j][3] * input_dim -
          static_cast<float>(input_dim - scaling_factors * image.rows) / 2.0;

      for (int k = 0; k < 4; k++) {
        result[j][k] = result[j][k] / scaling_factors;
      }
    }

    for (int j = 0; j < result.size(); j++) {
      result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
      result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
      result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
      result[j][3] = result[j][3] < 0 ? 0 : result[j][3];
      result[j][0] = result[j][0] > image.cols ? image.cols : result[j][0];
      result[j][2] = result[j][2] > image.cols ? image.cols : result[j][2];
      result[j][1] = result[j][1] > image.rows ? image.rows : result[j][1];
      result[j][3] = result[j][3] > image.rows ? image.rows : result[j][3];
    }
    for (int j = 0; j < result.size(); j++) {
      if(result[j][4] < 0.3){continue;}
      int x0 = static_cast<int>(result[j][0]);
      int y0 = static_cast<int>(result[j][1]);
      int x1 = static_cast<int>(result[j][2]);
      int y1 = static_cast<int>(result[j][3]);
      cv::Point p1(x0, y0);
      cv::Point p2(x1, y1);
      cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
      stringstream ss;
      ss << round(result[j][4] * 1000) / 1000.0;
      std::string str =
          labelToDisplayName[static_cast<int>(result[j][5])] + ":" + ss.str();
      cv::Point p5(x0, y0 + 10);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 0, 0), 0.5);

      fileMap << labelToDisplayName[static_cast<int>(result[j][5])] << " "
              << ss.str() << " "
              << static_cast<float>(result[j][0]) / image.cols << " "
              << static_cast<float>(result[j][1]) / image.rows << " "
              << static_cast<float>(result[j][2]) / image.cols << " "
              << static_cast<float>(result[j][3]) / image.rows << " "
              << image.cols << " " << image.rows << std::endl;
    }
    fileMap.close();
    stringstream ss;
    string outFile;
    ss << FLAGS_outputdir << "/yolov3_" << name << ".jpg";
    ss >> outFile;
    cv::imwrite(outFile.c_str(), image);
  }
}

template <typename Dtype, template <typename> class Qtype>
void Yolov3PostProcessor<Dtype, Qtype>::readLabels(vector<string>* labels) {
  if (!FLAGS_labels.empty()) {
    std::ifstream file(FLAGS_labels);
    if (file.fail())
      LOG(FATAL) << "failed to open labels file!";

    std::string line;
    while (getline(file, line)) {
      labels->push_back(line);
    }
    file.close();
  }
}
INSTANTIATE_ALL_CLASS(Yolov3PostProcessor);
