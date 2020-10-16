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

#include "glog/logging.h"
#ifdef USE_MLU
#include "cnrt.h"
#include "runner.hpp"
#include "pvanet_processor.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"
using std::vector;
using std::pair;

const std::string objects[21] = {"__background__",  // NOLINT
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};

template <typename Dtype, template <typename> class Qtype>
void PvanetPostProcessor<Dtype, Qtype>::writeVisualizeBBox(
                   const vector<cv::Mat>& images,
                   const vector<vector<vector<float>>>& detections,
                   const float threshold,
                   const map<int, string>& labels,
                   const vector<string>& img_names,
                   const int height, const int width,
                   const int from, const int to) {
  // Retrieve detections.
  for (int aa = from; aa < to; aa++) {
    if (img_names[aa] == "null") continue;
    vector<vector<float>> result = detections[aa];
    LOG(INFO) << "image " << img_names[aa] << ": " << result.size()
              << " objects";
    cv::Mat tmp = images[aa];
    cv::Mat img;
    if (FLAGS_yuv == 0) {
      cv::resize(tmp, img, cv::Size(width, height));
    } else {
      img = yuv420sp2Bgr24(tmp);
    }
    std::string name = img_names[aa];
    int pos_map = img_names[aa].rfind("/");
    if (pos_map > 0 && pos_map < img_names[aa].size()) {
      name = name.substr(pos_map + 1);
    }
    pos_map = name.rfind(".");
    if (pos_map > 0 && pos_map < name.size()) {
      name = name.substr(0, pos_map);
    }
    // this is used to cancel "pvanet_" in name
    std::string prefix = "pvanet_";
    name = name.substr(prefix.size());
    name = FLAGS_outputdir + "/" + name + ".txt";
    std::ofstream file_map(name);
    for (int i = 0; i < detections[aa].size(); i++) {
      if (result[i][4] == 0) continue;
      if (result[i][5] < threshold) continue;
      cv::Point p1, p2;
      p1.x = result[i][0];
      p1.y = result[i][1];
      p2.x = result[i][2];
      p2.y = result[i][3];
      cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 0), 8, 8, 0);
      stringstream s0;
      s0 << result[i][5];
      string s00 = s0.str();
      putText(img, objects[static_cast<int>(result[i][4])],
              cv::Point(p1.x, (p1.y + p2.y) / 2 - 10), 2, 0.5,
              cv::Scalar(255, 0, 0), 0, 8, 0);
      putText(img, s00.c_str(), cv::Point(p1.x, (p1.y + p2.y) / 2 + 10), 2, 0.5,
              cv::Scalar(255, 0, 0), 0, 8, 0);
      LOG(INFO) << "detection result: " << result[i][4] << " " << result[i][5]
                << " " << result[i][0] << " " << result[i][1] << " "
                << result[i][2] << " " << result[i][3];
      // the order for mAP is: label score x_min y_min x_max y_max
      file_map << objects[static_cast<int>(result[i][4])] << " "
               << static_cast<float>(result[i][5]) << " "
               << static_cast<float>(p1.x) / img.cols << " "
               << static_cast<float>(p1.y) / img.rows << " "
               << static_cast<float>(p2.x) / img.cols << " "
               << static_cast<float>(p2.y) / img.rows << std::endl;
    }
    if (FLAGS_yuv)
      imwrite((FLAGS_outputdir + "/" + img_names[aa] + ".jpg"), img);
    else
      imwrite((FLAGS_outputdir + "/" + img_names[aa]), img);
    file_map.close();
  }
}

template <typename Dtype, template <typename> class Qtype>
void PvanetPostProcessor<Dtype, Qtype>::readLabels(map<int, string>* label_name_map) {
  if (!FLAGS_labelmapfile.empty()) {
    std::ifstream file(FLAGS_labelmapfile);
    string line;

    vector<int> label_val;
    vector<string> display_value;
    while (getline(file, line)) {
      if (line.find("label:") != std::string::npos) {
        for (int i = 0; i < line.size(); i++) {
          if (isdigit(line[i])) {
            string s1 = line.substr(i, line.size() - i);
            std::stringstream s2(s1);
            int index;
            s2 >> index;
            label_val.push_back(index);
            break;
          }
        }
      }
      if (line.find("display_name:") != std::string::npos) {
        int first_index = line.find_first_of('\"');
        int last_index = line.find_last_of('\"');
        string s1 = line.substr(first_index + 1, last_index - first_index - 1);
        display_value.push_back(s1);
      }
    }
    for (int i = 0; i < label_val.size(); i++)
      label_name_map->insert(pair<int, string>(label_val[i], display_value[i]));
  }
}

template <typename Dtype, template <typename> class Qtype>
vector<vector<vector<float>>> PvanetPostProcessor<Dtype, Qtype>::detection_out(
    float* resultData, int roiCount, int batchsize) {
  vector<vector<vector<float>>> detections(batchsize);
  for (int batchId = 0; batchId < batchsize; batchId++) {
    float* batchData = resultData + batchId * roiCount;
    vector<vector<float>> resultInfo;
    for (int i = 0; i < roiCount / 6; i++) {
      vector<float> tmp(6, 1);
      tmp[0] = batchData[i * 6];
      tmp[1] = batchData[i * 6 + 1];
      tmp[2] = batchData[i * 6 + 2];
      tmp[3] = batchData[i * 6 + 3];
      tmp[4] = batchData[i * 6 + 4];
      tmp[5] = batchData[i * 6 + 5];
      resultInfo.push_back(tmp);
    }
    detections[batchId] = resultInfo;
  }
  return detections;
}

INSTANTIATE_ALL_CLASS(PvanetPostProcessor);
#endif
