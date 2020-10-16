/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#ifdef USE_MLU
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include "cnrt.h" // NOLINT
#include "common_functions.hpp"

DEFINE_string(offlinemodel, "",
    "The prototxt file used to find net configuration");
DEFINE_string(logdir, "", "path to dump log file, to terminal "
        "stderr by default");
DEFINE_string(images, "", "input file list");

void WrapInputLayer(vector<vector<cv::Mat> >* wrappedImages, float* inputData,
                    int n, int c, int h, int w) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = w;
  int height = h;
  int channels = c;
  for (int i = 0; i < n; ++i) {
    wrappedImages->push_back(vector<cv::Mat> ());
    for (int j = 0; j < channels; ++j) {
      cv::Mat channel(height, width, CV_32FC1, inputData);
      (*wrappedImages)[i].push_back(channel);
      inputData += width * height;
    }
  }
}

void readOneBatch(vector<vector<cv::Mat>>* inImages,
                  queue<string>* imageList) {
  vector<cv::Mat> rawImages;
  string file_id , file;
  cv::Mat prev_image;
  int image_read = 0;

  while (image_read < 1) {
    if (!imageList->empty()) {
      file = file_id = imageList->front();
      imageList->pop();
      if (file.find(" ") != string::npos)
        file = file.substr(0, file.find(" "));

      cv::Mat img;
      img = cv::imread(file, -1);

      if (img.data) {
        ++image_read;
        prev_image = img;
        rawImages.push_back(img);
      } else {
        LOG(INFO) << "failed to read " << file;
      }
    } else {
      if (image_read) {
        cv::Mat img;
        ++image_read;
        prev_image.copyTo(img);
        rawImages.push_back(img);
      } else {
        // if the que is empty and no file has been read, no more runs
        return;
      }
    }
  }

  inImages->push_back(rawImages);
}

void Preprocess(const vector<cv::Mat>& sourceImages, vector<vector<cv::Mat> >* destImages,
                int inChannel, cv::Size inGeometry) {
  /* Convert the input image to the input image format of the network. */
  CHECK(sourceImages.size() == destImages->size())
    << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    cv::Mat sample;
    if (sourceImages[i].channels() == 3 && inChannel == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2GRAY);
    else if (sourceImages[i].channels() == 4 && inChannel == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImages[i].channels() == 4 && inChannel == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2BGR);
    else if (sourceImages[i].channels() == 1 && inChannel == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = sourceImages[i];

    cv::Mat sample_resized;
    if (sample.size() != inGeometry)
      cv::resize(sample, sample_resized, inGeometry);
    else
      sample_resized = sample;
    cv::Mat sample_float;
    if (inChannel == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    cv::split(sample_float, (*destImages)[i]);
  }
}

void offlineRun(const std::string& fileName, const queue<string>& imageList) {
  if (fileName.empty()) {
    LOG(INFO) << "offline model path is NULL";
    return;
  }
  // check imageList
  CHECK(imageList.size() != 0) << "the size of imageList is 0";

  int modelSize = 0;
  cnrtGetModelSize(fileName.c_str(), &modelSize);
  FILE* offlineFp = fopen(fileName.c_str(), "r");
  fseek(offlineFp, 0, SEEK_END);
  int fileSize = ftell(offlineFp);
  int segInfoSize = fileSize - modelSize - 2;  // omit framwork flag
  fseek(offlineFp, modelSize + 2, SEEK_SET);
  LOG(INFO) << "seg info size: " << segInfoSize;
  char* segInfoBuf = new char[segInfoSize];

  if (segInfoSize !=
      fread(segInfoBuf, sizeof(char), segInfoSize, offlineFp)) {
    LOG(INFO) << "Read seg info from file failed!";
    return;
  }

  caffe::SegmentInfo* segInfo = new caffe::SegmentInfo;
  caffe::ReadProtoFromBinaryMem(
      reinterpret_cast<void*>(segInfoBuf), segInfoSize, segInfo);
  caffe::Net<float>* net = new caffe::Net<float>(segInfo->net_proto());
  cnrtDataType_t dtype = CNRT_FLOAT32;

  vector<vector<int>> input_shapes;
  vector<vector<int>> output_shapes;

  for (int k = 0; k < net->input_blobs().size(); k++) {
    vector<int> input_shape;
    input_shape = net->input_blobs()[k]->shape();
    input_shapes.push_back(input_shape);
  }
  for (int k = 0; k < net->output_blobs().size(); k++) {
    vector<int> output_shape;
    output_shape = net->output_blobs()[k]->shape();
    output_shapes.push_back(output_shape);
  }
  net->CopyTrainedLayersFrom(segInfo->net_weights());
  cnrtModel_t model;
  cnrtLoadModel(&model, fileName.c_str());

  int in_n = 1, in_c = 1, in_h = 1, in_w = 1;
  in_n = input_shapes[0][0];
  in_c = input_shapes[0][1];
  in_h = input_shapes[0][2];
  in_w = input_shapes[0][3];

  cv::Size inGeometry = cv::Size(in_w, in_h);
  void** cpuData = new(void*);
  cpuData[0] = new float[in_n * in_c * in_h * in_w];
  queue<string> imgList(imageList);

  int cnt = 0;
  while (!imgList.empty()) {
    vector<vector<cv::Mat>> inImages;
    vector<vector<string>> imageName;
    readOneBatch(&inImages, &imgList);

    vector<vector<cv::Mat> > preprocessedImages;
    WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData[0]),
                   in_n, in_c, in_h, in_w);
    Preprocess(inImages[0], &preprocessedImages, in_c, inGeometry);

    net->OfflineNetRun(*segInfo, model, dtype, cpuData);
    std::vector<caffe::Blob<float>*> outputBlobs = net->output_blobs();
    for (int i = 0; i < outputBlobs.size(); i ++) {
      caffe::Blob<float>* top_blob = outputBlobs[i];
      std::stringstream ss;
      ss << "output" << outputBlobs.size() * cnt + i;
      std::ofstream outputFile(ss.str(), std::ios::out);
      for (int j = 0; j < top_blob->count(); j++) {
        outputFile << top_blob->cpu_data()[j] << std::endl;
      }
      outputFile.close();
    }
    cnt++;
  }
  net->OfflineDestroy();

  delete [] reinterpret_cast<float*>(cpuData[0]);
  delete cpuData;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Do offlineline full run.\n"
        "Usage:\n"
        "    offline_full_run [FLAGS] offlinemodel\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/offline_full_run");
    return 1;
  }

  if (FLAGS_logdir != "") {
      FLAGS_log_dir = FLAGS_logdir;
  } else {
      //  log to terminal's stderr if no log path specified
      FLAGS_alsologtostderr = 1;
  }

  // read input data
  ImageReader img_reader(FLAGS_images);
  auto&& imageList = img_reader.getImageList();

  std::string offlineFile = FLAGS_offlinemodel;
  offlineRun(offlineFile, imageList[0]);

  return 0;
}
#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU
