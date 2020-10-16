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
#include <stdio.h>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "common_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::vector;
using std::string;
using std::queue;

DEFINE_string(model, "", "The prototxt file used to find net configuration");
DEFINE_string(weights, "", "The binary file used to set net parameter");
DEFINE_string(meanfile, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(
    meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_double(scale, 1, "scale for input data, mobilenet...");
DEFINE_string(mmode, "MFUS", "CPU, MLU or MFUS, MFUS mode");
DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");
DEFINE_string(images, "file_list", "The input file list");
DEFINE_string(labels, "synset_words.txt",
              "infomation about mapping from label to name");
DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
DEFINE_bool(show, false, "set tasks to display results.");
DEFINE_int32(sampling_rate, 1, "set the frame number of video segment.");
DEFINE_string(mcore, "MLU270", "Which core version you want to generate,"
              "only support MLU270 and MLU220.");
DEFINE_int32(batchsize, 1, "Read images size every batch for inference");
DEFINE_int32(core_number, 4, "The number of cores are used for inference");

class Detector {
 public:
  Detector(const string& modelFile, const string& weightsFile,
           const string& meanFile, const string& meanValues);
  ~Detector();
  void detect(const vector<cv::Mat>& images);
  vector<int> inputShape;
  int getBatchSize() { return batchSize; }
  void readImages(queue<string>* imagesQueue, int inputNumber,
                  vector<cv::Mat>* images, vector<string>* imageNames);
  int getResults(vector<string> imageNames, vector<string> labels);
  inline float runTime() { return runTime_; }

 private:
  void setMean(const string& meanFile, const string& meanValues);
  void wrapInputLayer(vector<vector<cv::Mat>>* inputImages);
  void preProcess(const vector<cv::Mat>& images,
                  vector<vector<cv::Mat>>* inputImages);

  Net<float>* network;
  cv::Size inputGeometry;
  int batchSize;
  int numberChannels;
  int depth;
  int inputNum, outputNum;
  cv::Mat mean_;
  float runTime_;
};

Detector::Detector(const string& modelFile, const string& weightsFile,
                   const string& meanFile, const string& meanValues):
    runTime_(0) {
  /* Load the network. */
  network = new Net<float>(modelFile, TEST);
  network->CopyTrainedLayersFrom(weightsFile);
  network->Forward();

  outputNum = network->num_outputs();
  Blob<float>* inputLayer = network->input_blobs()[0];
  batchSize = inputLayer->num();
  numberChannels = inputLayer->channels();
  inputShape = inputLayer->shape();
  depth = inputShape[2];
  inputGeometry = cv::Size(inputShape[4], inputShape[3]);

  LOG(INFO) << "input shape: ";
  LOG(INFO) << "num: " << inputShape[0];
  LOG(INFO) << "channels: " << inputShape[1];
  LOG(INFO) << "depth: " << inputShape[2];
  LOG(INFO) << "height: " << inputShape[3];
  LOG(INFO) << "width: " << inputShape[4];
  CHECK(numberChannels == 3 || numberChannels == 1)
      << "Input layer should have 1 or 3 channels.";
  /* Load the binaryproto mean file. */
  setMean(meanFile, meanValues);
}

Detector::~Detector() { delete network; }

void Detector::detect(const vector<cv::Mat>& images) {
  vector<vector<cv::Mat>> inputImages;
  wrapInputLayer(&inputImages);
  preProcess(images, &inputImages);

  float eventTimeUse;
  cnrtNotifier_t notifierBeginning, notifierEnd;
  if (FLAGS_mmode != "CPU") {
    cnrtCreateNotifier(&notifierBeginning);
    cnrtCreateNotifier(&notifierEnd);
    cnrtPlaceNotifier(notifierBeginning, caffe::Caffe::queue());
  }

  network->Forward();

  if (FLAGS_mmode != "CPU") {
    cnrtPlaceNotifier(notifierEnd, caffe::Caffe::queue());
    cnrtSyncQueue(caffe::Caffe::queue());
    cnrtNotifierDuration(notifierBeginning, notifierEnd, &eventTimeUse);
    this->runTime_ +=  eventTimeUse;
    printfMluTime(eventTimeUse);
  }
}

/* Load the mean file in binaryproto format. */
void Detector::setMean(const string& meanFile, const string& meanValues) {
  cv::Scalar channel_mean;
  if (!meanValues.empty()) {
    if (!meanFile.empty()) {
      LOG(INFO) << "Cannot specify mean file";
      LOG(INFO) << " and mean value at the same time; ";
      LOG(INFO) << "Mean value will be specified ";
    }
    stringstream ss(meanValues);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == numberChannels)
        << "Specify either one mean value or as many as channels: "
        << numberChannels;
    vector<cv::Mat> channels;
    for (int i = 0; i < numberChannels; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(inputGeometry.height, inputGeometry.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  } else {
    LOG(INFO) << "Mean value should be specified ";
  }
}

void Detector::wrapInputLayer(vector<vector<cv::Mat>>* inputImages) {
  int width = inputGeometry.width;
  int height = inputGeometry.height;

  Blob<float>* inputLayer = network->input_blobs()[0];
  float* inputData = inputLayer->mutable_cpu_data();
  for (int n = 0; n < batchSize; ++n) {
    for (int d = 0; d < depth; ++d) {
      (*inputImages).push_back(vector<cv::Mat>());
      for (int c = 0; c < numberChannels; ++c) {
        int offset = (c * depth + d) * height * width;
        cv::Mat channel(height, width, CV_32FC1, inputData + offset);
        (*inputImages)[n * depth + d].push_back(channel);
      }
    }
    inputData += inputLayer->count(1);
  }
}

void Detector::preProcess(const vector<cv::Mat>& images,
                          vector<vector<cv::Mat>>* inputImages) {
  CHECK(images.size() == inputImages->size())
      << "Size of imgs and input_imgs doesn't match";
  for (int i = 0; i < images.size(); ++i) {
    cv::Mat sample;
    int num_channels_ = inputShape[1];
    if (images[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGR2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2BGR);
    else if (images[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = images[i];

    cv::Mat sample_resized;
    cv::Rect select;
    cv::Mat sample_select;
    cv::resize(sample, sample_resized, cv::Size(171, 128));
    int left_x = (171 - inputGeometry.width) / 2;
    int top_y = (128 - inputGeometry.height) / 2;
    select = cv::Rect(cv::Point(left_x, top_y), inputGeometry);
    sample_select = sample_resized(select);

    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample_select.convertTo(sample_float, CV_32FC3);
    else
      sample_select.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    if ((FLAGS_meanvalue.empty() && FLAGS_meanfile.empty())) {
      sample_normalized = sample_float;
    } else {
      cv::subtract(sample_float, mean_, sample_normalized);
      if (FLAGS_scale != 1) {
        sample_normalized *= FLAGS_scale;
      }
    }
    cv::split(sample_normalized, (*inputImages)[i]);
  }
}

void Detector::readImages(queue<string>* imagesQueue, int inputNumber,
                          vector<cv::Mat>* images, vector<string>* imageNames) {
  int leftNumber = imagesQueue->size();
  string file = imagesQueue->front();

  for (int i = 0; i < inputNumber; i++) {
    if (i < leftNumber) {
      imageNames->push_back(file);
      file = imagesQueue->front();
      imagesQueue->pop();
      string file_str = file;
      if (file.find_last_of("/") != string::npos) {
        file_str = file_str.substr(file.find_last_of("/") + 1, file.length());
        file_str = file_str.substr(0, file.find_last_of("/"));
      }
      LOG(INFO) << "\nclassify for " << file_str;
      vector<cv::Mat> batchImage;
      int id;
      string label_str;
      if (file.find(" ") != string::npos) {
        label_str = file.substr(file.find(" ") + 1, file.length());
        file = file.substr(0, file.find(" "));
      }
      if (label_str.find(" ") != string::npos) {
        string id_str = label_str.substr(0, label_str.find(" "));
        id = atoi(id_str.c_str());
      }
      for (int i = 0; i < inputShape[2]; i++) {
        char fn_im[256];
        sprintf(fn_im, "%s/%06d.jpg", file.c_str(),
                id + i * FLAGS_sampling_rate);
        cv::Mat image = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
        if (!image.data) {
          LOG(FATAL) << "Could not open or find file " << fn_im;
          break;
        }
        images->push_back(image);
      }
    } else {
      imageNames->push_back(file);
      vector<cv::Mat> batchImage;
      char fn_im[256];
      sprintf(fn_im, "%s/%06d.jpg", file.c_str(), 1);
      cv::Mat image = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      if (!image.data) {
        LOG(FATAL) << "Could not open or find file " << fn_im;
        break;
      }
      images->push_back(image);
    }
  }
}

int Detector::getResults(vector<string> imageList, vector<string> labels) {
  Blob<float>* outputLayer = network->output_blobs()[0];
  const float* outputData = outputLayer->cpu_data();
  int top1 = 0;
  // output
  for (int n = 0; n < batchSize; n++) {
    vector<int> index(2, 0);
    vector<float> value(2, 0);
    for (int i = 0; i < outputLayer->count(1); i++) {
      float tmp_data = outputData[i];
      int tmp_index = i;
      for (int j = 0; j < 2; j++) {
        if (outputData[i] > value[j]) {
          std::swap(value[j], tmp_data);
          std::swap(index[j], tmp_index);
        }
      }
    }
    // printf
    std::stringstream stream;
    for (int k = 0; k < 2; k++) {
      stream << std::fixed << std::setprecision(4) << value[k] << " - "
             << index[k] << ": " << labels[index[k]] << std::endl;
    }
    LOG(INFO) << stream.str();

    // get accuracy
    string label_str;
    if (imageList[n].find_last_of(" ") != string::npos) {
      label_str = imageList[n].substr(imageList[n].find_last_of(" ") + 1);
    }
    stringstream ss(label_str);
    string item;
    bool has_top1 = false;
    while (getline(ss, item, ',')) {
      int label = std::atoi(item.c_str());
      if (label == index[0] && !has_top1) {
        top1++;
        has_top1 = true;
      }
    }
  }
  return top1;
}

int main(int argc, char** argv) {
  {
    const char* env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0) FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetStderrLogging(google::GLOG_WARNING);
  FLAGS_colorlogtostderr = true;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do classification using c3d network.\n"
      "Usage:\n c3d_online_multicore [FLAGS] \n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
      "examples/C3D/c3d_online_multicore");
    return 1;
  }
  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    FLAGS_alsologtostderr = 1;
  }

  if (FLAGS_mmode == "CPU") {
    Caffe::set_mode(Caffe::CPU);
  } else {
#ifdef USE_MLU
    cnmlInit(0);
    if (FLAGS_mcore != "MLU270" && FLAGS_mcore != "MLU220") {
	    LOG(FATAL) << "C3D only support MLU270 and MLU220.";
    }
    Caffe::set_rt_core(FLAGS_mcore);
    Caffe::set_mlu_device(FLAGS_mludevice);
    Caffe::set_mode(FLAGS_mmode);
    Caffe::setReshapeMode(Caffe::ReshapeMode::SETUPONLY);
    Caffe::setBatchsize(FLAGS_batchsize);
    Caffe::setCoreNumber(FLAGS_core_number);
    Caffe::setSimpleFlag(true);
#else
    LOG(FATAL) << "No other available modes, please recompile with USE_MLU!";
#endif
  }

  /* Create Detector class */
  Detector* detector =
      new Detector(FLAGS_model, FLAGS_weights, FLAGS_meanfile, FLAGS_meanvalue);

  /* Load labels*/
  std::vector<string> labels;
  std::ifstream labelsHandler(FLAGS_labels.c_str());
  CHECK(labelsHandler) << "Unable to open labels file " << FLAGS_labels;
  string line;
  while (std::getline(labelsHandler, line)) labels.push_back(line);
  labelsHandler.close();

  /* Load image files */
  queue<string> imageListQueue;
  int figuresNumber = 0;
  string lineTemp;
  std::ifstream filesHandler(FLAGS_images.c_str(), std::ios::in);
  CHECK(!filesHandler.fail()) << "Image file is invalid!";
  while (getline(filesHandler, lineTemp)) {
    imageListQueue.push(lineTemp);
    figuresNumber++;
  }
  filesHandler.close();
  LOG(INFO) << "there are " << figuresNumber << " figures in " << FLAGS_images;

  /* Detecting images */
  float timeUse;
  float totalTime = 0;
  int totalLatency = 0;
  struct timeval tpStart, tpEnd;
  int batchesNumber =
      ceil(static_cast<float>(figuresNumber) / detector->getBatchSize());
  float top1 = 0;
  for (int i = 0; i < batchesNumber; i++) {
    gettimeofday(&tpStart, NULL);
    vector<cv::Mat> images;
    vector<string> imageNames;
    /* Firstly read images from file list */
    detector->readImages(&imageListQueue, detector->getBatchSize(), &images,
                         &imageNames);

    TimePoint t1 = std::chrono::high_resolution_clock::now();

    /* Secondly fill images into input blob and do net forwarding */
    detector->detect(images);

    TimePoint t2 = std::chrono::high_resolution_clock::now();
    totalLatency += std::chrono::duration_cast<TimeDuration_us>(t2 - t1).count();

    top1 += detector->getResults(imageNames, labels);
    gettimeofday(&tpEnd, NULL);
    timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
              tpStart.tv_usec;
    totalTime += timeUse;
    LOG(INFO) << "Task execution time: " << timeUse << " us ";
    images.clear();
  }
  LOG(INFO) << "Total execution time: " << totalTime << " us";
  LOG(INFO) << "Global accuracy: ";
  LOG(INFO) << "accuracy1: " << top1 / figuresNumber << " (" << top1 << "/"
            << figuresNumber << ")";
#ifdef USE_MLU
  if (FLAGS_mmode != "CPU") {
    float throughput = figuresNumber * 1e6 / totalTime;
    int ave_latency = totalLatency * detector->getBatchSize() / figuresNumber;
    LOG(INFO) << "Latency: " << ave_latency /  1000;
    LOG(INFO) << "Throughput: " << throughput;
    dumpJson(figuresNumber, top1, (-1), (-1), ave_latency, throughput);
    Caffe::freeQueue();
    cnmlExit();
  }
#endif
  delete detector;
  return 0;
}
#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU  && USE OPENCV
