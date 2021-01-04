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
#include <algorithm>
#include <condition_variable>  // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>  // NOLINT
#include <opencv2/highgui/highgui.hpp>  // NOLINT
#include <opencv2/imgproc/imgproc.hpp>  // NOLINT
#include <queue>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "blocking_queue.hpp"
#include "cnrt.h"  // NOLINT
#include "common_functions.hpp"
#include "simple_interface.hpp"
#include "threadPool.h"

using std::map;
using std::pair;
using std::queue;
using std::string;
using std::stringstream;
using std::thread;
using std::vector;

std::condition_variable condition;
std::mutex condition_m;
int start;

DEFINE_string(offlinemodel, "",
                  "The prototxt file used to find net configuration");
DEFINE_string(meanfile, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(
    meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_double(scale, 1, "scale for input data, mobilenet...");
DEFINE_string(outputdir, ".", "The directoy used to save output images");
DEFINE_string(mludevice, "0",
    "set using mlu device number, set multidevice seperated by ','"
    "eg 0,1 when you use device number 0 and 1, default: 0");
DEFINE_string(images, "file_list", "The input file list");
DEFINE_string(labels, "synset_words.txt",
              "infomation about mapping from label to name");
DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
DEFINE_bool(show, false, "set tasks to display results.");
DEFINE_int32(sampling_rate, 1, "set the frame number of video segment.");


void setDeviceId(int dev_id) {
  unsigned devNum;
  CNRT_CHECK(cnrtGetDeviceCount(&devNum));
  if (dev_id >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(dev_id, devNum) << "Valid device count: " << devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_id));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));
}

int getResults(vector<string> imageList, vector<string> labels, float* outputData,
		int batch, int channel){
  int top1 = 0;
	for (int n=0; n<batch; n++) {
		vector<int> index(2, 0);
    vector<float> value(2, 0);
    for (int i = 0; i < channel; i++) {
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

class PostProcessor;

class Inferencer {
  public:
  Inferencer(const cnrtRuntimeContext_t rt_ctx, const int& id);
  ~Inferencer();
  int n() { return inNum_; }
  int c() { return inChannel_; }
  int d() { return inDepth_; }
  int h() { return inHeight_; }
  int w() { return inWidth_; }
  bool simpleFlag() { return simple_flag_; }
  vector<int*> outDimValues() { return outDimValues_; }
  void pushValidInputData(void** data);
  void pushFreeInputData(void** data);
  void** popValidInputData();
  void** popFreeInputData();
  void pushValidOutputData(void** data);
  void pushFreeOutputData(void** data);
  void** popValidOutputData();
  void** popFreeOutputData();
  void pushValidInputNames(vector<string> rawImages);
  void pushValidInputDataAndNames(void** data, const vector<string>& images);
  vector<string> popValidInputNames();
  vector<void*> outCpuPtrs_;
  vector<void*> outSyncPtrs_;
  void simpleRun();
  inline int inBlobNum() { return inBlobNum_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float inferencingTime() { return inferencingTime_; }
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor* p) { postProcessor_ = p; }
  inline int64_t* inputSizeArray() { return inputSizeArray_; }
  inline int64_t* outputSizeArray() { return outputSizeArray_; }
  inline cnrtDataType_t* inputDataTypeArray() { return inputDataTypeArray_; }
  inline cnrtDataType_t* outputDataTypeArray() { return outputDataTypeArray_; }
  void pushFreeInputTimeTraceData(InferenceTimeTrace* data) { freeInputTimetraceFifo_.push(data); }
  void pushValidInputTimeTraceData(InferenceTimeTrace* data) { validInputTimetraceFifo_.push(data); }
  void pushValidOutputTimeTraceData(InferenceTimeTrace* data) { validOutputTimetraceFifo_.push(data); }
  InferenceTimeTrace* popValidInputTimeTraceData() { return validInputTimetraceFifo_.pop(); }
  InferenceTimeTrace* popValidOutputTimeTraceData() { return validOutputTimetraceFifo_.pop(); }
  InferenceTimeTrace* popFreeInputTimeTraceData() { return freeInputTimetraceFifo_.pop(); }

  private:
  void getIODataDesc();

  private:
  BlockingQueue<void**> validInputFifo_;
  BlockingQueue<void**> freeInputFifo_;
  BlockingQueue<void**> validOutputFifo_;
  BlockingQueue<void**> freeOutputFifo_;
  BlockingQueue<vector<string>> imagesFifo_;
  BlockingQueue<InferenceTimeTrace*> freeInputTimetraceFifo_;
  BlockingQueue<InferenceTimeTrace*> validInputTimetraceFifo_;
  BlockingQueue<InferenceTimeTrace*> validOutputTimetraceFifo_;

  cnrtModel_t model_;
  cnrtQueue_t queue_;
  cnrtFunction_t function_;
  cnrtRuntimeContext_t rt_ctx_;
  cnrtDim3_t dim_;

  int64_t* inputSizeArray_;
  int64_t* outputSizeArray_;
  cnrtDataType_t* inputDataTypeArray_;
  cnrtDataType_t* outputDataTypeArray_;
  vector<int> inCounts_, outCounts_;
  vector<int> inDimNums_, outDimNums_;
  vector<int*> inDimValues_, outDimValues_;


  bool simple_flag_;
  int inBlobNum_, outBlobNum_;
  unsigned int inNum_, inChannel_, inDepth_, inHeight_, inWidth_;
  unsigned int outNum_, outChannel_, outHeight_, outWidth_;
  int threadId_;
  int deviceId_;
  int deviceSize_;
  int parallel_ = 1;
  float inferencingTime_;
  PostProcessor* postProcessor_;
  std::mutex infr_mutex_;
};

class PostProcessor {
  public:
  explicit PostProcessor(const int& deviceId)
      : threadId_(0), deviceId_(deviceId), top1_(0) {
    tp_ = new zl::ThreadPool(SimpleInterface::thread_num);
  }
  ~PostProcessor() {
    delete tp_;
  }
  void run();
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setInferencer(Inferencer* p) { inferencer_ = p; }
  inline int top1() { return top1_; }
  void appendTimeTrace(InferenceTimeTrace t) { timeTraces_.push_back(t); }
  virtual std::vector<InferenceTimeTrace> timeTraces() { return timeTraces_; }

  private:
  Inferencer* inferencer_;
  int threadId_;
  int deviceId_;
  int top1_;
  zl::ThreadPool* tp_;
  std::vector<InferenceTimeTrace> timeTraces_;
};

class DataProvider {
  public:
  DataProvider(const string& meanFile, const string& meanValue,
               const int& deviceId, const queue<string>& images)
      : threadId_(0), deviceId_(deviceId), imageList(images) {}
  ~DataProvider() {
    setDeviceId(deviceId_);
    delete [] reinterpret_cast<float*>(cpuData_[0]);
    delete cpuData_;
    if (inputSyncData_ != nullptr) {
      for (int i = 0; i < inferencer_->inBlobNum(); ++i) {
        if (inputSyncData_[i] != nullptr) free(inputSyncData_[i]);
      }
      free(inputSyncData_);
    }
    for (auto ptr : inPtrVector_) {
      for (int i = 0; i < this->inferencer_->inBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
    for (auto ptr : outPtrVector_) {
      for (int i = 0; i < this->inferencer_->outBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
    for(auto ptr : timetracePtrVector_) {
      if(!ptr) {
        free(ptr);
      }
    }
  }
  void run();
  void SetMean(const string&, const string&);
  void WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages);
  void Preprocess(const vector<cv::Mat>& srcImages,
                  vector<vector<cv::Mat>>* dstImages);
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setInferencer(Inferencer* p) {
    inferencer_ = p;
    inNum_ = p->n();
  }
  inline void pushInPtrVector(void** data) { inPtrVector_.push_back(data); }
  inline void pushOutPtrVector(void** data) { outPtrVector_.push_back(data); }
  inline void pushTimeTracePtrVector(InferenceTimeTrace* data) {
      timetracePtrVector_.push_back(data); }

  private:
  int inNum_, inChannel_, inDepth_, inHeight_, inWidth_;
  int threadId_;
  int deviceId_;
  cv::Mat mean_;
  queue<string> imageList;
  Inferencer* inferencer_;
  cv::Size inGeometry_;
  void** cpuData_;
  void** inputSyncData_;
  vector<vector<cv::Mat>> inImages_;
  vector<vector<string>> imageName_;
  vector<void**> inPtrVector_;
  vector<void**> outPtrVector_;
  vector<InferenceTimeTrace*> timetracePtrVector_;
};

void DataProvider::run() {
  setDeviceId(deviceId_);
  for (int i = 0; i < 2; i++) {
    int inputNum = inferencer_->inBlobNum();
    int outputNum = inferencer_->outBlobNum();
    void** inputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    void** outputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));

    // malloc input
    for (int i = 0; i < inputNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(inputMluPtrS[i]), inferencer_->inputSizeArray()[i]));
    }
    for (int i = 0; i < outputNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(outputMluPtrS[i]), inferencer_->outputSizeArray()[i]));
    }
    inferencer_->pushFreeInputData(inputMluPtrS);
    inferencer_->pushFreeOutputData(outputMluPtrS);
    pushInPtrVector(inputMluPtrS);
    pushOutPtrVector(outputMluPtrS);
    // malloc timeStamp
    InferenceTimeTrace* timestamp = reinterpret_cast<InferenceTimeTrace*>(
        malloc(sizeof(InferenceTimeTrace)));
    inferencer_->pushFreeInputTimeTraceData(timestamp);
    pushTimeTracePtrVector(timestamp);
  }

  inNum_ = inferencer_->n();
  inChannel_ = inferencer_->c();
  inDepth_ = inferencer_->d();
  inHeight_ = inferencer_->h();
  inWidth_ = inferencer_->w();
  inGeometry_ = cv::Size(inWidth_, inHeight_);
  SetMean(FLAGS_meanfile, FLAGS_meanvalue);
  cpuData_ = new (void*);
  cpuData_[0] = new float[inNum_ * inChannel_ * inDepth_ * inHeight_ * inWidth_];
  inputSyncData_ =
    reinterpret_cast<void**>(malloc(sizeof(void*) * inferencer_->inBlobNum()));
  inputSyncData_[0] = malloc(inferencer_->inputSizeArray()[0]);

  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [] { return start; });
  lk.unlock();

  while (imageList.size()) {
    vector<cv::Mat> rawImages;
    vector<string> imageNameVec;
    int imageLeft = imageList.size();
    string file = imageList.front();
    for (int i = 0; i < inNum_; i++) {
      if (i < imageLeft) {
        file = imageList.front();
        imageNameVec.push_back(file);
        imageList.pop();
        string file_str = file;
      	if (file.find_last_of("/") != string::npos) {
      	  file_str = file_str.substr(file.find_last_of("/") + 1, file.length());
      	  file_str = file_str.substr(0, file.find_last_of("/"));
      	}
      	LOG(INFO) << "\nclassify for " << file_str;

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
				for (int i = 0; i < inferencer_->d(); i++) {
        	char fn_im[256];
        	sprintf(fn_im, "%s/%06d.jpg", file.c_str(),
        	        id + i * FLAGS_sampling_rate);
					cv::Mat image = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
        	if (!image.data) {
        	  LOG(FATAL) << "Could not open or find file " << fn_im;
        	  break;
        	}
        	rawImages.push_back(image);
      	}
      } else {
        char fn_im[256];
      	sprintf(fn_im, "%s/%06d.jpg", file.c_str(), 1);
      	cv::Mat image = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      	if (!image.data) {
      	  LOG(FATAL) << "Could not open or find file " << fn_im;
      	  break;
      	}
        rawImages.push_back(image);
        imageNameVec.push_back("null");
      }
    }
    Timer prepareInput;
    vector<vector<cv::Mat>> preprocessedImages;
    WrapInputLayer(&preprocessedImages);
    Preprocess(rawImages, &preprocessedImages);
    prepareInput.log("prepare input data ...");

    void** mluData = inferencer_->popFreeInputData();
    Timer copyin;
    cnrtDataType_t mluDtype = inferencer_->inputDataTypeArray()[0];
    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    int dimValuesCpu[5] = {inNum_, inChannel_, inDepth_,
                           inHeight_, inWidth_};
    int dimOrder[5] = {0, 2, 3, 4, 1};  // NCDHW --> NDHWC
    TimePoint t1 = std::chrono::high_resolution_clock::now();
      CNRT_CHECK(cnrtTransOrderAndCast(cpuData_[0],
                                       cpuDtype,
                                       inputSyncData_[0],
                                       mluDtype,
                                       NULL,
                                       5,
                                       dimValuesCpu,
                                       dimOrder));
    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(mluData[0]),
                          inputSyncData_[0],
                          inferencer_->inputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin.log("copyin time ...");
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = inferencer_->popFreeInputTimeTraceData();
    timetrace->in_start = t1;
    timetrace->in_end = t2;
    inferencer_->pushValidInputTimeTraceData(timetrace);

    inferencer_->pushValidInputDataAndNames(mluData, imageNameVec);
  }
  LOG(INFO) << "DataProvider: no data ...";
}

void DataProvider::WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages) {
  int width = inferencer_->w();
  int height = inferencer_->h();
	int depth = inferencer_->d();
  float* inputData = reinterpret_cast<float*>(cpuData_[0]);
	int offset = inferencer_->c() * depth * width * height;
	for (int n = 0; n < inferencer_->n(); ++n) {
    for (int d = 0; d < inferencer_->d(); ++d) {
      (*wrappedImages).push_back(vector<cv::Mat>());
      for (int c = 0; c < inferencer_->c(); ++c) {
        int offset = (c * depth + d) * height * width;
        cv::Mat channel(height, width, CV_32FC1, inputData + offset);
        (*wrappedImages)[n * depth + d].push_back(channel);
      }
    }
    inputData += offset;
  }
}

void DataProvider::Preprocess(const vector<cv::Mat>& sourceImages,
                              vector<vector<cv::Mat>>* destImages) {
  // Convert the input image to the input image format of the network.
  CHECK(sourceImages.size() == destImages->size())
      << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    cv::Mat sample;
    int num_channels_ = inferencer_->c();
    if (sourceImages[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2GRAY);
    else if (sourceImages[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImages[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2BGR);
    else if (sourceImages[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGR);
    else if (sourceImages[i].channels() == 3 && num_channels_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2BGRA);
    else if (sourceImages[i].channels() == 1 && num_channels_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGRA);
    else
      sample = sourceImages[i];
    cv::Size inputGeometry(inferencer_->w(), inferencer_->h());
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

    cv::split(sample_normalized, (*destImages)[i]);
  }
}

void DataProvider::SetMean(const string& meanFile, const string& meanValue) {
  if (FLAGS_meanfile.empty() && FLAGS_meanvalue.empty()) return;
  cv::Scalar channel_mean;
  if (!meanValue.empty()) {
    if (!meanFile.empty()) {
      LOG(INFO) << "Cannot specify mean file";
      LOG(INFO) << " and mean value at the same time; ";
      LOG(INFO) << "Mean value will be specified ";
    }
    stringstream ss(meanValue);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == inChannel_)
        << "Specify either one mean value or as many as channels: "
        << inChannel_;
    vector<cv::Mat> channels;
    for (int i = 0; i < inChannel_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(inGeometry_.height, inGeometry_.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  } else {
    LOG(WARNING) << "Cannot support mean file";
  }
}

Inferencer::Inferencer(const cnrtRuntimeContext_t rt_ctx, const int& id)
    : simple_flag_(true) {
  this->rt_ctx_ = rt_ctx;
  this->threadId_ = id;
  this->inferencingTime_ = 0;

  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_FUNCTION,
                            reinterpret_cast<void**>(&this->function_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_MODEL_PARALLEL,
                            reinterpret_cast<void**>(&this->parallel_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_DEV_ORDINAL,
                            reinterpret_cast<void**>(&this->deviceId_));

  getIODataDesc();
}
// get function's I/O DataDesc,
// allocate I/O data space on CPU memory and prepare Input data;
void Inferencer::getIODataDesc() {
  CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray_,
        &inBlobNum_, function_));
  CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray_,
        &outBlobNum_, function_));
  CNRT_CHECK(cnrtGetInputDataType(&inputDataTypeArray_,
        &inBlobNum_, function_));
  CNRT_CHECK(cnrtGetOutputDataType(&outputDataTypeArray_,
        &outBlobNum_, function_));
  LOG(INFO) << "input blob num is " << inBlobNum_;

  inCounts_.resize(inBlobNum_, 1);
  outCounts_.resize(outBlobNum_, 1);
  inDimNums_.resize(inBlobNum_, 0);
  outDimNums_.resize(outBlobNum_, 0);
  inDimValues_.resize(inBlobNum_, nullptr);
  outDimValues_.resize(outBlobNum_, nullptr);

  for (int i = 0; i < inBlobNum_; i++) {
    CNRT_CHECK(cnrtGetInputDataShape(&(inDimValues_[i]),
        &(inDimNums_[i]), i, function_));
    LOG(INFO) << "input shape " << i << ": ";
    for (int j = 0; j < inDimNums_[i]; ++j) {
      this->inCounts_[i] *= inDimValues_[i][j];
      LOG(INFO) << "shape " << inDimValues_[i][j];
    }
    if (i == 0) {
      inNum_ = inDimValues_[i][0];
      inChannel_ = inDimValues_[i][4];
      inDepth_ = inDimValues_[i][1];
      inWidth_ = inDimValues_[i][2];
      inHeight_ = inDimValues_[i][3];
      LOG(INFO) << "input N:" << inNum_ << " C: " << inChannel_
                << " D: " << inDepth_ << " H: " << inHeight_
                << " W: " << inWidth_;
    }
  }

  for (int i = 0; i < outBlobNum_; i++) {
    CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues_[i]),
        &(outDimNums_[i]), i, function_));
    LOG(INFO) << "output shape " << i << ": ";
    for (int j = 0; j < outDimNums_[i]; ++j) {
      outCounts_[i] *= outDimValues_[i][j];
      LOG(INFO) << "shape " << outDimValues_[i][j];
    }
    if (0 == i) {
      outNum_ = outDimValues_[i][0];
      outChannel_ = outDimValues_[i][3];
      outHeight_ = outDimValues_[i][1];
      outWidth_ = outDimValues_[i][2];
      LOG(INFO) << "output N:" << outNum_ << " C: " << outChannel_
                << " H: " << outHeight_ << " W: " << outWidth_;
    }
    outCpuPtrs_.push_back(malloc(sizeof(float) * outCounts_[i]));
    outSyncPtrs_.push_back(malloc(outputSizeArray_[i]));
  }
}

Inferencer::~Inferencer() {
  setDeviceId(deviceId_);
  for (int i = 0; i < outCpuPtrs_.size(); i++) {
    if (outCpuPtrs_[i] != nullptr) {
      free(outCpuPtrs_[i]);
      outCpuPtrs_[i] = nullptr;
    }
  }
  for (int i = 0; i < outSyncPtrs_.size(); i++) {
    if (outSyncPtrs_[i] != nullptr) {
      free(outSyncPtrs_[i]);
      outSyncPtrs_[i] = nullptr;
    }
  }
}

void** Inferencer::popFreeInputData() { return freeInputFifo_.pop(); }

void** Inferencer::popValidInputData() { return validInputFifo_.pop(); }

void Inferencer::pushFreeInputData(void** data) { freeInputFifo_.push(data); }

void Inferencer::pushValidInputData(void** data) { validInputFifo_.push(data); }

void** Inferencer::popFreeOutputData() { return freeOutputFifo_.pop(); }

void** Inferencer::popValidOutputData() { return validOutputFifo_.pop(); }

void Inferencer::pushFreeOutputData(void** data) { freeOutputFifo_.push(data); }

void Inferencer::pushValidOutputData(void** data) {
  validOutputFifo_.push(data);
}

void Inferencer::pushValidInputNames(vector<string> images) {
  imagesFifo_.push(images);
}

vector<string> Inferencer::popValidInputNames() { return imagesFifo_.pop(); }

void Inferencer::pushValidInputDataAndNames(void** data, const vector<string>& images) {
  std::lock_guard<std::mutex> lk(infr_mutex_);
  pushValidInputData(data);
  pushValidInputNames(images);
}

void Inferencer::simpleRun() {
#define RES_SIZE 1

  // set device to runtime context binded device
  cnrtSetCurrentContextDevice(rt_ctx_);

	cnrtQueue_t queue[RES_SIZE];
  cnrtNotifier_t notifierBeginning[RES_SIZE];
  cnrtNotifier_t notifierEnd[RES_SIZE];

  for (int i = 0; i < RES_SIZE; i++) {
    CHECK(cnrtCreateQueue(&queue[i]) == CNRT_RET_SUCCESS)
        << "CNRT create queue error, thread_id " << threadId();
    cnrtCreateNotifier(&notifierBeginning[i]);
    cnrtCreateNotifier(&notifierEnd[i]);
  }
  float eventInterval[RES_SIZE] = {0};
  void** mluInData[RES_SIZE];
  void** mluOutData[RES_SIZE];

  auto do_pop = [&](int index, void** param) {
    mluInData[index] = popValidInputData();
    if (mluInData[index] == nullptr) return false;
    mluOutData[index] = popFreeOutputData();
    for (int i = 0; i < inBlobNum(); i++) {
      param[i] = mluInData[index][i];
    }
    for (int i = 0; i < outBlobNum(); i++) {
      param[inBlobNum() + i] = mluOutData[index][i];
    }

    return true;
  };

  auto do_invoke = [&](int index, void** param) {
    cnrtPlaceNotifier(notifierBeginning[index], queue[index]);
    CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx_, param, queue[index], nullptr));
  };

  auto do_sync = [&](int index) {
    cnrtPlaceNotifier(notifierEnd[index], queue[index]);
    if (cnrtSyncQueue(queue[index]) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning[index], notifierEnd[index],
                           &eventInterval[index]);
      inferencingTime_ += eventInterval[index];
      printfMluTime(eventInterval[index]);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }
    pushValidOutputData(mluOutData[index]);
    pushFreeInputData(mluInData[index]);
  };

  while (true) {
    void* param[inBlobNum() + outBlobNum()];
    if (do_pop(0, static_cast<void**>(param)) == false) {
      break;
    }
    TimePoint t1 = std::chrono::high_resolution_clock::now();

    do_invoke(0, static_cast<void**>(param));

    do_sync(0);

    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = popValidInputTimeTraceData();
    timetrace->compute_start = t1;
    timetrace->compute_end = t2;
    pushValidOutputTimeTraceData(timetrace);
  }

  for (int i = 0; i < RES_SIZE; i++) {
    cnrtDestroyNotifier(&notifierBeginning[i]);
    cnrtDestroyNotifier(&notifierEnd[i]);
    cnrtDestroyQueue(queue[i]);
  }

  // tell postprocessor to exit
  pushValidOutputData(nullptr);
}


void PostProcessor::run() {
  setDeviceId(deviceId_);
  Inferencer* infr = inferencer_;  // avoid line wrap

	std::vector<string> labels;
  std::ifstream labelsHandler(FLAGS_labels.c_str());
  CHECK(labelsHandler) << "Unable to open labels file " << FLAGS_labels;
  string line;
  while (std::getline(labelsHandler, line)) labels.push_back(line);
  labelsHandler.close();


  std::vector<std::future<void>> futureVector;
  while (true) {
    void** mluOutData = infr->popValidOutputData();
    if (nullptr == mluOutData) break;  // no more data to process
    TimePoint t1 = std::chrono::high_resolution_clock::now();

    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(infr->outSyncPtrs_[0]),
                          mluOutData[0],
                          infr->outputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    cnrtDataType_t mluDtype = infr->outputDataTypeArray()[0];
    int dimValuesMlu[4] = {infr->outDimValues()[0][0], infr->outDimValues()[0][1],
                           infr->outDimValues()[0][2], infr->outDimValues()[0][3]};
    int dimOrder[4] = {0, 3, 1, 2};  // NHWC --> NCHW
    if (cpuDtype != mluDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(infr->outSyncPtrs_[0]),
                                       mluDtype,
                                       reinterpret_cast<void*>(infr->outCpuPtrs_[0]),
                                       cpuDtype,
                                       nullptr,
                                       4,
                                       dimValuesMlu,
                                       dimOrder));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(infr->outSyncPtrs_[0],
                                    mluDtype,
                                    infr->outCpuPtrs_[0],
                                    4,
                                    dimValuesMlu,
                                    dimOrder));
    }
    infr->pushFreeOutputData(mluOutData);
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = infr->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    infr->pushFreeInputTimeTraceData(timetrace);

    Timer dumpTimer;
		vector<string> origin_img = infr->popValidInputNames();
		float* data = reinterpret_cast<float*>(infr->outCpuPtrs_[0]);
		top1_ += getResults(origin_img, labels, data,
			 infr->outDimValues()[0][0], infr->outDimValues()[0][3]);


		dumpTimer.log("dump imgs time ...");
		for (int i = 0; i < futureVector.size(); i++) {
    	futureVector[i].get();
  	}
	}
}

class Pipeline {
  public:
  Pipeline(const string& offlinemodel, const string& meanFile,
           const string& meanValue, const int& id, const int& deviceId,
           const int& devicesize,
           const vector<queue<string>>& images);
  ~Pipeline();
  void run();
  inline DataProvider* dataProvider() { return data_provider_; }
  inline Inferencer* inferencer() { return inferencer_; }
  inline PostProcessor* postProcessor() { return postProcessor_; }

  private:
  vector<DataProvider*> data_providers_;
  DataProvider* data_provider_;
  Inferencer* inferencer_;
  PostProcessor* postProcessor_;
};
Pipeline::Pipeline(const string& offlinemodel, const string& meanFile,
                   const string& meanValue, const int& id, const int& deviceId,
                   const int& devicesize,
                   const vector<queue<string>>& images)
    : data_providers_(SimpleInterface::data_provider_num_),
      data_provider_(nullptr),
      inferencer_(nullptr),
      postProcessor_(nullptr) {
  auto& simpleInterface = SimpleInterface::getInstance();
  auto dev_runtime_contexts = simpleInterface.get_runtime_contexts();
  inferencer_ = new Inferencer(dev_runtime_contexts[deviceId % devicesize][0], id);
  postProcessor_ = new PostProcessor(deviceId);

  postProcessor_->setInferencer(inferencer_);
  postProcessor_->setThreadId(id);
  inferencer_->setPostProcessor(postProcessor_);
  inferencer_->setThreadId(id);

  int data_provider_num = SimpleInterface::data_provider_num_;
  for (int i = 0; i < data_provider_num; i++) {
    data_providers_[i] = new DataProvider(meanFile, meanValue, deviceId,
        images[data_provider_num * id + i]);
    data_providers_[i]->setInferencer(inferencer_);
    data_providers_[i]->setThreadId(id);
  }
}

Pipeline::~Pipeline() {
  for (auto data_provider : data_providers_) {
    delete data_provider;
  }

  if (inferencer_) {
    delete inferencer_;
  }

  if (postProcessor_) {
    delete postProcessor_;
  }
}

void Pipeline::run() {
  int data_provider_num = 1;
  data_provider_num = data_providers_.size();
  vector<thread*> threads(data_provider_num + 2, nullptr);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i] = new thread(&DataProvider::run, data_providers_[i]);
  }
  threads[data_provider_num] =
      new thread(&Inferencer::simpleRun, inferencer_);
  threads[data_provider_num + 1] =
      new thread(&PostProcessor::run, postProcessor_);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i]->join();
    delete threads[i];
  }

  inferencer_->pushValidInputData(nullptr);

  for (int i = 0; i < 2; i++) {
    threads[data_provider_num + i]->join();
    delete threads[data_provider_num + i];
  }
}

int main(int argc, char* argv[]) {
  {
    const char* env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0) FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do classification using c3d network.\n"
      "Usage:\n"
      "    c3d_offline_multicore [FLAGS] \n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(
        argv[0], "examples/yolo_v3/yolov3_offline_multicore");
    return 1;
  }

  auto& simpleInterface = SimpleInterface::getInstance();
  // if simple_compile option has been specified to 1 by user, simple compile
  // thread);
  int provider_num = 1;
  simpleInterface.setFlag(true);
  provider_num = SimpleInterface::data_provider_num_;

  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  std::ifstream files_tmp(FLAGS_images.c_str(), std::ios::in);
  // get device ids
  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }
  int totalThreads = deviceIds_.size();
  int imageNum = 0;
  vector<string> files;
  std::string line_tmp;
  vector<queue<string>> imageList(totalThreads * provider_num);
  if (files_tmp.fail()) {
    LOG(ERROR) << "open " << FLAGS_images << " file fail!";
    return 1;
  } else {
    while (getline(files_tmp, line_tmp)) {
      imageList[imageNum % totalThreads].push(line_tmp);
      imageNum++;
    }
  }
  files_tmp.close();
  LOG(INFO) << "there are " << imageNum << " figures in " << FLAGS_images;

  cnrtInit(0);
  simpleInterface.loadOfflinemodel(FLAGS_offlinemodel, deviceIds_, 1, 1);

  vector<thread*> stageThreads;
  vector<Pipeline*> pipelineVector;
  for (int i = 0; i < totalThreads; i++) {
    Pipeline* pipeline;
    if (imageList.size()) {
      pipeline =
        new Pipeline(FLAGS_offlinemodel, FLAGS_meanfile,
            FLAGS_meanvalue, i, deviceIds_[i % deviceIds_.size()],
            deviceIds_.size(), imageList);
    }

    stageThreads.push_back(new thread(&Pipeline::run, pipeline));
    pipelineVector.push_back(pipeline);
  }

  float execTime;
  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
  {
    std::lock_guard<std::mutex> lk(condition_m);
    LOG(INFO) << "Notify to start ...";
  }
  start = 1;
  condition.notify_all();
  for (int i = 0; i < stageThreads.size(); i++) {
    stageThreads[i]->join();
  }
  gettimeofday(&tpend, NULL);
  execTime = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec -
             tpstart.tv_usec;
  LOG(INFO) << "c3d_detection() execution time: " << execTime << " us";
  float mluTime = 0;
  float top1 = 0;
  for (int i = 0; i < pipelineVector.size(); i++) {
    mluTime += pipelineVector[i]->inferencer()->inferencingTime();
    top1 += pipelineVector[i]->postProcessor()->top1();
  }

  LOG(INFO) << "accuracy1: " << top1 / imageNum << " (" << top1 << "/"
            << imageNum << ")";
  int batchsize = pipelineVector[0]->inferencer()->n();
  std::vector<InferenceTimeTrace> timetraces;
  for (auto iter: pipelineVector) {
    for(auto tc: iter->postProcessor()->timeTraces()) {
      timetraces.push_back(tc);
    }
  }
  printPerfTimeTraces(timetraces, batchsize, mluTime);
  saveResultTimeTrace(timetraces, top1, (-1), (-1), imageNum, batchsize, mluTime);

  for (auto iter : pipelineVector) {
    if (iter != nullptr) {
      delete iter;
    }
  }
  simpleInterface.destroyRuntimeContext();
  cnrtDestroy();
}

#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             << " of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // USE_MLU
