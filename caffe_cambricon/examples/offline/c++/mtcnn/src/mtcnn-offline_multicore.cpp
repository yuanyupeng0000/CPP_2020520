/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <syscall.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <mutex> // NOLINT
#include <string>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <map>
#include <numeric>
#include <sstream>
#include <functional>

#include "cnrt.h" // NOLINT
#include "common_functions.hpp"
#define DBG printf
#define ENABLE_TIMEING 0

#undef CNRT_CHECK
#define CNRT_CHECK(condition)                                   \
  do {                                                          \
    cnrtRet_t status = condition;                               \
    if (status != CNRT_RET_SUCCESS)                             \
      printf("%p [%s:%d]%s %s\n", this, __FUNCTION__, __LINE__, \
             cnrtGetErrorStr(status), #condition);              \
  } while (0)

using std::string;
using std::vector;
using std::stringstream;
using std::mutex;
using std::thread;
using std::map;

using cv::Mat;
using cv::Size;
using cv::Rect;
DEFINE_string(images, "", "The input file list");
DEFINE_string(models, "", "infomation about mapping from label to name");
DEFINE_int32(int8, 1, "fp16 or int8 mode, default is fp16(0)");
DEFINE_int32(first_conv, 1, "Use first_conv on preprocess");
DEFINE_string(mludevice, "0",
            "set using mlu device number, default: 0");
DEFINE_int32(threads, 1, "thread number");
DEFINE_string(mcore, "MLU100", "mcore");
DEFINE_int32(batchsize, 1, "batch size");
DEFINE_int32(core_number, 1, "core number");

class FaceDetector {
  public:
  enum NMS_TYPE {
    MIN,
    UNION,
  };
  struct BoundingBox {
    // rect two points
    float x1, y1;
    float x2, y2;
    // regression
    float dx1, dy1;
    float dx2, dy2;
    // cls
    float score;
    // inner points
    float points_x[5];
    float points_y[5];
  };

  struct CmpBoundingBox {
    bool operator()(const BoundingBox &b1, const BoundingBox &b2) {
      if (b1.score == b2.score)
        return b1.x1 == b2.x1 ? b1.y1 > b2.y1 : b1.x1 > b2.x1;
      else
        return b1.score > b2.score;
    }
  };

  public:
  static bool Init();
  static bool Destory();
  float GetRuntime();
  int GetLatency();
  void AddRuntime(float elapseTime);
  void AddLatency(int time);
  bool Open();
  FaceDetector(int nDevNum, int nDevChannel);
  ~FaceDetector();
  void prepareRuntimeContext(cnrtRuntimeContext_t *ctx,
               cnrtFunction_t function, int deviceId,
               const bool& channel_dup, uint64_t stack_size);
  void loadOfflinemodel(const string& offlinemodel,
       const vector<int>& deviceIds, const bool& channel_dup,
       cnrtModel_t* model, cnrtRuntimeContext_t* ctx);
  void initialPnet(vector<string> pnet_model_path);
  void initialRnet(string rnet_model_path);
  void initialOnet(string onet_model_path);
  void wrapInputLayer(const vector<int> &input_shape, float *input_dataBuffer,
                      vector<cv::Mat> *input_channels);
  void pyrDown(const vector<cv::Mat> &img,
               std::vector<cv::Mat> *input_channels);
  void buildInputChannels(const std::vector<cv::Mat> &img_channels,
                          const std::vector<BoundingBox> &boxes,
                          const cv::Size &target_size,
                          std::vector<cv::Mat> *input_channels);
  void getPnetSoftmax(float *x, int row, int column);
  void getROnetSoftmax(float *x, int n, int out_c, int out_h, int out_w);
  void generateBoundingBox(const vector<float> &boxRegs,
                           const vector<int> &box_shape,
                           const vector<float> &cls,
                           const vector<int> &cls_shape, float scale_w, float scale_h,
                           const float threshold,
                           vector<BoundingBox> *filterOutBoxes);
  void filteroutBoundingBox(const vector<BoundingBox> &boxes,
                            const vector<float> &boxRegs,
                            const vector<int> &box_shape,
                            const vector<float> &cls,
                            const vector<int> &cls_shape,
                            const vector<float> &points,
                            const vector<int> &points_shape, float threshold,
                            vector<BoundingBox> *filterOutBoxes);
  void nms(vector<BoundingBox> *boxes, float threshold, NMS_TYPE type,
           vector<BoundingBox> *filterOutBoxes);
  void nmsGlobal();
  void doPnet(int streamid);
  void doRnet();
  void doOnet();
  vector<BoundingBox> detect(const cv::Mat &img, int minsize, float p_thres,
                             float r_thres, float o_thres);

  private:
  float img_mean;
  float img_var;
  int pnet_task_num;
  float P_thres;
  float R_thres;
  float O_thres;
  int min_size;
  int img_H;
  int img_W;
  mutex mtx;
  vector<BoundingBox> totalBoxes;
  vector<cv::Mat> sample_norm_channels;
  vector<vector<int>> m_idxs;

  int m_nDevNum;
  int m_nDevChannel;
  bool m_running;

  float runTime_;
  int latency_;

  cnrtDim3_t dim;
  cnrtQueue_t queue_;

  sem_t sem_in[10];
  sem_t sem_out[10];
  vector<thread *> tasks;

  // Pnet
  vector<cnrtModel_t> pnet_cnrt_model;
  vector<cnrtFunction_t> pnet_function;
  vector<vector<int>> pnet_in_shape;
  vector<vector<int>> pnet_cls_shape;
  vector<vector<int>> pnet_box_shape;
  vector<vector<int>> pnet_in_size;
  vector<vector<int>> pnet_out_size;
  vector<vector<float *>> pnet_input_cpu;
  vector<vector<float *>> pnet_output_cpu;
  vector<vector<void *>> pnet_input_temp_cpu;
  vector<vector<void *>> pnet_output_temp_cpu;
  vector<void **> pnet_param;
  vector<void **> pnet_inputCpuPtrS;
  vector<void **> pnet_outputCpuPtrS;
  vector<void **> pnet_inputCpuTempPtrS;
  vector<void **> pnet_outputCpuTempPtrS;
  vector<void **> pnet_inputMluPtrS;
  vector<void **> pnet_outputMluPtrS;
  vector<int64_t *> pnet_inputSizeS;
  vector<int64_t *> pnet_outputSizeS;
  vector<int> pnet_inputNum;
  vector<int> pnet_outputNum;
  vector<cnrtRuntimeContext_t> pnet_runtime_context;

  // Rnet
  cnrtModel_t rnet_cnrt_model;
  cnrtFunction_t rnet_function;
  vector<int> rnet_in_shape;
  vector<int> rnet_cls_shape;
  vector<int> rnet_box_shape;
  vector<int> rnet_in_size;
  vector<int> rnet_out_size;
  vector<float *> rnet_input_cpu;
  vector<float *> rnet_output_cpu;
  vector<void *> rnet_input_temp_cpu;
  vector<void *> rnet_output_temp_cpu;
  void **rnet_param;
  void **rnet_inputCpuPtrS;
  void **rnet_outputCpuPtrS;
  void **rnet_inputCpuTempPtrS;
  void **rnet_outputCpuTempPtrS;
  void **rnet_inputMluPtrS;
  void **rnet_outputMluPtrS;
  int rnet_inputNum;
  int rnet_outputNum;
  int64_t* rnet_inputSizeS;
  int64_t* rnet_outputSizeS;
  cnrtRuntimeContext_t rnet_runtime_context;

  // Onet
  cnrtModel_t onet_cnrt_model;
  cnrtFunction_t onet_function;
  vector<int> onet_in_shape;
  vector<int> onet_cls_shape;
  vector<int> onet_box_shape;
  vector<int> onet_points_shape;
  vector<int> onet_in_size;
  vector<int> onet_out_size;
  vector<float *> onet_input_cpu;
  vector<float *> onet_output_cpu;
  vector<void *> onet_input_temp_cpu;
  vector<void *> onet_output_temp_cpu;
  void **onet_param;
  void **onet_inputCpuPtrS;
  void **onet_outputCpuPtrS;
  void **onet_inputCpuTempPtrS;
  void **onet_outputCpuTempPtrS;
  void **onet_inputMluPtrS;
  void **onet_outputMluPtrS;
  int onet_inputNum;
  int onet_outputNum;
  int64_t* onet_inputSizeS;
  int64_t* onet_outputSizeS;
  cnrtRuntimeContext_t onet_runtime_context;
};


FaceDetector::FaceDetector(int nDevNum, int nDevChannel) {
  m_running = false;
  runTime_ = 0;
  latency_ = 0;

  if (FLAGS_first_conv) {
    img_mean = 0;
    img_var = 1;
  } else {
    img_mean = 127.5;
    img_var = 0.0078125;
  }
  m_nDevNum = nDevNum;
  m_nDevChannel = nDevChannel;
  pnet_task_num = 1;
}

bool FaceDetector::Open() {
  dim = {1, 1, 1};
  vector<string> pnet_model_path;
  string rnet_model_path;
  string onet_model_path;

  std::ifstream file(FLAGS_models);
  if (file.fail()) LOG(FATAL) << "failed to open model file!";

  std::string line;
  while (getline(file, line)) {
    int pos = line.find(" ");
    if (line.substr(0, pos) == "pnet")
      pnet_model_path.push_back(line.substr(pos + 1));
    if (line.substr(0, pos) == "rnet") rnet_model_path = line.substr(pos + 1);
    if (line.substr(0, pos) == "onet") onet_model_path = line.substr(pos + 1);
  }
  file.close();

  unsigned dev_num;
  CNRT_CHECK(cnrtGetDeviceCount(&dev_num));
  if (dev_num == 0 || m_nDevNum >= dev_num) return false;

  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, m_nDevNum));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  CNRT_CHECK(cnrtCreateQueue(&queue_));

  initialPnet(pnet_model_path);
  initialRnet(const_cast<char *>(rnet_model_path.c_str()));
  initialOnet(const_cast<char *>(onet_model_path.c_str()));

  tasks.resize(pnet_task_num);
  m_idxs.resize(pnet_task_num);

  m_running = true;
  for (int i = 0; i < 10; i++) {
    sem_init(&sem_in[i], 0, 0);
    sem_init(&sem_out[i], 0, 0);
    m_idxs[i % pnet_task_num].push_back(i);
  }
  for (int i = 0; i < pnet_task_num; i++) {
    tasks[i] = new thread(std::bind(&FaceDetector::doPnet, this, i));
  }
  return true;
}

bool FaceDetector::Init() { return (CNRT_RET_SUCCESS == cnrtInit(0)); }
bool FaceDetector::Destory() {
  cnrtDestroy();
  return true;
}

void FaceDetector::AddRuntime(float elapseTime) {
  mtx.lock();
  runTime_ += elapseTime;
  mtx.unlock();
}

void FaceDetector::AddLatency(int count_ms) {
  mtx.lock();
  latency_ += count_ms;
  mtx.unlock();
}

float FaceDetector::GetRuntime() {
  return runTime_;
}

int FaceDetector::GetLatency() {
  return latency_;
}

FaceDetector::~FaceDetector() {
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, m_nDevNum));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  m_running = false;

  for (int i = 0; i < pnet_task_num; i++) {
    tasks[i]->join();
  }

  for (int i = 0; i < 10; i++) {
    sem_destroy(&sem_in[i]);
    sem_destroy(&sem_out[i]);
  }

  CNRT_CHECK(cnrtDestroyQueue(queue_));
  // pNet
  for (int i = 0; i < pnet_cnrt_model.size(); i++) {
    CNRT_CHECK(cnrtDestroyFunction(pnet_function[i]));
    CNRT_CHECK(cnrtUnloadModel(pnet_cnrt_model[i]));

    if (pnet_param[i]) free(pnet_param[i]);
    pnet_param[i] = NULL;

    if (pnet_inputCpuPtrS[i]) free(pnet_inputCpuPtrS[i]);
    pnet_inputCpuPtrS[i] = NULL;
    if (pnet_inputCpuTempPtrS[i]) free(pnet_inputCpuTempPtrS[i]);
    pnet_inputCpuTempPtrS[i] = NULL;

    if (pnet_outputCpuPtrS[i]) free(pnet_outputCpuPtrS[i]);
    pnet_outputCpuPtrS[i] = NULL;

    if (pnet_outputCpuTempPtrS[i]) free(pnet_outputCpuTempPtrS[i]);
    pnet_outputCpuTempPtrS[i] = NULL;

    if (pnet_inputMluPtrS[i])
      CNRT_CHECK(cnrtFreeArray(pnet_inputMluPtrS[i], pnet_inputNum[i]));
    pnet_inputMluPtrS[i] = NULL;

    if (pnet_outputMluPtrS[i])
      CNRT_CHECK(cnrtFreeArray(pnet_outputMluPtrS[i], pnet_outputNum[i]));
    pnet_outputMluPtrS[i] = NULL;

    for (auto tmp : pnet_input_cpu[i]) {
      free(tmp);
      tmp = NULL;
    }
    for (auto tmp : pnet_input_temp_cpu[i]) {
      free(tmp);
      tmp = NULL;
    }
    for (auto tmp : pnet_output_cpu[i]) {
      free(tmp);
      tmp = NULL;
    }
    for (auto tmp : pnet_output_temp_cpu[i]) {
      free(tmp);
      tmp = NULL;
    }
  }

  // rNet
  CNRT_CHECK(cnrtDestroyFunction(rnet_function));
  CNRT_CHECK(cnrtUnloadModel(rnet_cnrt_model));

  if (rnet_param) free(rnet_param);
  rnet_param = NULL;

  if (rnet_inputCpuPtrS) free(rnet_inputCpuPtrS);
  rnet_inputCpuPtrS = NULL;

  if (rnet_inputCpuTempPtrS) free(rnet_inputCpuTempPtrS);
  rnet_inputCpuTempPtrS = NULL;

  if (rnet_outputCpuPtrS) free(rnet_outputCpuPtrS);
  rnet_outputCpuPtrS = NULL;

  if (rnet_outputCpuTempPtrS) free(rnet_outputCpuTempPtrS);
  rnet_outputCpuTempPtrS = NULL;

  if (rnet_inputMluPtrS) {
    for (int i = 0; i < rnet_inputNum; i++) {
     CNRT_CHECK(cnrtFree(rnet_inputMluPtrS[i]));
    }
    rnet_inputMluPtrS = NULL;
  }


  if (rnet_outputMluPtrS) {
    for (int i = 0; i < rnet_outputNum; i++) {
      CNRT_CHECK(cnrtFree(rnet_outputMluPtrS[i]));
    }
    rnet_outputMluPtrS = NULL;
  }

  for (auto tmp : rnet_input_cpu) {
    free(tmp);
    tmp = NULL;
  }
  for (auto tmp : rnet_input_temp_cpu) {
    free(tmp);
    tmp = NULL;
  }
  for (auto tmp : rnet_output_cpu) {
    free(tmp);
    tmp = NULL;
  }
  for (auto tmp : rnet_output_temp_cpu) {
    free(tmp);
    tmp = NULL;
  }

  // oNet
  CNRT_CHECK(cnrtDestroyFunction(onet_function));
  CNRT_CHECK(cnrtUnloadModel(onet_cnrt_model));

  if (onet_param) free(onet_param);
  onet_param = NULL;

  if (onet_inputCpuPtrS) free(onet_inputCpuPtrS);
  onet_inputCpuPtrS = NULL;

  if (onet_inputCpuPtrS) free(onet_inputCpuTempPtrS);
  onet_inputCpuPtrS = NULL;

  if (onet_outputCpuPtrS) free(onet_outputCpuPtrS);
  onet_outputCpuTempPtrS = NULL;

  if (onet_outputCpuPtrS) free(onet_outputCpuTempPtrS);
  onet_outputCpuTempPtrS = NULL;

  if (onet_inputMluPtrS) {
    for (int i = 0; i < onet_inputNum; i++) {
      CNRT_CHECK(cnrtFree(onet_inputMluPtrS[i]));
    }
    onet_inputMluPtrS = NULL;
  }

  if (onet_outputMluPtrS) {
    for (int i = 0; i < onet_outputNum; i++) {
      CNRT_CHECK(cnrtFree(onet_outputMluPtrS[i]));
    }
    onet_outputMluPtrS = NULL;
  }

  for (auto tmp : onet_input_cpu) {
    free(tmp);
    tmp = NULL;
  }
  for (auto tmp : onet_input_temp_cpu) {
    free(tmp);
    tmp = NULL;
  }
  for (auto tmp : onet_output_cpu) {
    free(tmp);
    tmp = NULL;
  }
  for (auto tmp : onet_output_temp_cpu) {
    free(tmp);
    tmp = NULL;
  }
}

void FaceDetector::prepareRuntimeContext(cnrtRuntimeContext_t *ctx,
       cnrtFunction_t function, int deviceId,
       const bool& channel_dup, uint64_t stack_size) {
  cnrtRuntimeContext_t rt_ctx;

  if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to create runtime context";
  }

  // set device ordinal. if not set, a random device will be used
  cnrtSetRuntimeContextDeviceId(rt_ctx, deviceId);

  // Instantiate the runtime context on actual MLU device
  // All cnrtSetRuntimeContext* interfaces must be caller prior to cnrtInitRuntimeContext
  if (cnrtInitRuntimeContext(rt_ctx, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to initialize runtime context";
  }

  *ctx = rt_ctx;
}


// load model and get function. init runtime context for each device
void FaceDetector::loadOfflinemodel(const string& offlinemodel,
       const vector<int>& deviceIds, const bool& channel_dup,
       cnrtModel_t* model, cnrtRuntimeContext_t* ctx) {
  cnrtFunction_t function;
  uint64_t stack_size;
  cnrtLoadModel(model, offlinemodel.c_str());
  cnrtQueryModelStackSize(*model, &stack_size);

  const string name = "subnet0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, *model, name.c_str());

  for (auto device : deviceIds) {
    prepareRuntimeContext(ctx, function, device, channel_dup, stack_size);
  }

  cnrtDestroyFunction(function);
}


void FaceDetector::initialPnet(vector<string> pnet_model_path) {
  vector<int> deviceIds_;

  deviceIds_.push_back(0);
  pnet_runtime_context.clear();
  for (int i = 0; i < pnet_model_path.size(); i++) {
    cnrtModel_t cnrt_model;
    cnrtFunction_t function;
    vector<int> in_shape;
    vector<int> cls_shape;
    vector<int> box_shape;
    vector<int> in_size;
    vector<int> out_size;
    vector<float *> input_cpu;
    vector<float *> output_cpu;
    vector<void *> input_temp_cpu;
    vector<void *> output_temp_cpu;
    void **param;
    void **inputCpuPtrS;
    void **outputCpuPtrS;
    void **inputCpuTempPtrS;
    void **outputCpuTempPtrS;
    void **inputMluPtrS;
    void **outputMluPtrS;
    int64_t *inputSizeS;
    int64_t *outputSizeS;
    vector<int*> inDimValues, outDimValues;
    int inDimNums, outDimNums;
    int inputNum, outputNum;
    cnrtRuntimeContext_t ctx;
    loadOfflinemodel(pnet_model_path[i], deviceIds_, true, &cnrt_model, &ctx);
    pnet_runtime_context.push_back(ctx);
    cnrtGetRuntimeContextInfo(ctx, CNRT_RT_CTX_FUNCTION,
        reinterpret_cast<void **>(&function));
    CNRT_CHECK(cnrtGetInputDataSize(&inputSizeS, &inputNum, function));
    CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeS, &outputNum, function));

    inDimValues.resize(inputNum, nullptr);
    outDimValues.resize(outputNum, nullptr);
    for (int idx = 0; idx < inputNum; ++idx) {
      CNRT_CHECK(cnrtGetInputDataShape(&(inDimValues[idx]),
            &inDimNums, idx, function));
      if (0 == idx) {
        in_shape.push_back(inDimValues[idx][0]);
        in_shape.push_back(inDimValues[idx][3]);
        in_shape.push_back(inDimValues[idx][1]);
        in_shape.push_back(inDimValues[idx][2]);
      }
      in_size.push_back(inDimValues[idx][0] * inDimValues[idx][1] *
                        inDimValues[idx][2] * inDimValues[idx][3]);
    }

    CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[0]), &outDimNums, 0, function));
    box_shape.push_back(outDimValues[0][0]);
    box_shape.push_back(outDimValues[0][3]);
    box_shape.push_back(outDimValues[0][1]);
    box_shape.push_back(outDimValues[0][2]);
    out_size.push_back(outDimValues[0][0] * outDimValues[0][1] *
                       outDimValues[0][2] * outDimValues[0][3]);

    CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[1]), &outDimNums, 1, function));
    cls_shape.push_back(outDimValues[1][0]);
    cls_shape.push_back(outDimValues[1][3]);
    cls_shape.push_back(outDimValues[1][1]);
    cls_shape.push_back(outDimValues[1][2]);
    out_size.push_back(outDimValues[1][0] * outDimValues[1][1] *
                       outDimValues[1][2] * outDimValues[1][3]);

    inputCpuPtrS = reinterpret_cast<void **>(malloc(sizeof(void *) * inputNum));
    inputCpuTempPtrS = reinterpret_cast<void **>(malloc(sizeof(void *) * inputNum));
    inputMluPtrS = reinterpret_cast<void**>(
        malloc(sizeof(void*) * inputNum));
    for (int j = 0; j < inputNum; j++) {
      float *input_data;
      input_data = reinterpret_cast<float *>(malloc(sizeof(float) * in_size[j]));
      input_cpu.push_back(input_data);
      inputCpuPtrS[j] = reinterpret_cast<void *>(input_data);
      void* input_temp_data = reinterpret_cast<void*>(malloc(sizeof(float) * in_size[j]));
      input_temp_cpu.push_back(input_temp_data);
      inputCpuTempPtrS[j] = reinterpret_cast<void *>(input_temp_data);
      cnrtMalloc(&(inputMluPtrS[j]), inputSizeS[j]);
    }

    outputCpuPtrS =
        reinterpret_cast<void **>(malloc(sizeof(void *) * outputNum));
    outputCpuTempPtrS =
        reinterpret_cast<void **>(malloc(sizeof(void *) * outputNum));
    outputMluPtrS = reinterpret_cast<void**>(
        malloc(sizeof(void*) * outputNum));
    for (int j = 0; j < outputNum; j++) {
      float *output_data;
      output_data = reinterpret_cast<float *>(malloc(
                                sizeof(float) * out_size[j]));
      output_cpu.push_back(output_data);
      outputCpuPtrS[j] = reinterpret_cast<void*>(output_data);
      void* output_temp_data = reinterpret_cast<void*>(malloc(
                                 sizeof(float) * out_size[j]));
      output_temp_cpu.push_back(output_temp_data);
      outputCpuTempPtrS[j] = reinterpret_cast<void*>(output_temp_data);
      cnrtMalloc(&(outputMluPtrS[j]), outputSizeS[j]);
    }

    param = reinterpret_cast<void **>(
        malloc(sizeof(void *) * (inputNum + outputNum)));

    for (int j = 0; j < inputNum; j++) {
      param[j] = inputMluPtrS[j];
    }

    for (int j = 0; j < outputNum; j++) {
      param[inputNum + j] = outputMluPtrS[j];
    }

    pnet_cnrt_model.push_back(cnrt_model);
    pnet_function.push_back(function);
    pnet_in_shape.push_back(in_shape);
    pnet_cls_shape.push_back(cls_shape);
    pnet_box_shape.push_back(box_shape);
    pnet_in_size.push_back(in_size);
    pnet_out_size.push_back(out_size);
    pnet_input_cpu.push_back(input_cpu);
    pnet_output_cpu.push_back(output_cpu);
    pnet_input_temp_cpu.push_back(input_temp_cpu);
    pnet_output_temp_cpu.push_back(output_temp_cpu);
    pnet_param.push_back(param);
    pnet_inputCpuPtrS.push_back(inputCpuPtrS);
    pnet_outputCpuPtrS.push_back(outputCpuPtrS);
    pnet_inputCpuTempPtrS.push_back(inputCpuTempPtrS);
    pnet_outputCpuTempPtrS.push_back(outputCpuTempPtrS);
    pnet_inputMluPtrS.push_back(inputMluPtrS);
    pnet_outputMluPtrS.push_back(outputMluPtrS);
    pnet_inputSizeS.push_back(inputSizeS);
    pnet_outputSizeS.push_back(outputSizeS);
    pnet_inputNum.push_back(inputNum);
    pnet_outputNum.push_back(outputNum);
  }
}

void FaceDetector::initialRnet(string rnet_model_path) {
  vector<cv::Mat> n_channels;
  vector<int> deviceIds_;
  vector<int*> inDimValues, outDimValues;
  int inDimNums, outDimNums;

  deviceIds_.push_back(0);
  loadOfflinemodel(rnet_model_path, deviceIds_, true,
      &rnet_cnrt_model, &rnet_runtime_context);
  cnrtGetRuntimeContextInfo(rnet_runtime_context, CNRT_RT_CTX_FUNCTION,
          reinterpret_cast<void **>(&rnet_function));
  cnrtGetInputDataSize(&rnet_inputSizeS, &rnet_inputNum, rnet_function);
  cnrtGetOutputDataSize(&rnet_outputSizeS, &rnet_outputNum, rnet_function);
  inDimValues.resize(rnet_inputNum, nullptr);
  outDimValues.resize(rnet_outputNum, nullptr);
  for (int idx = 0; idx < rnet_inputNum; ++idx) {
    CNRT_CHECK(cnrtGetInputDataShape(&(inDimValues[idx]),
          &inDimNums, idx, rnet_function));
    if (0 == idx) {
      rnet_in_shape.push_back(inDimValues[idx][0]);
      rnet_in_shape.push_back(inDimValues[idx][3]);
      rnet_in_shape.push_back(inDimValues[idx][1]);
      rnet_in_shape.push_back(inDimValues[idx][2]);
    }
    rnet_in_size.push_back(inDimValues[idx][0] * inDimValues[idx][1] *
                      inDimValues[idx][2] * inDimValues[idx][3]);
  }

  CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[0]),
        &outDimNums, 0, rnet_function));
  rnet_box_shape.push_back(outDimValues[0][0]);
  rnet_box_shape.push_back(outDimValues[0][3]);
  rnet_box_shape.push_back(outDimValues[0][1]);
  rnet_box_shape.push_back(outDimValues[0][2]);
  rnet_out_size.push_back(outDimValues[0][0] * outDimValues[0][1] *
                     outDimValues[0][2] * outDimValues[0][3]);

  CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[1]),
        &outDimNums, 1, rnet_function));
  rnet_cls_shape.push_back(outDimValues[1][0]);
  rnet_cls_shape.push_back(outDimValues[1][3]);
  rnet_cls_shape.push_back(outDimValues[1][1]);
  rnet_cls_shape.push_back(outDimValues[1][2]);
  rnet_out_size.push_back(outDimValues[1][0] * outDimValues[1][1] *
                    outDimValues[1][2] * outDimValues[1][3]);

  rnet_inputCpuPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * rnet_inputNum));
  rnet_inputCpuTempPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * rnet_inputNum));
  rnet_inputMluPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * rnet_inputNum));
  for (int i = 0; i < rnet_inputNum; i++) {
    float *input_data;
    input_data = reinterpret_cast<float *>(malloc(sizeof(float) * rnet_in_size[i]));
    rnet_input_cpu.push_back(input_data);
    void* input_temp_data = reinterpret_cast<void*>(malloc(
                               sizeof(float) * rnet_in_size[i]));
    rnet_input_temp_cpu.push_back(input_temp_data);
    rnet_inputCpuPtrS[i] = reinterpret_cast<void *>(input_data);
    rnet_inputCpuTempPtrS[i] = reinterpret_cast<void *>(input_temp_data);
    cnrtMalloc(&(rnet_inputMluPtrS[i]), rnet_inputSizeS[i]);
  }
  rnet_outputCpuPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * rnet_outputNum));
  rnet_outputCpuTempPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * rnet_outputNum));
  rnet_outputMluPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * rnet_outputNum));
  for (int i = 0; i < rnet_outputNum; i++) {
    float *output_data;
    output_data = reinterpret_cast<float *>(malloc(
                             sizeof(float) * rnet_out_size[i]));
    rnet_output_cpu.push_back(output_data);
    void* output_temp_data = reinterpret_cast<void*>(malloc(
                             sizeof(float) * rnet_out_size[i]));
    rnet_output_temp_cpu.push_back(output_temp_data);
    rnet_outputCpuPtrS[i] = reinterpret_cast<void *>(output_data);
    rnet_outputCpuTempPtrS[i] = reinterpret_cast<void *>(output_temp_data);
    cnrtMalloc(&(rnet_outputMluPtrS[i]), rnet_outputSizeS[i]);
  }

  rnet_param = reinterpret_cast<void **>(
      malloc(sizeof(void *) * (rnet_inputNum + rnet_outputNum)));
  for (int i = 0; i < rnet_inputNum; i++) {
    rnet_param[i] = rnet_inputMluPtrS[i];
  }

  for (int i = 0; i < rnet_outputNum; i++) {
    rnet_param[rnet_inputNum + i] = rnet_outputMluPtrS[i];
  }
}

void FaceDetector::initialOnet(string onet_model_path) {
  vector<cv::Mat> n_channels;
  vector<int> deviceIds_;
  vector<int*> inDimValues, outDimValues;
  int inDimNums, outDimNums;

  deviceIds_.push_back(0);
  loadOfflinemodel(onet_model_path, deviceIds_, true,
      &onet_cnrt_model, &onet_runtime_context);
  cnrtGetRuntimeContextInfo(onet_runtime_context, CNRT_RT_CTX_FUNCTION,
          reinterpret_cast<void **>(&onet_function));
  cnrtGetInputDataSize(&onet_inputSizeS, &onet_inputNum, onet_function);
  cnrtGetOutputDataSize(&onet_outputSizeS, &onet_outputNum, onet_function);

  inDimValues.resize(onet_inputNum, nullptr);
  outDimValues.resize(onet_outputNum, nullptr);
  for (int idx = 0; idx < onet_inputNum; ++idx) {
    CNRT_CHECK(cnrtGetInputDataShape(&(inDimValues[idx]),
          &inDimNums, idx, onet_function));
    if (0 == idx) {
      onet_in_shape.push_back(inDimValues[idx][0]);
      onet_in_shape.push_back(inDimValues[idx][3]);
      onet_in_shape.push_back(inDimValues[idx][1]);
      onet_in_shape.push_back(inDimValues[idx][2]);
    }
    onet_in_size.push_back(inDimValues[idx][0] * inDimValues[idx][1] *
                      inDimValues[idx][2] * inDimValues[idx][3]);
  }

  CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[0]),
        &outDimNums, 0, onet_function));
  onet_box_shape.push_back(outDimValues[0][0]);
  onet_box_shape.push_back(outDimValues[0][3]);
  onet_box_shape.push_back(outDimValues[0][1]);
  onet_box_shape.push_back(outDimValues[0][2]);
  onet_out_size.push_back(outDimValues[0][0] * outDimValues[0][1] *
                     outDimValues[0][2] * outDimValues[0][3]);

  CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[1]),
        &outDimNums, 1, onet_function));
  onet_points_shape.push_back(outDimValues[1][0]);
  onet_points_shape.push_back(outDimValues[1][3]);
  onet_points_shape.push_back(outDimValues[1][1]);
  onet_points_shape.push_back(outDimValues[1][2]);
  onet_out_size.push_back(outDimValues[1][0] * outDimValues[1][1] *
                     outDimValues[1][2] * outDimValues[1][3]);

  CNRT_CHECK(cnrtGetOutputDataShape(&(outDimValues[2]),
        &outDimNums, 2, onet_function));
  onet_cls_shape.push_back(outDimValues[2][0]);
  onet_cls_shape.push_back(outDimValues[2][3]);
  onet_cls_shape.push_back(outDimValues[2][1]);
  onet_cls_shape.push_back(outDimValues[2][2]);
  onet_out_size.push_back(outDimValues[2][0] * outDimValues[2][1] *
                    outDimValues[2][2] * outDimValues[2][3]);

  onet_inputCpuPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * onet_inputNum));
  onet_inputCpuTempPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * onet_inputNum));
  onet_inputMluPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * onet_inputNum));
  for (int i = 0; i < onet_inputNum; i++) {
    float *input_data;
    input_data = reinterpret_cast<float *>(malloc(
                               sizeof(float) * onet_in_size[i]));
    onet_input_cpu.push_back(input_data);
    void* input_temp_data = reinterpret_cast<void*>(malloc(
                               sizeof(float) * onet_in_size[i]));
    onet_input_temp_cpu.push_back(input_temp_data);
    onet_inputCpuPtrS[i] = reinterpret_cast<void *>(input_data);
    onet_inputCpuTempPtrS[i] = reinterpret_cast<void*>(input_temp_data);
    cnrtMalloc(&(onet_inputMluPtrS[i]), onet_inputSizeS[i]);
  }
  onet_outputCpuPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * onet_outputNum));
  onet_outputCpuTempPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * onet_outputNum));
  onet_outputMluPtrS =
      reinterpret_cast<void **>(malloc(sizeof(void *) * onet_outputNum));
  for (int i = 0; i < onet_outputNum; i++) {
    float *output_data;
    output_data = reinterpret_cast<float *>(malloc(
                             sizeof(float) * onet_out_size[i]));
    onet_output_cpu.push_back(output_data);
    void* output_temp_data = reinterpret_cast<void*>(malloc(
                             sizeof(float) * onet_out_size[i]));
    onet_output_temp_cpu.push_back(output_temp_data);
    onet_outputCpuPtrS[i] = reinterpret_cast<void *>(output_data);
    onet_outputCpuTempPtrS[i] = reinterpret_cast<void *>(output_temp_data);
    cnrtMalloc(&(onet_outputMluPtrS[i]), onet_outputSizeS[i]);
  }

  onet_param = reinterpret_cast<void **>(
      malloc(sizeof(void *) * (onet_inputNum + onet_outputNum)));

  for (int i = 0; i < onet_inputNum; i++) {
    onet_param[i] = onet_inputMluPtrS[i];
  }

  for (int i = 0; i < onet_outputNum; i++) {
    onet_param[onet_inputNum + i] = onet_outputMluPtrS[i];
  }
}

void FaceDetector::wrapInputLayer(const vector<int> &input_shape,
                                  float *input_dataBuffer,
                                  vector<cv::Mat> *input_channels) {
    int width = input_shape[3];
    int height = input_shape[2];
    float *f_input_data = input_dataBuffer;
    unsigned char *uc_input_data = (unsigned char *)input_dataBuffer;
    int margin = width * height;
    int type = CV_32FC1;

    if (FLAGS_first_conv)
      type = CV_8UC1;
    for (int j = 0; j < input_shape[0]; j++) {
      for (int i = 0; i < input_shape[1]; i++) {
        if (FLAGS_first_conv) {
          cv::Mat channel(height, width, type, uc_input_data);
          input_channels->push_back(channel);
          uc_input_data += margin;
        } else {
          cv::Mat channel(height, width, type, f_input_data);
          input_channels->push_back(channel);
          f_input_data += margin;
        }
      }
    }
}

void FaceDetector::pyrDown(const vector<cv::Mat> &img,
                           vector<cv::Mat> *input_channels) {
  assert(img.size() == input_channels->size());
  int hs = (*input_channels)[0].rows;
  int ws = (*input_channels)[0].cols;
  cv::Mat img_resized;
  for (int i = 0; i < img.size(); i++) {
    cv::resize(img[i], (*input_channels)[i], cv::Size(ws, hs),
               cv::INTER_NEAREST);
  }
}

void FaceDetector::buildInputChannels(const vector<cv::Mat> &img_channels,
                                      const vector<BoundingBox> &boxes,
                                      const cv::Size &target_size,
                                      vector<cv::Mat> *input_channels) {
  assert(img_channels.size() * boxes.size() == input_channels->size());

  cv::Rect img_rect(0, 0, img_channels[0].cols, img_channels[0].rows);

  for (int n = 0; n < boxes.size(); n++) {
    cv::Rect rect;
    rect.x = boxes[n].x1;
    rect.y = boxes[n].y1;
    rect.width = boxes[n].x2 - boxes[n].x1 + 1;
    rect.height = boxes[n].y2 - boxes[n].y1 + 1;
    cv::Rect cuted_rect = rect & img_rect;
    cv::Rect inner_rect(cuted_rect.x - rect.x, cuted_rect.y - rect.y,
                        cuted_rect.width, cuted_rect.height);
    for (int c = 0; c < img_channels.size(); c++) {
      int type;
      if (FLAGS_first_conv)
        type  = CV_8UC1;
      else
        type = CV_32FC1;
      cv::Mat tmp(rect.height, rect.width, type, cv::Scalar(0));
      img_channels[c](cuted_rect).copyTo(tmp(inner_rect));
      cv::resize(tmp, (*input_channels)[n * img_channels.size() + c],
                 target_size);
    }
  }
}

void FaceDetector::generateBoundingBox(const vector<float> &boxRegs,
                                       const vector<int> &box_shape,
                                       const vector<float> &cls,
                                       const vector<int> &cls_shape,
                                       float scale_w, float scale_h,
                                       const float threshold,
                                       vector<BoundingBox> *filterOutBoxes) {
  // clear output element
  filterOutBoxes->clear();
  int stride = 2;
  int cellsize = 12;
  assert(box_shape.size() == cls_shape.size());
  assert(box_shape[3] == cls_shape[3] && box_shape[2] == cls_shape[2]);
  assert(box_shape[1] == 4 && cls_shape[1] == 2);
  int w = box_shape[3];
  int h = box_shape[2];
  // int n = box_shape[0];
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float score = cls[0 * 2 * w * h + 1 * w * h + w * y + x];
      if (score >= threshold) {
        BoundingBox box;
        box.dx1 = boxRegs[0 * w * h + w * y + x];
        box.dy1 = boxRegs[1 * w * h + w * y + x];
        box.dx2 = boxRegs[2 * w * h + w * y + x];
        box.dy2 = boxRegs[3 * w * h + w * y + x];

        box.x1 = floor((stride * x + 1) / scale_w);
        box.y1 = floor((stride * y + 1) / scale_w);
        box.x2 = floor((stride * x + cellsize) / scale_w);
        box.y2 = floor((stride * y + cellsize) / scale_w);
        box.score = score;

        // add elements
        filterOutBoxes->push_back(box);
      }
    }
  }
}

void FaceDetector::filteroutBoundingBox(
    const vector<FaceDetector::BoundingBox> &boxes,
    const vector<float> &boxRegs, const vector<int> &box_shape,
    const vector<float> &cls, const vector<int> &cls_shape,
    const vector<float> &points, const vector<int> &points_shape,
    float threshold, vector<FaceDetector::BoundingBox> *filterOutBoxes) {
  filterOutBoxes->clear();
  assert(box_shape.size() == cls_shape.size());
  // assert(box_shape[0] == boxes.size() && cls_shape[0] == boxes.size());
  assert(box_shape[1] == 4 && cls_shape[1] == 2);

  for (int i = 0; i < boxes.size(); i++) {
    float score = cls[i * 2 + 1];
    if (score > threshold) {
      BoundingBox box = boxes[i];
      float w = boxes[i].y2 - boxes[i].y1 + 1;
      float h = boxes[i].x2 - boxes[i].x1 + 1;
      if (points.size() > 0) {
        for (int p = 0; p < 5; p++) {
          box.points_x[p] = points[i * 10 + 5 + p] * w + boxes[i].x1 - 1;
          box.points_y[p] = points[i * 10 + p] * h + boxes[i].y1 - 1;
        }
      }
      box.dx1 = boxRegs[i * 4 + 0];
      box.dy1 = boxRegs[i * 4 + 1];
      box.dx2 = boxRegs[i * 4 + 2];
      box.dy2 = boxRegs[i * 4 + 3];

      box.x1 = boxes[i].x1 + box.dy1 * w;
      box.y1 = boxes[i].y1 + box.dx1 * h;
      box.x2 = boxes[i].x2 + box.dy2 * w;
      box.y2 = boxes[i].y2 + box.dx2 * h;

      // rerec
      w = box.x2 - box.x1;
      h = box.y2 - box.y1;
      float l = std::max(w, h);
      box.x1 += (w - l) * 0.5;
      box.y1 += (h - l) * 0.5;
      box.x2 = box.x1 + l;
      box.y2 = box.y1 + l;
      box.score = score;
      filterOutBoxes->push_back(box);
    }
  }
}

void FaceDetector::nms(vector<BoundingBox> *boxes, float threshold,
                       NMS_TYPE type, vector<BoundingBox> *filterOutBoxes) {
  filterOutBoxes->clear();
  if ((*boxes).size() == 0) return;

  // descending sort
  sort((*boxes).begin(), (*boxes).end(), CmpBoundingBox());
  vector<size_t> idx((*boxes).size());
  for (int i = 0; i < idx.size(); i++) {
    idx[i] = i;
  }
  while (idx.size() > 0) {
    int good_idx = idx[0];
    filterOutBoxes->push_back((*boxes)[good_idx]);
    // hypothesis : the closer the scores are similar
    vector<size_t> tmp = idx;
    idx.clear();
    for (int i = 1; i < tmp.size(); i++) {
      int tmp_i = tmp[i];
      float inter_x1 = std::max((*boxes)[good_idx].x1, (*boxes)[tmp_i].x1);
      float inter_y1 = std::max((*boxes)[good_idx].y1, (*boxes)[tmp_i].y1);
      float inter_x2 = std::min((*boxes)[good_idx].x2, (*boxes)[tmp_i].x2);
      float inter_y2 = std::min((*boxes)[good_idx].y2, (*boxes)[tmp_i].y2);

      float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
      float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

      float inter_area = w * h;
      float area_1 = ((*boxes)[good_idx].x2 - (*boxes)[good_idx].x1 + 1) *
                     ((*boxes)[good_idx].y2 - (*boxes)[good_idx].y1 + 1);
      float area_2 =
          ((*boxes)[i].x2 - (*boxes)[i].x1 + 1) * ((*boxes)[i].y2 - (*boxes)[i].y1 + 1);
      float o = (type == UNION ? (inter_area / (area_1 + area_2 - inter_area))
                               : (inter_area / std::min(area_1, area_2)));
      if (o <= threshold) idx.push_back(tmp_i);
    }
  }
}

void FaceDetector::nmsGlobal() {
  if (totalBoxes.size() > 0) {
    vector<BoundingBox> globalFilterBoxes;
    nms(&totalBoxes, 0.7, UNION, &globalFilterBoxes);
    totalBoxes.clear();
    for (int i = 0; i < globalFilterBoxes.size(); i++) {
      float regw = globalFilterBoxes[i].y2 - globalFilterBoxes[i].y1;
      float regh = globalFilterBoxes[i].x2 - globalFilterBoxes[i].x1;
      BoundingBox box;
      float x1 = globalFilterBoxes[i].x1 + globalFilterBoxes[i].dy1 * regw;
      float y1 = globalFilterBoxes[i].y1 + globalFilterBoxes[i].dx1 * regh;
      float x2 = globalFilterBoxes[i].x2 + globalFilterBoxes[i].dy2 * regw;
      float y2 = globalFilterBoxes[i].y2 + globalFilterBoxes[i].dx2 * regh;
      float h = y2 - y1;
      float w = x2 - x1;
      float l = std::max(h, w);
      x1 += (w - l) * 0.5;
      y1 += (h - l) * 0.5;
      x2 = x1 + l;
      y2 = y1 + l;
      box.x1 = x1, box.x2 = x2, box.y1 = y1, box.y2 = y2;
      if (box.x1 > 0 && box.x1 < box.x2 - min_size && box.y1 > 0 &&
          box.y1 < box.y2 - min_size && box.x2 < img_W && box.y2 < img_H) {
        totalBoxes.push_back(box);
      }
    }
  }
}

void FaceDetector::doPnet(int streamid) {
  vector<cv::Mat> pyr_channels;
  vector<BoundingBox> filterOutBoxes;
  vector<BoundingBox> nmsOutBoxes;
  cnrtQueue_t pnet_queue;

  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, m_nDevNum));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  CNRT_CHECK(cnrtCreateQueue(&pnet_queue));

  cnrtNotifier_t pnet_notifierBeginning, pnet_notifierEnd;
  cnrtCreateNotifier(&pnet_notifierBeginning);
  cnrtCreateNotifier(&pnet_notifierEnd);

  vector<int> &midx = m_idxs[streamid];

  while (m_running) {
    for (auto idx : midx) {
      if (sem_trywait(&sem_in[idx])) {
        usleep(100);
        continue;
      }

      float eventInterval = 0;
      int tmp_dim_order[4] = {0, 2, 3, 1};
      int *dim_order = tmp_dim_order;
      pyr_channels.clear();
      wrapInputLayer(pnet_in_shape[idx], pnet_input_cpu[idx][0], &pyr_channels);
      pyrDown(sample_norm_channels, &pyr_channels);

      cnrtDataType_t* pnet_input_data_type = nullptr;
      int tmp_inputNum = 0;
      cnrtGetInputDataType(&pnet_input_data_type,
          &tmp_inputNum, pnet_function[idx]);
      TimePoint t1 = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < pnet_inputNum[idx]; i++) {
        void* temp_ptr = nullptr;
        int tmp_shape[] = {pnet_in_shape[idx][0],
                           pnet_in_shape[idx][1],
                           pnet_in_shape[idx][2],
                           pnet_in_shape[idx][3]};
        cnrtDataType_t cpu_data_type = CNRT_FLOAT32;
        if (FLAGS_first_conv)
          cpu_data_type = CNRT_UINT8;
        if (pnet_input_data_type[i] != cpu_data_type) {
          cnrtTransOrderAndCast(pnet_inputCpuPtrS[idx][i],
                           cpu_data_type,
                           pnet_inputCpuTempPtrS[idx][i],
                           pnet_input_data_type[i],
                           NULL,
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = pnet_inputCpuTempPtrS[idx][i];
        } else {
          cnrtTransDataOrder(pnet_inputCpuPtrS[idx][i],
                           cpu_data_type,
                           pnet_inputCpuTempPtrS[idx][i],
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = pnet_inputCpuTempPtrS[idx][i];
        }
        cnrtMemcpy(pnet_inputMluPtrS[idx][i],
                  temp_ptr,
                  pnet_inputSizeS[idx][i],
                  CNRT_MEM_TRANS_DIR_HOST2DEV);
      }

      cnrtPlaceNotifier(pnet_notifierBeginning, pnet_queue);
      CNRT_CHECK(cnrtInvokeRuntimeContext(pnet_runtime_context[idx],
            pnet_param[idx], pnet_queue, nullptr));
      cnrtPlaceNotifier(pnet_notifierEnd, pnet_queue);
      CNRT_CHECK(cnrtSyncQueue(pnet_queue));
      cnrtNotifierDuration(pnet_notifierBeginning, pnet_notifierEnd, &eventInterval);
      AddRuntime(eventInterval);
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      int latency_ms = std::chrono::duration_cast<TimeDuration_ms>(t2 - t1).count();
      AddLatency(latency_ms);

      cnrtDataType_t* pnet_output_data_type = nullptr;
      int tmp_outputNum = 0;
      cnrtGetOutputDataType(&pnet_output_data_type,
          &tmp_outputNum, pnet_function[idx]);
      for (int i = 0; i < pnet_outputNum[idx]; i++) {
        cnrtMemcpy(pnet_outputCpuTempPtrS[idx][i],
                   pnet_outputMluPtrS[idx][i],
                   pnet_outputSizeS[idx][i],
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        int tmp_out_shape[4];
        int tmp_out_order[4] = {0, 3, 1, 2};
        if (i == 0) {
          tmp_out_shape[0] = pnet_box_shape[idx][0];
          tmp_out_shape[1] = pnet_box_shape[idx][2];
          tmp_out_shape[2] = pnet_box_shape[idx][3];
          tmp_out_shape[3] = pnet_box_shape[idx][1];
        } else if (i == 1) {
          tmp_out_shape[0] = pnet_cls_shape[idx][0];
          tmp_out_shape[1] = pnet_cls_shape[idx][2];
          tmp_out_shape[2] = pnet_cls_shape[idx][3];
          tmp_out_shape[3] = pnet_cls_shape[idx][1];
        } else {
          LOG(ERROR) << "Pnet Out Num error!";
          exit(-1);
        }

        if (pnet_output_data_type[i] != CNRT_FLOAT32) {
          cnrtTransOrderAndCast(pnet_outputCpuTempPtrS[idx][i],
                           pnet_output_data_type[i],
                           pnet_outputCpuPtrS[idx][i],
                           CNRT_FLOAT32,
                           NULL,
                           4,
                           tmp_out_shape,
                           tmp_out_order);
        } else {
          cnrtTransDataOrder(pnet_outputCpuTempPtrS[idx][i],
                           pnet_output_data_type[i],
                           pnet_outputCpuPtrS[idx][i],
                           4,
                           tmp_out_shape,
                           tmp_out_order);
        }
      }

      /* copy the output layer to a vector*/
      const float *begin1 = pnet_output_cpu[idx][1];
      const float *end1 = pnet_out_size[idx][1] + begin1;
      vector<float> pnet_cls(begin1, end1);

      const float *begin0 = pnet_output_cpu[idx][0];
      const float *end0 = pnet_out_size[idx][0] + begin0;
      vector<float> pnet_regs(begin0, end0);

      filterOutBoxes.clear();
      nmsOutBoxes.clear();
      float cur_sc_w = 1.0 * pnet_in_shape[idx][3] / img_W;
      float cur_sc_h = 1.0 * pnet_in_shape[idx][2] / img_H;
      generateBoundingBox(pnet_regs, pnet_box_shape[idx], pnet_cls,
                          pnet_cls_shape[idx], cur_sc_w, cur_sc_h,
                          P_thres, &filterOutBoxes);

      nms(&filterOutBoxes, 0.5, UNION, &nmsOutBoxes);

      mtx.lock();
      if (nmsOutBoxes.size() > 0) {
        totalBoxes.insert(totalBoxes.end(), nmsOutBoxes.begin(),
                          nmsOutBoxes.end());
      }
      mtx.unlock();

      sem_post(&sem_out[idx]);
    }
  }
  CNRT_CHECK(cnrtDestroyQueue(pnet_queue));
  cnrtDestroyNotifier(&pnet_notifierBeginning);
  cnrtDestroyNotifier(&pnet_notifierEnd);
}

void FaceDetector::doRnet() {
  vector<cv::Mat> n_channels;
  vector<BoundingBox> filterOutBoxes;
  vector<BoundingBox> nmsOutBoxes;


  cnrtNotifier_t rnet_notifierBeginning, rnet_notifierEnd;
  cnrtCreateNotifier(&rnet_notifierBeginning);
  cnrtCreateNotifier(&rnet_notifierEnd);
  float eventInterval = 0;
  int tmp_dim_order[4] = {0, 2, 3, 1};
  int *dim_order = tmp_dim_order;
  if (totalBoxes.size() > 0) {
    vector<float> rnet_cls_all, rnet_regs_all;
    const int batchsize = rnet_in_shape[0];
    const int loop_num = totalBoxes.size() / batchsize;
    const int rem = totalBoxes.size() % batchsize;
    vector<BoundingBox> batchBoxs;
    // n_channels.clear();
    wrapInputLayer(rnet_in_shape, rnet_input_cpu[0], &n_channels);

    for (int i = 0; i < loop_num; i++) {
      batchBoxs.clear();
      for (int j = 0; j < batchsize; j++) {
        batchBoxs.push_back(totalBoxes.at(j + i * batchsize));
      }
      TimePoint t1 = std::chrono::high_resolution_clock::now();
      buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(24, 24),
                         &n_channels);
      cnrtDataType_t* rnet_input_data_type = nullptr;
      cnrtGetInputDataType(&rnet_input_data_type, &rnet_inputNum, rnet_function);
      for (int i = 0; i < rnet_inputNum; i++) {
        void* temp_ptr = nullptr;
        int tmp_shape[] = {rnet_in_shape[0],
                           rnet_in_shape[1],
                           rnet_in_shape[2],
                           rnet_in_shape[3]};
        cnrtDataType_t cpu_data_type = CNRT_FLOAT32;
        if (FLAGS_first_conv)
          cpu_data_type = CNRT_UINT8;
        if (rnet_input_data_type[i] != cpu_data_type) {
          cnrtTransOrderAndCast(rnet_inputCpuPtrS[i],
                           cpu_data_type,
                           rnet_inputCpuTempPtrS[i],
                           rnet_input_data_type[i],
                           NULL,
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = rnet_inputCpuTempPtrS[i];
        } else {
          cnrtTransDataOrder(rnet_inputCpuPtrS[i],
                           cpu_data_type,
                           rnet_inputCpuTempPtrS[i],
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = rnet_inputCpuTempPtrS[i];
        }
        cnrtMemcpy(rnet_inputMluPtrS[i],
                   temp_ptr,
                   rnet_inputSizeS[i],
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
      }

      cnrtPlaceNotifier(rnet_notifierBeginning, queue_);
      CNRT_CHECK(cnrtInvokeRuntimeContext(rnet_runtime_context,
            rnet_param, queue_, nullptr));
      cnrtPlaceNotifier(rnet_notifierEnd, queue_);
      CNRT_CHECK(cnrtSyncQueue(queue_));
      cnrtNotifierDuration(rnet_notifierBeginning, rnet_notifierEnd, &eventInterval);
      AddRuntime(eventInterval);
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      int latency_ms = std::chrono::duration_cast<TimeDuration_ms>(t2 - t1).count();
      AddLatency(latency_ms);
      cnrtDataType_t* rnet_output_data_type = nullptr;
      cnrtGetOutputDataType(&rnet_output_data_type, &rnet_outputNum, rnet_function);

      for (int i = 0; i < rnet_outputNum; i++) {
        cnrtMemcpy(rnet_outputCpuTempPtrS[i],
                   rnet_outputMluPtrS[i],
                   rnet_outputSizeS[i],
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        int tmp_out_shape[4];
        int tmp_out_order[4] = {0, 3, 1, 2};
        if (i == 0) {
          tmp_out_shape[0] = rnet_box_shape[0];
          tmp_out_shape[1] = rnet_box_shape[2];
          tmp_out_shape[2] = rnet_box_shape[3];
          tmp_out_shape[3] = rnet_box_shape[1];
        } else if (i == 1) {
          tmp_out_shape[0] = rnet_cls_shape[0];
          tmp_out_shape[1] = rnet_cls_shape[2];
          tmp_out_shape[2] = rnet_cls_shape[3];
          tmp_out_shape[3] = rnet_cls_shape[1];
        } else {
          LOG(ERROR) << "Rnet Out Num error!";
          exit(-1);
        }
        if (rnet_output_data_type[i] != CNRT_FLOAT32) {
          cnrtTransOrderAndCast(rnet_outputCpuTempPtrS[i],
                           rnet_output_data_type[i],
                           rnet_outputCpuPtrS[i],
                           CNRT_FLOAT32,
                           NULL,
                           4,
                           tmp_out_shape,
                           tmp_out_order);
        } else {
          cnrtTransDataOrder(rnet_outputCpuTempPtrS[i],
                             CNRT_FLOAT32,
                             rnet_outputCpuPtrS[i],
                             4,
                             tmp_out_shape,
                             tmp_out_order);
        }
      }

      /* copy the output layer to a vector*/
      const float *begin1 = rnet_output_cpu[1];
      const float *end1 = rnet_out_size[1] + begin1;
      vector<float> rnet_cls(begin1, end1);
      rnet_cls_all.insert(rnet_cls_all.end(), rnet_cls.begin(), rnet_cls.end());

      const float *begin0 = rnet_output_cpu[0];
      const float *end0 = rnet_out_size[0] + begin0;
      vector<float> rnet_regs(begin0, end0);
      rnet_regs_all.insert(rnet_regs_all.end(), rnet_regs.begin(),
                           rnet_regs.end());
    }
    if (rem > 0) {
      batchBoxs.clear();
      for (int j = 0; j < rem; j++) {
        batchBoxs.push_back(totalBoxes.at(j + loop_num * batchsize));
      }
      for (int j = rem; j < batchsize; j++) {
        batchBoxs.push_back(totalBoxes.at(totalBoxes.size() - 1));
      }
      TimePoint t1 = std::chrono::high_resolution_clock::now();

      buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(24, 24),
                         &n_channels);

      cnrtDataType_t* rnet_input_data_type = nullptr;
      cnrtGetInputDataType(&rnet_input_data_type, &rnet_inputNum, rnet_function);
      for (int i = 0; i < rnet_inputNum; i++) {
        void* temp_ptr = nullptr;
        int tmp_shape[] = {rnet_in_shape[0],
                           rnet_in_shape[1],
                           rnet_in_shape[2],
                           rnet_in_shape[3]};
        cnrtDataType_t cpu_data_type = CNRT_FLOAT32;
        if (FLAGS_first_conv)
          cpu_data_type = CNRT_UINT8;
        if (rnet_input_data_type[i] != cpu_data_type) {
          cnrtTransOrderAndCast(rnet_inputCpuPtrS[i],
                           cpu_data_type,
                           rnet_inputCpuTempPtrS[i],
                           rnet_input_data_type[i],
                           NULL,
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = rnet_inputCpuTempPtrS[i];
        } else {
          cnrtTransDataOrder(rnet_inputCpuPtrS[i],
                           cpu_data_type,
                           rnet_inputCpuTempPtrS[i],
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = rnet_inputCpuTempPtrS[i];
        }
        cnrtMemcpy(rnet_inputMluPtrS[i],
                   temp_ptr,
                   rnet_inputSizeS[i],
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
      }

      cnrtPlaceNotifier(rnet_notifierBeginning, queue_);
      CNRT_CHECK(cnrtInvokeRuntimeContext(rnet_runtime_context,
            rnet_param, queue_, nullptr));
      cnrtPlaceNotifier(rnet_notifierEnd, queue_);
      CNRT_CHECK(cnrtSyncQueue(queue_));
      cnrtNotifierDuration(rnet_notifierBeginning, rnet_notifierEnd, &eventInterval);
      AddRuntime(eventInterval);
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      int latency_ms = std::chrono::duration_cast<TimeDuration_ms>(t2 - t1).count();
      AddLatency(latency_ms);

      cnrtDataType_t* rnet_output_data_type = nullptr;
      cnrtGetOutputDataType(&rnet_output_data_type, &rnet_outputNum, rnet_function);

      for (int i = 0; i < rnet_outputNum; i++) {
        cnrtMemcpy(rnet_outputCpuTempPtrS[i],
                   rnet_outputMluPtrS[i],
                   rnet_outputSizeS[i],
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        int tmp_out_shape[4];
        int tmp_out_order[4] = {0, 3, 1, 2};
        if (i == 0) {
          tmp_out_shape[0] = rnet_box_shape[0];
          tmp_out_shape[1] = rnet_box_shape[2];
          tmp_out_shape[2] = rnet_box_shape[3];
          tmp_out_shape[3] = rnet_box_shape[1];
        } else if (i == 1) {
          tmp_out_shape[0] = rnet_cls_shape[0];
          tmp_out_shape[1] = rnet_cls_shape[2];
          tmp_out_shape[2] = rnet_cls_shape[3];
          tmp_out_shape[3] = rnet_cls_shape[1];
        } else {
          LOG(ERROR) << "Rnet Out Num error!";
          exit(-1);
        }
        if (rnet_output_data_type[i] != CNRT_FLOAT32) {
          cnrtTransOrderAndCast(rnet_outputCpuTempPtrS[i],
                           rnet_output_data_type[i],
                           rnet_outputCpuPtrS[i],
                           CNRT_FLOAT32,
                           NULL,
                           4,
                           tmp_out_shape,
                           tmp_out_order);
        } else {
          cnrtTransDataOrder(rnet_outputCpuTempPtrS[i],
                             CNRT_FLOAT32,
                             rnet_outputCpuPtrS[i],
                             4,
                             tmp_out_shape,
                             tmp_out_order);
        }
      }

      /* copy the output layer to a vector*/
      const float *begin1 = rnet_output_cpu[1];
      int out1_rem_size =
          rem * rnet_cls_shape[1] * rnet_cls_shape[2] * rnet_cls_shape[3];
      const float *end1 = out1_rem_size + begin1;
      vector<float> rnet_cls(begin1, end1);
      rnet_cls_all.insert(rnet_cls_all.end(), rnet_cls.begin(), rnet_cls.end());

      const float *begin0 = rnet_output_cpu[0];
      int out0_rem_size =
          rem * rnet_box_shape[1] * rnet_box_shape[2] * rnet_box_shape[3];
      const float *end0 = out0_rem_size + begin0;
      vector<float> rnet_regs(begin0, end0);
      rnet_regs_all.insert(rnet_regs_all.end(), rnet_regs.begin(),
                           rnet_regs.end());
    }

    filterOutBoxes.clear();
    filteroutBoundingBox(totalBoxes, rnet_regs_all, rnet_box_shape,
                         rnet_cls_all, rnet_cls_shape, vector<float>(),
                         vector<int>(), R_thres, &filterOutBoxes);
    nms(&filterOutBoxes, 0.7, UNION, &totalBoxes);

    if (totalBoxes.size() > 0) {
      vector<BoundingBox> globalFilterBoxes(totalBoxes);
      totalBoxes.clear();
      for (int i = 0; i < globalFilterBoxes.size(); i++) {
        if (globalFilterBoxes.at(i).x1 > 0 &&
            globalFilterBoxes.at(i).x1 <
                globalFilterBoxes.at(i).x2 - min_size &&
            globalFilterBoxes.at(i).y1 > 0 &&
            globalFilterBoxes.at(i).y1 <
                globalFilterBoxes.at(i).y2 - min_size &&
            globalFilterBoxes.at(i).x2 < img_W &&
            globalFilterBoxes.at(i).y2 < img_H) {
          totalBoxes.push_back(globalFilterBoxes.at(i));
        }
      }
    }
  }
  cnrtDestroyNotifier(&rnet_notifierBeginning);
  cnrtDestroyNotifier(&rnet_notifierEnd);
}

void FaceDetector::doOnet() {
  vector<cv::Mat> n_channels;
  vector<BoundingBox> filterOutBoxes;
  vector<BoundingBox> nmsOutBoxes;

  cnrtNotifier_t onet_notifierBeginning, onet_notifierEnd;
  cnrtCreateNotifier(&onet_notifierBeginning);
  cnrtCreateNotifier(&onet_notifierEnd);
  float eventInterval = 0;
  int tmp_dim_order[4] = {0, 2, 3, 1};
  int *dim_order = tmp_dim_order;

  if (totalBoxes.size() > 0) {
    vector<float> onet_cls_all, onet_regs_all, onet_points_all;
    onet_cls_all.clear();
    onet_regs_all.clear();
    onet_points_all.clear();
    const int batchsize = onet_in_shape[0];
    const int loop_num = totalBoxes.size() / batchsize;
    const int rem = totalBoxes.size() % batchsize;
    vector<BoundingBox> batchBoxs;

    wrapInputLayer(onet_in_shape, onet_input_cpu[0], &n_channels);
    for (int i = 0; i < loop_num; i++) {
      batchBoxs.clear();
      for (int j = 0; j < batchsize; j++) {
        batchBoxs.push_back(totalBoxes.at(j + i * batchsize));
      }

      buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(48, 48),
                         &n_channels);
      cnrtDataType_t* onet_input_data_type = nullptr;
      TimePoint t1 = std::chrono::high_resolution_clock::now();
      cnrtGetInputDataType(&onet_input_data_type, &onet_inputNum, onet_function);
      for (int i = 0; i < onet_inputNum; i++) {
        void* temp_ptr = nullptr;
        int tmp_shape[] = {onet_in_shape[0],
                           onet_in_shape[1],
                           onet_in_shape[2],
                           onet_in_shape[3]};
        cnrtDataType_t cpu_data_type = CNRT_FLOAT32;
        if (FLAGS_first_conv) {
          cpu_data_type = CNRT_UINT8;
        }
        if (onet_input_data_type[i] != cpu_data_type) {
          cnrtTransOrderAndCast(onet_inputCpuPtrS[i],
                           cpu_data_type,
                           onet_inputCpuTempPtrS[i],
                           onet_input_data_type[i],
                           NULL,
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = onet_inputCpuTempPtrS[i];
        } else {
          cnrtTransDataOrder(onet_inputCpuPtrS[i],
                           cpu_data_type,
                           onet_inputCpuTempPtrS[i],
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = onet_inputCpuTempPtrS[i];
        }
        cnrtMemcpy(onet_inputMluPtrS[i],
                   temp_ptr,
                   onet_inputSizeS[i],
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
      }
      cnrtPlaceNotifier(onet_notifierBeginning, queue_);
      CNRT_CHECK(cnrtInvokeRuntimeContext(onet_runtime_context,
            onet_param, queue_, nullptr));
      cnrtPlaceNotifier(onet_notifierEnd, queue_);
      CNRT_CHECK(cnrtSyncQueue(queue_));
      cnrtNotifierDuration(onet_notifierBeginning, onet_notifierEnd, &eventInterval);
      AddRuntime(eventInterval);
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      int latency_ms = std::chrono::duration_cast<TimeDuration_ms>(t2 - t1).count();
      AddLatency(latency_ms);

      cnrtDataType_t* onet_output_data_type = nullptr;
      cnrtGetOutputDataType(&onet_output_data_type, &onet_outputNum, onet_function);
      for (int i = 0; i < onet_outputNum; i++) {
        cnrtMemcpy(onet_outputCpuTempPtrS[i],
                   onet_outputMluPtrS[i],
                   onet_outputSizeS[i],
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        int tmp_out_shape[4];
        int tmp_out_order[4] = {0, 3, 1, 2};
        if (i == 0) {
          tmp_out_shape[0] = onet_box_shape[0];
          tmp_out_shape[1] = onet_box_shape[2];
          tmp_out_shape[2] = onet_box_shape[3];
          tmp_out_shape[3] = onet_box_shape[1];
        } else if (i == 1) {
          tmp_out_shape[0] = onet_points_shape[0];
          tmp_out_shape[1] = onet_points_shape[2];
          tmp_out_shape[2] = onet_points_shape[3];
          tmp_out_shape[3] = onet_points_shape[1];
        } else if (i == 2) {
          tmp_out_shape[0] = onet_cls_shape[0];
          tmp_out_shape[1] = onet_cls_shape[2];
          tmp_out_shape[2] = onet_cls_shape[3];
          tmp_out_shape[3] = onet_cls_shape[1];
        } else {
          LOG(ERROR) << "Rnet Out Num error!";
          exit(-1);
        }
        if (onet_output_data_type[i] != CNRT_FLOAT32) {
          cnrtTransOrderAndCast(onet_outputCpuTempPtrS[i],
                           onet_output_data_type[i],
                           onet_outputCpuPtrS[i],
                           CNRT_FLOAT32,
                           NULL,
                           4,
                           tmp_out_shape,
                           tmp_out_order);
        } else {
          cnrtTransDataOrder(onet_outputCpuTempPtrS[i],
                            CNRT_FLOAT32,
                            onet_outputCpuPtrS[i],
                            4,
                            tmp_out_shape,
                            tmp_out_order);
        }
      }
      /* copy the output layer to a vector*/
      const float *begin2 = onet_output_cpu[2];
      const float *end2 = onet_out_size[2] + begin2;
      vector<float> onet_cls(begin2, end2);
      onet_cls_all.insert(onet_cls_all.end(), onet_cls.begin(), onet_cls.end());

      const float *begin0 = onet_output_cpu[0];
      const float *end0 = onet_out_size[0] + begin0;
      vector<float> onet_regs(begin0, end0);
      onet_regs_all.insert(onet_regs_all.end(), onet_regs.begin(),
                           onet_regs.end());

      const float *begin1 = onet_output_cpu[1];
      const float *end1 = onet_out_size[1] + begin1;
      vector<float> onet_points(begin1, end1);
      onet_points_all.insert(onet_points_all.end(), onet_points.begin(),
                             onet_points.end());
    }
    if (rem > 0) {
      batchBoxs.clear();
      for (int j = 0; j < rem; j++) {
        batchBoxs.push_back(totalBoxes.at(j + loop_num * batchsize));
      }
      for (int j = rem; j < batchsize; j++) {
        batchBoxs.push_back(totalBoxes.at(totalBoxes.size() - 1));
      }

      buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(48, 48),
                         &n_channels);
      TimePoint t1 = std::chrono::high_resolution_clock::now();
      cnrtDataType_t* onet_input_data_type = nullptr;
      cnrtGetInputDataType(&onet_input_data_type, &onet_inputNum, onet_function);
      for (int i = 0; i < onet_inputNum; i++) {
        void* temp_ptr = nullptr;
        int tmp_shape[] = {onet_in_shape[0],
                           onet_in_shape[1],
                           onet_in_shape[2],
                           onet_in_shape[3]};
        cnrtDataType_t cpu_data_type = CNRT_FLOAT32;
        if (FLAGS_first_conv) {
          cpu_data_type = CNRT_UINT8;
        }
        if (onet_input_data_type[i] != cpu_data_type) {
          cnrtTransOrderAndCast(onet_inputCpuPtrS[i],
                           cpu_data_type,
                           onet_inputCpuTempPtrS[i],
                           onet_input_data_type[i],
                           NULL,
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = onet_inputCpuTempPtrS[i];
        } else {
          cnrtTransDataOrder(onet_inputCpuPtrS[i],
                           cpu_data_type,
                           onet_inputCpuTempPtrS[i],
                           4,
                           tmp_shape,
                           dim_order);
          temp_ptr = onet_inputCpuTempPtrS[i];
        }
        cnrtMemcpy(onet_inputMluPtrS[i],
                   temp_ptr,
                   onet_inputSizeS[i],
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
      }

      cnrtPlaceNotifier(onet_notifierBeginning, queue_);
      CNRT_CHECK(cnrtInvokeRuntimeContext(onet_runtime_context,
            onet_param, queue_, nullptr));
      cnrtPlaceNotifier(onet_notifierEnd, queue_);
      CNRT_CHECK(cnrtSyncQueue(queue_));
      cnrtNotifierDuration(onet_notifierBeginning, onet_notifierEnd, &eventInterval);
      AddRuntime(eventInterval);
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      int latency_ms = std::chrono::duration_cast<TimeDuration_ms>(t2 - t1).count();
      AddLatency(latency_ms);

      cnrtDataType_t* onet_output_data_type = nullptr;
      cnrtGetOutputDataType(&onet_output_data_type, &onet_outputNum, onet_function);

      for (int i = 0; i < onet_outputNum; i++) {
        cnrtMemcpy(onet_outputCpuTempPtrS[i],
                   onet_outputMluPtrS[i],
                   onet_outputSizeS[i],
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        int tmp_out_shape[4];
        int tmp_out_order[4] = {0, 3, 1, 2};
        if (i == 0) {
            tmp_out_shape[0] = onet_box_shape[0];
            tmp_out_shape[1] = onet_box_shape[2];
            tmp_out_shape[2] = onet_box_shape[3];
            tmp_out_shape[3] = onet_box_shape[1];
        } else if (i == 1) {
            tmp_out_shape[0] = onet_points_shape[0];
            tmp_out_shape[1] = onet_points_shape[2];
            tmp_out_shape[2] = onet_points_shape[3];
            tmp_out_shape[3] = onet_points_shape[1];
        } else if (i == 2) {
            tmp_out_shape[0] = onet_cls_shape[0];
            tmp_out_shape[1] = onet_cls_shape[2];
            tmp_out_shape[2] = onet_cls_shape[3];
            tmp_out_shape[3] = onet_cls_shape[1];
        } else {
            LOG(ERROR) << "Rnet Out Num error!";
            exit(-1);
        }
        if (onet_output_data_type[i] != CNRT_FLOAT32) {
          cnrtTransOrderAndCast(onet_outputCpuTempPtrS[i],
                           onet_output_data_type[i],
                           onet_outputCpuPtrS[i],
                           CNRT_FLOAT32,
                           NULL,
                           4,
                           tmp_out_shape,
                           tmp_out_order);
        } else {
          cnrtTransDataOrder(onet_outputCpuTempPtrS[i],
                            CNRT_FLOAT32,
                            onet_outputCpuPtrS[i],
                            4,
                            tmp_out_shape,
                            tmp_out_order);
        }
      }
      /* copy the output layer to a vector*/
      const float *begin2 = onet_output_cpu[2];
      int out2_rem_size =
          rem * onet_cls_shape[1] * onet_cls_shape[2] * onet_cls_shape[3];
      const float *end2 = out2_rem_size + begin2;
      vector<float> onet_cls(begin2, end2);
      onet_cls_all.insert(onet_cls_all.end(), onet_cls.begin(), onet_cls.end());

      const float *begin0 = onet_output_cpu[0];
      int out0_rem_size =
          rem * onet_box_shape[1] * onet_box_shape[2] * onet_box_shape[3];
      const float *end0 = out0_rem_size + begin0;
      vector<float> onet_regs(begin0, end0);
      onet_regs_all.insert(onet_regs_all.end(), onet_regs.begin(),
                           onet_regs.end());

      const float *begin1 = onet_output_cpu[1];
      int out1_rem_size = rem * onet_points_shape[1] * onet_points_shape[2] *
                          onet_points_shape[3];
      const float *end1 = out1_rem_size + begin1;
      vector<float> onet_points(begin1, end1);
      onet_points_all.insert(onet_points_all.end(), onet_points.begin(),
                             onet_points.end());
    }

    filterOutBoxes.clear();
    filteroutBoundingBox(totalBoxes, onet_regs_all, onet_box_shape,
                         onet_cls_all, onet_cls_shape, onet_points_all,
                         onet_points_shape, O_thres, &filterOutBoxes);
    nms(&filterOutBoxes, 0.7, MIN, &totalBoxes);

    if (totalBoxes.size() > 0) {
      vector<BoundingBox> globalFilterBoxes(totalBoxes);
      totalBoxes.clear();
      for (int i = 0; i < globalFilterBoxes.size(); i++) {
        if (globalFilterBoxes.at(i).x1 > 0 &&
            globalFilterBoxes.at(i).x1 <
                globalFilterBoxes.at(i).x2 - min_size &&
            globalFilterBoxes.at(i).y1 > 0 &&
            globalFilterBoxes.at(i).y1 <
                globalFilterBoxes.at(i).y2 - min_size &&
            globalFilterBoxes.at(i).x2 < img_W &&
            globalFilterBoxes.at(i).y2 < img_H) {
          totalBoxes.push_back(globalFilterBoxes.at(i));
        }
      }
    }
  }
  cnrtDestroyNotifier(&onet_notifierBeginning);
  cnrtDestroyNotifier(&onet_notifierEnd);
}
vector<FaceDetector::BoundingBox> FaceDetector::detect(const cv::Mat &img,
                                                       int minsize,
                                                       float p_thres,
                                                       float r_thres,
                                                       float o_thres) {
  P_thres = p_thres;
  R_thres = r_thres;
  O_thres = o_thres;
  min_size = minsize;

  cv::Mat sample_normalized;
  cv::Mat img_tmp;

  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, m_nDevNum));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  if (FLAGS_first_conv) {
    sample_normalized = img.t();
  } else {
    img.convertTo(img_tmp, CV_32FC3, img_var, -img_mean * img_var);
    sample_normalized = img_tmp.t();
  }
  img_H = sample_normalized.rows;
  img_W = sample_normalized.cols;

  // split the input image
  sample_norm_channels.clear();
  cv::split(sample_normalized, sample_norm_channels);


  // BGR to RGB
  cv::Mat tmp = sample_norm_channels[0];
  sample_norm_channels[0] = sample_norm_channels[2];
  sample_norm_channels[2] = tmp;
  int pad_type = CV_8UC1;
  if (FLAGS_first_conv)
    pad_type = CV_8UC1;
  else
    pad_type = CV_32FC1;
  cv::Mat pad_channel(img_H, img_W, pad_type, cv::Scalar(0));
  sample_norm_channels.push_back(pad_channel);

  for (int i = 0; i < 10; i++) {
    sem_post(&sem_in[i]);
  }
  for (int i = 0; i < 10; i++) {
    sem_wait(&sem_out[i]);
  }

  // Pnet
  nmsGlobal();

  // Rnet
  doRnet();

  //// Onet
  doOnet();

  for (int i = 0; i < totalBoxes.size(); i++) {
    std::swap(totalBoxes[i].x1, totalBoxes[i].y1);
    std::swap(totalBoxes[i].x2, totalBoxes[i].y2);
    for (int k = 0; k < 5; k++) {
      std::swap(totalBoxes[i].points_x[k], totalBoxes[i].points_y[k]);
    }
  }
  return totalBoxes;
}

struct FaceDetectorOutput {
  vector<FaceDetector::BoundingBox> res;
  int dx;
  int dy;
  float scale;
};

string getTime() {
  time_t timep;
  time(&timep);
  char tmp[64];
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
  return tmp;
}



struct detection_results {
  void add(int index, const FaceDetectorOutput &output) {
    _mtx.lock();
    results[index] = output;
    _mtx.unlock();
  }

  void save(const string &benchmark_output, const vector<string> &lines) {
    _mtx.lock();
    std::ofstream fd(benchmark_output.c_str(), std::ios::app);

    map<int, FaceDetectorOutput>::iterator iter;
    iter = results.begin();

    while (iter != results.end()) {
      int index = iter->first;
      FaceDetectorOutput output = iter->second;
      vector<FaceDetector::BoundingBox> &res = output.res;
      int dx = output.dx;
      int dy = output.dy;
      float scale = output.scale;

      fd << lines[index] << std::endl;
      fd << res.size() << std::endl;

      for (int k = 0; k < res.size(); k++) {
        float x1 = (res[k].x1 - dx) / scale;
        float y1 = (res[k].y1 - dy) / scale;
        float x2 = (res[k].x2 - dx) / scale;
        float y2 = (res[k].y2 - dy) / scale;
        fd << x1 << " " << y1 << " " << x2 - x1 << " " << y2 - y1 << " "
           << res[k].score << std::endl;
      }
      iter++;
    }
    _mtx.unlock();
  }

  map<int, FaceDetectorOutput> results;
  mutex _mtx;
};

void *face_detector_task(int nDevNum, int nDevChannel, const vector<string> &lines,
                         int from, int to, float *mluTime, int* latency,
                         detection_results *results) {
  FaceDetector *fd = new FaceDetector(nDevNum, nDevChannel % 4);

  // get .cambricon model
  fd->Open();

  int index = 0;

  for (index = from; index < to;) {
    string filepath = lines[index];
    cv::Mat image = cv::imread(filepath.c_str());

    const int ih = image.rows;
    const int iw = image.cols;

    if (ih == 0 || iw == 0) {
      printf("open file:%s failed\n", filepath.c_str());
      break;
    }
    const int h = 1080;
    const int w = 1920;

    float scale = 0;
    if (ih > h || iw > w) {
      scale = std::min(w * 1.0 / iw, h * 1.0 / ih);
    } else {
      // Note: do not remove this code, elsewise meanAp will drop 2%
      scale = std::min(w / iw, h / ih);
    }
    int nw = static_cast<int>(iw * scale);
    int nh = static_cast<int>(ih * scale);
    int dx = (w - nw) / 2;
    int dy = (h - nh) / 2;

    Mat im(Size(w, h), CV_8UC3, cv::Scalar::all(128));

    Mat dstroi = im(Rect(dx, dy, nw, nh));
    Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));
    resized.copyTo(dstroi);

    FaceDetectorOutput output;
    output.dx = dx;
    output.dy = dy;
    output.scale = scale;

    // start 3-stage detaction
    output.res = fd->detect(im, 30, 0.75, 0.85, 0.85);
    vector<FaceDetector::BoundingBox> res = output.res;
    for (int k = 0; k < res.size(); k++) {
      float x1 = (res[k].x1 - dx) / scale;
      float y1 = (res[k].y1 - dy) / scale;
      float x2 = (res[k].x2 - dx) / scale;
      float y2 = (res[k].y2 - dy) / scale;

      cv::rectangle(image, cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                    cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                    cv::Scalar(0, 255, 255), 2);

      for (int i = 0; i < 5; i++) {
        float x1 = (res[k].points_x[i] - dx) / scale;
        float y1 = (res[k].points_y[i] - dy) / scale;
        cv::circle(image, cv::Point(static_cast<int>(x1), static_cast<int>(y1)), 2,
                   cv::Scalar(0, 255, 255), 2);
      }
    }
    stringstream ss;
    ss << "mtcnn_cnrt_orgin_" << index << ".jpg";
    cv::imwrite(ss.str(), image);

    results->add(index, output);
    index++;
  }
  *mluTime = fd->GetRuntime();
  *latency = fd->GetLatency();
  delete fd;
  return NULL;
}

int mtcnn_fddb_benchmark_multicore() {
  vector<float> mluTime;
  vector<int> latencys;
  string benchmark_output = "mtcnn.txt";
  remove(benchmark_output.c_str());

  vector<string> lines;
  {
    string line;
    if (!FLAGS_images.empty()) {
      std::fstream file(FLAGS_images);
      if (file.fail()) LOG(FATAL) << "failed to open image file!";
      while (getline(file, line)) {
        lines.push_back(line);
      }
      file.close();
    }
  }

  int nTotalFiles = lines.size();

  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }

  int nDevCount = deviceIds_.size();
  int nDevChannelCount = FLAGS_threads;
  int nTaskSize = nDevCount * nDevChannelCount;
  int nBlockSize = nTotalFiles / nTaskSize;

  detection_results results;

  vector<thread *> tasks;
  mluTime.resize(nTaskSize);
  tasks.resize(nTaskSize);
  latencys.resize(nTaskSize);
  int index = 0;

  Timer timer;
  for (int i = 0; i < nDevCount; i++) {
    for (int ch = 0; ch < nDevChannelCount; ch++) {
      int dev = deviceIds_[i];
      int from = index * nBlockSize;
      int to = (index + 1) * nBlockSize;

      if (index == nTaskSize - 1) {
        to = nTotalFiles;
      }

      tasks[index] = new thread(face_detector_task, dev, ch, std::ref(lines), from,
                                to, &mluTime[index], &latencys[index], &results);
      index += 1;
    }
  }

  for (auto t : tasks) {
    t->join();
  }

  timer.log("Total execution time");
  float execTime = timer.getDuration();

  results.save(benchmark_output, lines);

  float totalMluTime = std::accumulate(mluTime.begin(), mluTime.end(), 0);
  float totalLatencyMs = std::accumulate(latencys.begin(), latencys.end(), 0);
  float aveLatency = totalLatencyMs/nTotalFiles;
  float throughput = nTotalFiles*1e6/execTime;
  float hw_latency = totalMluTime / (nTotalFiles * 1000);
  LOG(INFO) << "latency: " << aveLatency;
  LOG(INFO) << "Throughput: " << throughput;
  LOG(INFO) << "HardwareLatency(ms): " << hw_latency;
  dumpJson(nTotalFiles, (-1), (-1), (-1), aveLatency * 1000, throughput, hw_latency);
  return 0;
}

int main(int argc, char **argv) {
  {
    const char * env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0)
      FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do detection using mtcnn-demo.\n"
      "Usage:\n"
      "    mtcnn-demo [FLAGS] image_list model_list\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/mtcnn/mtcnn-demo");
    return 1;
  }

  FaceDetector::Init();
  int ret = mtcnn_fddb_benchmark_multicore();
  FaceDetector::Destory();
  return ret;
}
