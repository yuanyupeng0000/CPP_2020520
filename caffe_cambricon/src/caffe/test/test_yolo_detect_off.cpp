// Copyright [2018] <cambricon>
#ifdef USE_MLU
#include <cnrt.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include"caffe/common.hpp"
#include "test_yolo_input_data.hpp"
#include "caffe/mlu/data_trans.hpp"
using std::ostringstream;
using std::vector;
using std::string;
DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
int main(int argc, char* argv[]) {
::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if ( argc < 4 ) {
    LOG(INFO) << "USAGE: " << argv[0] <<": <cambricon_file>" \
    << " <output_file> <function_name0> <function_name1> ...";
    exit(0);
  }
  cnrtInit(0);
  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  if (FLAGS_mludevice >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(FLAGS_mludevice, devNum) << "valid device count: " <<devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, FLAGS_mludevice);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  cnrtModel_t model;
  string fname = (string)argv[1];
  printf("load file: %s\n", fname.c_str());
  cnrtLoadModel(&model, fname.c_str());
  cnrtFunction_t function;

  for ( int n = 3 ; n < argc ; n++ ) {
      string name = (string) argv[n];
      cnrtCreateFunction(&function);
      cnrtExtractFunction(&function, model, name.c_str());
      // 3. get function's I/O DataDesc
      int inputNum, outputNum;
      int64_t* inputSizeArray;
      int64_t* outputSizeArray;
      cnrtDataType_t* inputDataTypeArray;
      cnrtDataType_t* outputDataTypeArray;
      CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray, &inputNum, function));
      CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray, &outputNum, function));
      CNRT_CHECK(cnrtGetInputDataType(&inputDataTypeArray, &inputNum, function));
      CNRT_CHECK(cnrtGetOutputDataType(&outputDataTypeArray, &outputNum, function));
      // 4. allocate I/O data space on CPU memory and prepare Input data
      void** inputCpuPtrS  = reinterpret_cast<void**>\
                             (malloc(sizeof(void*) * inputNum));
      void** inputSyncPtrS = reinterpret_cast<void**>\
                             (malloc(sizeof(void*) * inputNum));
      void** inputSyncTmpPtrS = reinterpret_cast<void**>\
                                (malloc(sizeof(void*) * inputNum));
      void** outputCpuPtrS = reinterpret_cast<void**>\
                             (malloc(sizeof(void*) * outputNum));
      void** outputSyncPtrS = reinterpret_cast<void**>\
                              (malloc(sizeof(void*) * outputNum));
      vector<float*> output_cpu;
      vector<int> in_count;
      vector<int> out_count;
      vector<vector<int>> inputShape;
      vector<vector<int>> outputShape;
      vector<int*> inputDimValues(inputNum, 0);
      vector<int> inputDimNumS(inputNum, 0);
      vector<int*> outputDimValues(outputNum, 0);
      vector<int> outputDimNumS(outputNum, 0);
      void** param = reinterpret_cast<void**>\
                     (malloc(sizeof(void*)*(inputNum + outputNum)));
      for ( int i = 0; i < inputNum; i++ ) {
        float* databuf;
        databuf = reinterpret_cast<float*>(malloc(sizeof(float) * inputSizeArray[i]));
        for (int j = 0; j < inputSizeArray[i]; ++j) {
          databuf[j]= yolo_input_data::input_cpu_data[j];
        }
        in_count.push_back(inputSizeArray[i]);
        inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);
        inputSyncPtrS[i] = reinterpret_cast<void*>(malloc(inputSizeArray[i]));
        inputSyncTmpPtrS[i] = reinterpret_cast<void*>(malloc(inputSizeArray[i] / 4 * 3));
        cnrtGetInputDataShape(&(inputDimValues[i]), &(inputDimNumS[i]), i, function);
        vector<int> in_shape;
        for (int idx = 0; idx < inputDimNumS[i]; idx++) {
          in_shape.push_back(inputDimValues[i][idx]);
        }
        inputShape.push_back(in_shape);
      }
      for ( int i = 0; i < outputNum; i++ ) {
        int op;
        float* outcpu;
        cnrtGetInputDataShape(&(outputDimValues[i]), &(outputDimNumS[i]), i, function);
        vector<int> out_shape;
        for (int idx = 0; idx < outputDimNumS[i]; idx++) {
          op = op * outputDimValues[i][idx];
          out_shape.push_back(outputDimValues[i][idx]);
        }
        outputShape.push_back(out_shape);
        outcpu = reinterpret_cast<float*>(malloc(op * sizeof(float)));
        out_count.push_back(op);
        output_cpu.push_back(outcpu);
        outputCpuPtrS[i] = reinterpret_cast<void*>(outcpu);
        outputSyncPtrS[i] =
          reinterpret_cast<void*>(malloc(outputSizeArray[i]));
      }
      // 5. allocate I/O data space on MLU memory and copy Input data
      // Only 1 batch so far
      void** inputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
      void** outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
      for ( int i = 0; i < inputNum; i++ ) {
        cnrtMalloc(&(inputMluPtrS[i]), inputSizeArray[i]);
        param[i] = inputMluPtrS[i];
      }
      for ( int i = 0; i < outputNum; i++ ) {
        cnrtMalloc(&(outputMluPtrS[i]), outputSizeArray[i]);
        param[inputNum + i] = outputMluPtrS[i];
      }
      // 6. create cnrt_queue and run function
      cnrtQueue_t cnrt_queue;
      cnrtCreateQueue(&cnrt_queue);
      cnrtDataType_t inputCpuDtype = CNRT_FLOAT32;
      cnrtDataType_t inputMluDtype = inputDataTypeArray[0];

      for (int i = 0; i < inputNum; i++) {
        bool useFirstConv = inputMluDtype == CNRT_UINT8 && inputShape[i][3] == 4;
        caffe::transAndCast(reinterpret_cast<void*>(inputCpuPtrS[i]), inputCpuDtype,
                     reinterpret_cast<void*>(inputSyncPtrS[i]), inputMluDtype,
                     reinterpret_cast<void*>(inputSyncTmpPtrS[i]),
                     caffe::to_cpu_shape(inputShape[i]),
                     useFirstConv, "CPU2MLU");

        CNRT_CHECK(cnrtMemcpy(inputMluPtrS[i], inputSyncPtrS[i],
              inputSizeArray[i], CNRT_MEM_TRANS_DIR_HOST2DEV));
      }
      //  create start_event and end_event
      cnrtNotifier_t notifierBeginning, notifierEnd;
      cnrtCreateNotifier(&notifierBeginning);
      cnrtCreateNotifier(&notifierEnd);
      float event_time_use;
      //  run MLU
      cnrtRuntimeContext_t rt_ctx;

      if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
        LOG(FATAL)<< "Failed to create runtime context";
      }

      // set device ordinal. if not set, a random device will be used
      cnrtSetRuntimeContextDeviceId(rt_ctx, 0);

      // Instantiate the runtime context on actual MLU device
      // All cnrtSetRuntimeContext* interfaces must be caller prior to cnrtInitRuntimeContext
      if (cnrtInitRuntimeContext(rt_ctx, nullptr) != CNRT_RET_SUCCESS) {
        LOG(FATAL)<< "Failed to initialize runtime context";
      }
      //  place start_event to cnrt_queue
      cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
      CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx, param, cnrt_queue, nullptr));
      //  place end_event to cnrt_queue
      cnrtPlaceNotifier(notifierEnd, cnrt_queue);
      if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
        //  get start_event and end_event elapsed time
        cnrtNotifierDuration(notifierBeginning, notifierEnd, &event_time_use);
        LOG(INFO) << " execution time: " << event_time_use;
      } else {
        LOG(INFO) << " SyncQueue error ";
      }
      cnrtDataType_t outputCpuDtype = CNRT_FLOAT32;
      cnrtDataType_t outputMluDtype = outputDataTypeArray[0];
      for (int i = 0; i < outputNum; i++) {
        CNRT_CHECK(cnrtMemcpy(outputSyncPtrS[i], outputMluPtrS[i],
              outputSizeArray[i], CNRT_MEM_TRANS_DIR_DEV2HOST));
        caffe::transAndCast(reinterpret_cast<void*>(outputSyncPtrS[i]),
                     outputMluDtype,
                     reinterpret_cast<void*>(outputCpuPtrS[i]),
                     outputCpuDtype,
                     nullptr,
                     outputShape[i],
                     false,
                     "MLU2CPU");
      }
      for ( int i = 0; i < outputNum; i++ ) {
          LOG(INFO) << "copying output data of output " <<
            i <<" function " << argv[n];
          ostringstream ss;
          if ( outputNum > 1 ) {
              ss << argv[2] << "_"<< argv[n] << i;
          } else {
              ss << argv[2] << "_"<< argv[n];
          }
          string output_name = ss.str();
          LOG(INFO) << "writting output file of segment " << argv[n]
            << " output: "<< i <<" output file name: " << output_name;
          std::ofstream fout(output_name, std::ios::out);
          fout << std::flush;
          for (int j = 0; j < out_count[i]; ++j) {
              fout << output_cpu[i][j] << std::endl;
          }
          fout << std::flush;
          fout.close();
      }
      for ( auto flo : output_cpu ) {
          free(flo);
      }
      output_cpu.clear();
      // 8. free memory space
      free(inputCpuPtrS);
      free(outputCpuPtrS);
      free(inputSyncPtrS);
      free(inputSyncTmpPtrS);
      free(outputSyncPtrS);
      cnrtFreeArray(inputMluPtrS, inputNum);
      cnrtFreeArray(outputMluPtrS, outputNum);
      cnrtDestroyQueue(cnrt_queue);
      cnrtDestroyFunction(function);
  }
  cnrtUnloadModel(model);
  cnrtDestroy();
  return 0;
}
#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif
