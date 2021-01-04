// Copyright [2018] <cambricon>
#ifdef USE_MLU
#include <fstream>
#include <string>
#include <vector>
#include "offengine.hpp"
#include "caffe/mlu/data_trans.hpp"
using std::vector;
using std::string;

namespace caffetool {
  OfflineEngine::OfflineEngine(const string& name):
    model_name_(name),
    model(NULL),
    function(NULL),
    icpuptrs(reinterpret_cast<void**>(NULL)),
    ocpuptrs(reinterpret_cast<void**>(NULL)),
    imluptrs(reinterpret_cast<void**>(NULL)) {
  }

  OfflineEngine::~OfflineEngine() {
    if (model != NULL)
      cnrtUnloadModel(model);

    for (int i = 0; i < inum_; i++) {
      delete in_shape_[i];
      free(icpuptrs[i]);
    }

    for (int i = 0; i < onum_; i++) {
      delete out_shape_[i];
      free(ocpuptrs[i]);
    }

    cnrtDestroy();
  }

  const vector<int> OfflineEngine::in_shape(int index) {
    CHECK_GE(index, 0);
    CHECK_LE(index, in_shape_.size() );
    return *in_shape_[index];
  }

  const vector<int> OfflineEngine::out_shape(int index) {
    CHECK_GE(index, 0);
    CHECK_LE(index, out_shape_.size());
    return *out_shape_[index];
  }

  void OfflineEngine::OpenDevice(const int& deviceId) {
    cnrtInit(0);
    unsigned devNum;
    cnrtGetDeviceCount(&devNum);
    if (deviceId >= 0) {
      CHECK_NE(devNum, 0) << "No device found";
      CHECK_LT(deviceId, devNum) << "valid device count: " <<devNum;
    } else {
      LOG(FATAL) << "Invalid device number";
    }
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, deviceId);
    cnrtSetCurrentDevice(dev);
    deviceId_ = deviceId;
  }

  void OfflineEngine::LoadModel() {
    CHECK(!model_name_.empty());
    LOG(INFO) << "load model " << model_name_;
    CNRT_CHECK(cnrtLoadModel(&model, model_name_.c_str()));
  }

  void OfflineEngine::CreateFunc() {
    CHECK(model != NULL);
    CHECK(!subnet_name_.empty());

    CNRT_CHECK(cnrtCreateFunction(&function));
    CNRT_CHECK(cnrtExtractFunction(&function, model, subnet_name_.c_str()));

    CNRT_CHECK(cnrtGetInputDataSize(&inputSizeArray, &inum_, function));
    CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeArray, &onum_, function));
    CNRT_CHECK(cnrtGetInputDataType(&inputDataTypeArray, &inum_, function));
    CNRT_CHECK(cnrtGetOutputDataType(&outputDataTypeArray, &onum_, function));
    for (int i = 0; i < inum_; i++) {
      cnrtGetInputDataShape(&(inputDimValues[i]), &(inputDimNumS[i]), i, function);
      vector<int>* shape = new vector<int>;
      for (int idx = 0; idx < inputDimNumS[i]; idx++) {
        shape->push_back(inputDimValues[i][idx]);
      }
      in_shape_.push_back(shape);
    }

    for (int i = 0; i < onum_; i++) {
      cnrtGetInputDataShape(&(outputDimValues[i]), &(outputDimNumS[i]), i, function);
      vector<int>* shape = new vector<int>;
      for (int idx = 0; idx < outputDimNumS[i]; idx++) {
        shape->push_back(outputDimValues[i][idx]);
      }
      out_shape_.push_back(shape);
    }
  }

  void OfflineEngine::AllocMemory() {
    // allocate I/O data space on CPU memory and prepare Input data
    const size_t sz_vptr = sizeof(void*);
    icpuptrs = reinterpret_cast<void**> (malloc(sz_vptr * inum_));
    ocpuptrs = reinterpret_cast<void**> (malloc(sz_vptr * onum_));
    inputSyncPtrS = reinterpret_cast<void**>(malloc(sz_vptr * inum_));
    inputSyncTmpPtrS = reinterpret_cast<void**>(malloc(sz_vptr * inum_));
    outputSyncPtrS = reinterpret_cast<void**>(malloc(sz_vptr * onum_));
    param = reinterpret_cast<void**> (malloc(sz_vptr * (inum_ + onum_)));

    AllocateInputMemory();
    AllocateOutputMemory();
    AllocateMLUMemroy();
  }

  void OfflineEngine::AllocateInputMemory() {
    for ( int i = 0; i < inum_; i++ ) {
      float* icpu;
      icpu = reinterpret_cast<float*>(malloc(sizeof(float) * inputSizeArray[i]));
      icpuptrs[i] = reinterpret_cast<void*>(icpu);
      inputSyncPtrS[i] = reinterpret_cast<void*>(malloc(inputSizeArray[i]));
      inputSyncTmpPtrS[i] = reinterpret_cast<void*>(malloc(inputSizeArray[i] / 4 * 3));
    }
  }

  void OfflineEngine::AllocateOutputMemory() {
    for ( int i = 0; i < onum_; i++ ) {
      float* ocpu;
      ocpu = reinterpret_cast<float*>(malloc(outputSizeArray[i] * sizeof(float)));
      ocpuptrs[i] = reinterpret_cast<void*>(ocpu);
      outputSyncPtrS[i] =
          reinterpret_cast<void*>(malloc(outputSizeArray[i]));
    }
  }

  void OfflineEngine::AllocateMLUMemroy() {
    for ( int i = 0; i < inum_; i++ ) {
      cnrtMalloc(&(imluptrs[i]), inputSizeArray[i]);
      param[i] = imluptrs[i];
    }

    for ( int i = 0; i < onum_; i++ ) {
      cnrtMalloc(&(omluptrs[i]), outputSizeArray[i]);
      param[inum_ + i] = omluptrs[i];
    }
  }

  int OfflineEngine::Run() {
    cnrtQueue_t cnrt_queue;
    cnrtCreateQueue(&cnrt_queue);
    cnrtDataType_t inputCpuDtype = CNRT_FLOAT32;
    cnrtDataType_t inputMluDtype = inputDataTypeArray[0];
    for (int i = 0; i < inum_; i++) {
      bool useFirstConv = inputMluDtype == CNRT_UINT8 && in_shape(i)[3] == 4;
      caffe::transAndCast(reinterpret_cast<void*>(icpuptrs[i]), inputCpuDtype,
                   reinterpret_cast<void*>(inputSyncPtrS[i]), inputMluDtype,
                   reinterpret_cast<void*>(inputSyncTmpPtrS[i]),
                   caffe::to_cpu_shape(in_shape(i)),
                   useFirstConv, "CPU2MLU");

      CNRT_CHECK(cnrtMemcpy(imluptrs[i], inputSyncPtrS[i],
            inputSizeArray[i], CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    //  create start_event and end_event
    cnrtNotifier_t notifierBeginning, notifierEnd;
    cnrtCreateNotifier(&notifierBeginning);
    cnrtCreateNotifier(&notifierEnd);
    float time_use;

    cnrtRuntimeContext_t rt_ctx;
    if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
      LOG(FATAL)<< "Failed to create runtime context";
    }
    cnrtSetRuntimeContextDeviceId(rt_ctx, deviceId_);
    cnrtInitRuntimeContext(rt_ctx, NULL);
    //  place start_event into cnrt_queue
    cnrtPlaceNotifier(notifierBeginning, cnrt_queue);
    cnrtInvokeRuntimeContext(rt_ctx, param, cnrt_queue, nullptr);
    //  place end_event into cnrt_queue
    cnrtPlaceNotifier(notifierEnd, cnrt_queue);

    int syncfailed = 0;
    if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
      //  get start_event and end_event elapsed time
      cnrtNotifierDuration(notifierBeginning, notifierEnd, &time_use);
      LOG(INFO) << "execution time(us): " << time_use;
    } else {
      syncfailed = 1;
      LOG(INFO) << "synccnrt_queue error ";
    }

    cnrtDestroyQueue(cnrt_queue);
    cnrtDestroyNotifier(&notifierBeginning);
    cnrtDestroyNotifier(&notifierEnd);

    return syncfailed;
  }

  void OfflineEngine::CopyOut(const string& filename) {
    cnrtDataType_t outputCpuDtype = CNRT_FLOAT32;
    cnrtDataType_t outputMluDtype = outputDataTypeArray[0];
    for ( int i = 0; i < onum_; i++ ) {
      CNRT_CHECK(cnrtMemcpy(outputSyncPtrS[i], omluptrs[i],
            outputSizeArray[i], CNRT_MEM_TRANS_DIR_DEV2HOST));
      caffe::transAndCast(reinterpret_cast<void*>(outputSyncPtrS[i]),
                   outputMluDtype,
                   reinterpret_cast<void*>(ocpuptrs[i]),
                   outputCpuDtype,
                   nullptr,
                   out_shape(i),
                   false,
                   "MLU2CPU");
    }
    for ( int i = 0; i < onum_; i++ ) {
      LOG(INFO) << "copy out output " << i <<" of " << subnet_name_;

      string name;
      name = filename + "_" + subnet_name_;
      if ( onum_ > 1 )
        name += ToString(i);

      LOG(INFO) << "write output " << i << " of " << subnet_name_
                <<" to file : " << name;

      std::ofstream fout(name, std::ios::out);
      for (int j = 0; j < out_count[i]; j++) {
        fout << output_cpu[i][j] << std::endl;
      }
      fout.close();
    }
  }

  void OfflineEngine::FreeResources() {
    for ( auto flo : output_cpu ) {
      free(flo);
    }
    output_cpu.clear();
    free(icpuptrs);
    free(ocpuptrs);
    cnrtFreeArray(imluptrs, inum_);
    cnrtFreeArray(omluptrs, onum_);
    cnrtDestroyFunction(function);
  }
}  // namespace caffetool
#endif
