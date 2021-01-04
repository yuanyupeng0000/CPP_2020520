#ifdef USE_MLU
#include <assert.h>
#include <cnrt.h>
#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include"caffe/common.hpp"
#include "caffe/mlu/data_trans.hpp"
using std::ostringstream;
using std::vector;
using std::string;
using size_t = std::size_t;

DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
struct NParray {
  NParray(float* d, std::vector<int> s) {
    data = d;
    int sz = 1;
    for (size_t i = 0; i < s.size(); ++i) {
      sz *= s[i];
    }
    size = sz;
    shape = s;
  }
  float& operator[](std::vector<int> idxs) {
    assert(idxs.size() == shape.size());
    int tmp = 1;
    int id = 0;
    for (int i = idxs.size()-1; i >= 0; --i) {
      id += idxs[i] * tmp;
      tmp *= shape[i];
    }
    return data[id];
  }
  void print() {
    int jj = shape.back();
    int ii = size / jj;
    int cnt = 0;
    for (int i = 0; i < ii; ++i) {
      for (int j = 0; j < jj; ++j) {
        LOG(INFO) << data[cnt++] << " ";
      }
    }
  }
  float* data;
  int size;
  std::vector<int> shape;
};
void np_reshape(NParray* np_array, std::vector<int> shape) {
  int sz = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    sz *= shape[i];
  }
  assert(sz == (*np_array).size);
  (*np_array).shape = shape;
}

void print_vec(std::vector<int> const& vec) {
  LOG(INFO) << "[ ";
  for (size_t i = 0; i < vec.size(); ++i) {
    LOG(INFO) << vec[i] << ' ';
  }
  LOG(INFO) << "]\n";
}

void np_transpose(NParray* np_array, std::vector<int> trans) {
  assert(trans.size() == (*np_array).shape.size());
  float* tmp_data = new float[(*np_array).size];
  std::vector<int> idx((*np_array).shape.size(), -1);
  std::vector<int> nshape((*np_array).shape.size(), 0);
  auto& shape = (*np_array).shape;
  for (int i = 0; i < trans.size(); i++) {
    nshape[i] = shape[trans[i]];
  }

  auto get_id = [&trans, &shape, &nshape]
    (std::vector<int> idx, bool do_trans = false) ->int {
    std::vector<int> s;
    if (do_trans) {
      std::vector<int> nidx(trans.size(), 0);
      for (int i = 0; i < trans.size(); i++) {
        nidx[i] = idx[trans[i]];
      }
      idx = nidx;
      s = nshape;
    } else {
      s = shape;
    }
    int tmp = 1;
    int id = 0;
    for (int i = idx.size()-1; i >= 0; --i) {
      id += idx[i] * tmp;
      tmp *= s[i];
    }
    return id;
  };

  int j = 0;
  int total_cnt = 0;
  while (j >= 0) {
    if (j == shape.size()) {
      int k1 = get_id(idx, true);
      int k2 = get_id(idx);
      tmp_data[k1] = (*np_array).data[k2];
      total_cnt++;
      j--;
    } else if (shape[j]-1 > idx[j]) {
      idx[j] += 1;
      j++;
    } else {
      idx[j] = -1;
      j--;
    }
  }
  // PLOG << total_cnt << " " << np_array.size;
  assert(total_cnt == (*np_array).size);

  for (int i = 0; i < (*np_array).size; ++i)
    (*np_array).data[i] = tmp_data[i];
  (*np_array).shape = nshape;

  delete [] tmp_data;
}

void pad_normal_distribution_data(float* data, int count, int seed,
    float mean, float var) {
  assert(data);
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(mean, var);
  for (int i = 0; i < count; i++) {
    data[i] = normal(gen);
  }
}

void pad_uniform_distribution_data(float* data, int count, int seed,
    float min, float max) {
  assert(data);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> uniform(min, max);
  for (int i = 0; i < count; i++) {
    data[i] = uniform(gen);
  }
}

int main(int argc, char** argv) {
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
      int num_rois = 112;
      int raw_count = 1*784*38*38;
      float* data_raw = static_cast<float*>(malloc(raw_count*sizeof(float)));
      pad_normal_distribution_data(data_raw, raw_count, 1000, 0, 1);
      for (int i = 0; i < raw_count; i++) {
        data_raw[i] = data_raw[i] * 3 + 5;
      }


      float* cx = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* cy = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* w = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* h = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* idx = static_cast<float*>(malloc(num_rois*sizeof(float)));
      memset(idx, 0, num_rois * sizeof(float)); // NOLINT
      pad_uniform_distribution_data(cx, num_rois, 1000, 0, 38);
      pad_uniform_distribution_data(cy, num_rois, 1000, 0, 38);
      pad_uniform_distribution_data(w, num_rois, 1000, 0, 10);
      pad_uniform_distribution_data(h, num_rois, 1000, 0, 10);

      float* x1 = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* y1 = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* x2 = static_cast<float*>(malloc(num_rois*sizeof(float)));
      float* y2 = static_cast<float*>(malloc(num_rois*sizeof(float)));
      for (int i = 0; i < num_rois; i++) {
        x1[i] = cx[i] - w[i] / 2;
        x1[i] = std::min(x1[i], static_cast<float>(38));
        y1[i] = cy[i] - h[i] / 2;
        y1[i] = std::min(y1[i], static_cast<float>(38));
        x2[i] = cx[i] + w[i] / 2;
        x2[i] = std::min(x2[i], static_cast<float>(38));
        y2[i] = cy[i] + h[i] / 2;
        y2[i] = std::min(y2[i], static_cast<float>(38));
      }

      free(cx);
      free(cy);
      free(w);
      free(h);

      int unit_num = 5;
      std::vector<int> data_shape = {1, 784, 38, 38};
      std::vector<int> rois_shape = {112, 1, 1, 5};
      NParray* np_data = new NParray(data_raw, data_shape);

      std::vector<int> data_shape_new = {1, 16, 49, 38, 38 };
      std::vector<int> data_trans = {0, 2, 1, 3, 4};
      std::vector<int> data_shape_end = {1, 784, 38, 38};
      np_reshape(np_data, data_shape_new);
      np_transpose(np_data, data_trans);
      np_reshape(np_data, data_shape_end);
      float* rois_conc_data = static_cast<float*>
        (malloc(num_rois * unit_num * sizeof(float)));
      for (int i = 0; i < num_rois; i++) {
        rois_conc_data[i * unit_num] = x1[i];
        rois_conc_data[i * unit_num + 1] = y1[i];
        rois_conc_data[i * unit_num + 2] = x2[i];
        rois_conc_data[i * unit_num + 3] = y2[i];
        rois_conc_data[i * unit_num + 4] = idx[i];
      }
      free(x1);
      free(x2);
      free(y1);
      free(y2);
      free(idx);
      NParray* np_rois = new NParray(rois_conc_data, rois_shape);

      for (int i = 0; i < inputNum; i++) {
        if (i == 0) {
          inputCpuPtrS[i] = static_cast<void*>(np_data->data);
        } else {
          inputCpuPtrS[i] = static_cast<void*>(np_rois->data);
        }
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
        int op = 1;
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
        LOG(INFO) << " SyncQueue Error ";
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
      free(data_raw);
      free(rois_conc_data);
      free(np_data);
      free(np_rois);
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
