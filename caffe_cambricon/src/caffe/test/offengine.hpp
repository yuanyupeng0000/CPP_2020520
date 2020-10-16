#ifdef USE_MLU
#ifndef TEST_OFFENGINE_HPP_
#define TEST_OFFENGINE_HPP_

#include <glog/logging.h>
#include <sstream>
#include <string>
#include <vector>
#include "cnrt.h"  // NOLINT
using std::vector;
using std::string;

namespace caffetool {
class OfflineEngine {
  public:
  explicit OfflineEngine(const string& name);
  virtual ~OfflineEngine();

  const string model_name() { return model_name_; }

  void set_model_name(const string& name) {
    CHECK(!name.empty());
    model_name_ = name;
  }

  void set_subnet_name(const string& name) {
    CHECK(!name.empty());
    subnet_name_ = name;
  }

  template <typename T>
  std::string ToString(T val) {
    std::stringstream stream;
    stream << val;
    return stream.str();
  }

  const int inum() { return inum_; }
  const int onum() { return onum_; }
  float** inputbufs() { return reinterpret_cast<float**>(icpuptrs); }

  const vector<int> in_shape(int index);
  const vector<int> out_shape(int index);
  void OpenDevice(const int& deviceId);
  void LoadModel();
  void CreateFunc();
  void AllocMemory();
  int Run();
  void CopyOut(const string& filename);
  void FreeResources();

  virtual void FillInput() = 0;

  private:
  void AllocateInputMemory();
  void AllocateOutputMemory();
  void AllocateMLUMemroy();

  const unsigned int BATCH_1 = 1;
  const cnrtDataType_t type = CNRT_FLOAT32;
  const cnrtDimOrder_t order = CNRT_NCHW;
  const cnrtFunctionType_t BLK = CNRT_FUNC_TYPE_BLOCK;
  const cnrtMemTransDir_t TO_DEV = CNRT_MEM_TRANS_DIR_HOST2DEV;
  const cnrtMemTransDir_t TO_HOST = CNRT_MEM_TRANS_DIR_DEV2HOST;

  string model_name_;
  string subnet_name_;

  cnrtModel_t model;
  cnrtFunction_t function;

  int inum_, onum_;
  vector<vector<int>*> in_shape_;
  vector<vector<int>*> out_shape_;

  vector<float*> output_cpu;
  vector<int> in_count;
  vector<int> out_count;
  void** icpuptrs;
  void** inputSyncPtrS;
  void** inputSyncTmpPtrS;
  void** ocpuptrs;
  void** outputSyncPtrS;
  void** param;
  void** imluptrs;
  void** omluptrs;
  int64_t* inputSizeArray;
  int64_t* outputSizeArray;
  cnrtDataType_t* inputDataTypeArray;
  cnrtDataType_t* outputDataTypeArray;
  vector<int*> inputDimValues;
  vector<int> inputDimNumS;
  vector<int*> outputDimValues;
  vector<int> outputDimNumS;
  int deviceId_;
};
}  // namespace caffetool
#endif  // TEST_OFFENGINE_HPP_
#endif  // USE_MLU
