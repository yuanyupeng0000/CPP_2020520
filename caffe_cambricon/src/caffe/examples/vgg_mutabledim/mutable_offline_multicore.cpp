#if defined(USE_MLU) && defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <condition_variable> // NOLINT
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread> // NOLINT
#include <utility>
#include <vector>

#include "cnrt.h" // NOLINT
#include "blocking_queue.hpp"
#include "common_functions.hpp"
#include "simple_interface.hpp"
#include "threadPool.h"

using std::map;
using std::max;
using std::min;
using std::vector;
using std::string;
using std::queue;
using std::stringstream;
using std::thread;
using std::pair;

DEFINE_string(meanfile, "", "mean file used to subtract from the input image.");
DEFINE_string(meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either meanfile or meanvalue should be provided, not both.");
DEFINE_int32(threads, 1, "threads, should "
                         "be lower than or equal to 32 ");
DEFINE_int32(channel_dup, 1, "Enable const memory auto channel duplication. "
                         "Could improve performance when multithreading."
                         "Works only with apiversion 3");
DEFINE_int32(simple_compile, 1, "Use simple compile interface or not.");
DEFINE_string(images, "", "input file list");
DEFINE_string(labels, "", "label to name");
DEFINE_string(labelmapfile, "",
    "prototxt with infomation about mapping from label to name");
DEFINE_int32(fix8, 0, "fp16(0) or fix8(1) mode. Default is fp16");
DEFINE_int32(int8, -1, "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
    "If specified, use int8 value, else, use fix8 value");
DEFINE_int32(yuv, 0, "bgr(0) or yuv(1) mode. Default is bgr");
DEFINE_double(scale, 1, "scale for input data, mobilenet...");
DEFINE_string(logdir, "", "path to dump log file, to terminal stderr by default");
DEFINE_int32(fifosize, 4, "set FIFO size of mlu input and output buffer, default is 2");
DEFINE_string(mludevice, "0",
    "set using mlu device number, set multidevice seperated by ','"
    "eg 0,1 when you use device number 0 and 1, default: 0");
DEFINE_string(functype, "1H16",
    "Specify the core to run on the arm device."
    "Set the options to 1H16 or 1H8, the default is 1H16.");
DEFINE_int32(Bangop, 0, "Use Bang Operator or not");
DEFINE_int32(preprocess_option, 0, "Use it to choose Image preprocess:"
    "0: image resize to input size,"
    "1: center input size crop from resized image with shorter size = 256,"
    "2: center input size crop from resized image into 256 x 256.");
DEFINE_string(output_dtype, "INVALID",
    "Specifies the type of output in the middle of the model.");
DEFINE_string(offlinemodel, "",
                  "The prototxt file used to find net configuration");

template<typename Dtype, template <typename> class Qtype>
class DataProvider;
template<typename Dtype, template <typename> class Qtype>
class PostProcessor;
template<typename Dtype, template <typename> class Qtype>
class Runner;
template <typename Dtype, template <typename> class Qtype>
class Pipeline;

template<typename Dtype, template <typename> class Qtype>
class DataProvider {
  public:
  explicit DataProvider(const string& meanfile,
                        const string& meanvalue,
                        const queue<string>& images):
                        allocatedSize_(0),
                        threadId_(0), deviceId_(0),
                        meanFile_(meanfile), meanValue_(meanvalue),
                        imageList(images), initMode(true) {}
  virtual ~DataProvider() {
    freeHostMemory();
    for (auto ptr : inPtrVector_) {
      for (int i = 0; i < this->runner_->inBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
    for (auto ptr : outPtrVector_) {
      for (int i = 0; i < this->runner_->outBlobNum(); i++) {
        cnrtFree(ptr[i]);
      }
      if (ptr != nullptr) free(ptr);
    }
    for (auto ptr : paramDescVector_) {
      CNRT_CHECK(cnrtDestroyParamDescArray(ptr,
                 this->runner_->inBlobNum() + this->runner_->outBlobNum()));
    }
  }
  void freeHostMemory() {
    setDeviceId(this->deviceId_);
    delete [] reinterpret_cast<float*>(cpuData_[0]);
    delete cpuData_;
    delete [] reinterpret_cast<char*>(syncCpuData_[0]);
    delete syncCpuData_;
  }
  void readOneBatch(int number);
  bool imageIsEmpty();
  void preRead();
  virtual void SetMeanFile() {}
  virtual void SetMeanValue();
  void SetMean();
  void WrapInputLayer(vector<vector<cv::Mat>>* wrappedImages, float* inputData);
  void Preprocess(const vector<cv::Mat>& srcImages, vector<vector<cv::Mat> >* dstImages);
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setRunner(Runner<Dtype, Qtype> *p) {
    runner_ = p;
    inNum_ = p->n();  // make preRead happy
  }
  inline void pushInPtrVector(Dtype* data) { inPtrVector_.push_back(data); }
  inline void pushOutPtrVector(Dtype* data) { outPtrVector_.push_back(data); }
  virtual void runParallel();
  virtual void runSerial() {}

  cv::Mat ResizeMethod(cv::Mat sample, int inputDim, int mode);

  protected:
  void allocateMemory(int queueLength, vector<int> shape);

  protected:
  int inNum_;
  int inChannel_;
  int inHeight_;
  int inWidth_;
  cv::Size inGeometry_;
  int allocatedSize_;

  int threadId_;
  int deviceId_;

  string meanFile_;
  string meanValue_;
  cv::Mat mean_;

  queue<string> imageList;
  vector<vector<cv::Mat>> inImages_;
  vector<vector<string>> imageName_;

  bool initMode;

  Runner<Dtype, Qtype> *runner_;

  private:
  Dtype* cpuData_;
  Dtype* syncCpuData_;
  vector<Dtype*> inPtrVector_;
  vector<Dtype*> outPtrVector_;
  vector<cnrtParamDescArray_t> paramDescVector_;
};

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::allocateMemory(int queueLength, vector<int> inShape) {
  Runner<Dtype, Qtype> *runner = static_cast<Runner<Dtype, Qtype>*>(this->runner_);
  int inBlobNum = runner->inBlobNum();
  int outBlobNum = runner->outBlobNum();

  if(!initMode) {
    freeHostMemory();
    for(int j = 0; j < FLAGS_fifosize; j++) {
      auto ptr = runner->popFreeInputData();
      cnrtFree(ptr);
    }
    for(int j = 0; j < FLAGS_fifosize; j++) {
      auto ptr = runner->popFreeOutputData();
      cnrtFree(ptr);
    }
  }
  for (int i = 0; i < queueLength; i++) {
    void** inputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * inBlobNum));
    void** outputMluPtrS =
      reinterpret_cast<void**>(malloc(sizeof(void*) * outBlobNum));

    // malloc input, it is ok to use runer's size
    for (int i = 0; i < inBlobNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(inputMluPtrS[i]), runner->inputSizeArray()[i]));
    }
    for (int i = 0; i < outBlobNum; i++) {
      CNRT_CHECK(cnrtMalloc(&(outputMluPtrS[i]), runner->outputSizeArray()[i]));
    }
    runner->pushFreeInputData(inputMluPtrS);
    runner->pushFreeOutputData(outputMluPtrS);
    pushInPtrVector(inputMluPtrS);
    pushOutPtrVector(outputMluPtrS);
  }
  cpuData_ = new(void*);
  cpuData_[0] = new float[this->inChannel_ *
                          this->inHeight_ * this->inWidth_ * this->inNum_];
  syncCpuData_ = new(void*);
  syncCpuData_[0] = new char[runner->inputSizeArray()[0]];
  initMode = false;
}

void setDeviceId(int deviceID) {
  unsigned devNum;
  CNRT_CHECK(cnrtGetDeviceCount(&devNum));
  if (deviceID >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(deviceID, devNum) << "Valid device count: " <<devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  LOG(INFO) << "Using MLU device " << deviceID;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, deviceID));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));
}
template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::readOneBatch(int number) {
  vector<cv::Mat> rawImages;
  vector<string> imageNameVec;
  string file_id , file;
  cv::Mat prev_image;
  int image_read = 0;

  while (image_read < number) {
    if (!this->imageList.empty()) {
      file = file_id = this->imageList.front();
      this->imageList.pop();
      if (file.find(" ") != string::npos)
        file = file.substr(0, file.find(" "));
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(file, inGeometry_);
      } else {
        img = cv::imread(file, -1);
      }
      if (img.data) {
        ++image_read;
        prev_image = img;
        imageNameVec.push_back(file_id);
        rawImages.push_back(img);
      } else {
        LOG(INFO) << "failed to read " << file;
      }
    } else {
        // if the que is empty and no file has been read, no more runs
        return;
      }
    }

  this->inImages_.push_back(rawImages);
  this->imageName_.push_back(imageNameVec);
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::WrapInputLayer(vector<vector<cv::Mat> >* wrappedImages,
                                  float* inputData) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = this->inWidth_;
  int height = this->inHeight_;
  int channels = FLAGS_yuv ? 1 : this->inChannel_;
  for (int i = 0; i < this->inNum_; ++i) {
    wrappedImages->push_back(vector<cv::Mat> ());
    for (int j = 0; j < channels; ++j) {
      if (FLAGS_yuv) {
        cv::Mat channel(height, width, CV_8UC1, reinterpret_cast<char*>(inputData));
        (*wrappedImages)[i].push_back(channel);
        inputData += width * height / 4;
      } else {
        cv::Mat channel(height, width, CV_32FC1, inputData);
        (*wrappedImages)[i].push_back(channel);
        inputData += width * height;
      }
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::Preprocess(const vector<cv::Mat>& sourceImages,
    vector<vector<cv::Mat> >* destImages) {
  /* Convert the input image to the input image format of the network. */
  CHECK(sourceImages.size() == destImages->size())
    << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    if (FLAGS_yuv) {
      cv::Mat sample_yuv;
      sourceImages[i].convertTo(sample_yuv, CV_8UC1);
      cv::split(sample_yuv, (*destImages)[i]);
      continue;
    }
    cv::Mat sample;
    if (sourceImages[i].channels() == 3 && inChannel_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2BGRA);
    else
      sample = sourceImages[i];
    cv::Mat sample_resized;
    if (sample.size() != inGeometry_) {
      sample_resized = ResizeMethod(sample, 256, 1);
    } else {
      sample_resized = sample;
    }

    cv::Mat sample_float;
    if (this->inChannel_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else if (this->inChannel_ == 4)
      sample_resized.convertTo(sample_float, CV_32FC4);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    bool int8 = (FLAGS_int8 != -1) ? FLAGS_int8 : FLAGS_fix8;
    if (!int8 && (!meanFile_.empty() || !meanValue_.empty())) {
      cv::subtract(sample_float, mean_, sample_normalized);
      if (FLAGS_scale != 1) {
        sample_normalized *= FLAGS_scale;
      }
    } else {
      sample_normalized = sample_float;
    }
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, (*destImages)[i]);
  }
}
template <typename Dtype, template <typename> class Qtype>
cv::Mat DataProvider<Dtype, Qtype>::ResizeMethod(cv::Mat sample, int inputDim,
    int mode) {
  int left_x, top_y, new_h, new_w;
  float img_w, img_h, img_scale;
  cv::Mat sample_temp;
  cv::Mat sample_temp_416;
  cv::Mat sample_temp_bgr;
  cv::Rect select;
  switch (mode) {
    case 0:  // resize source image into inputdim * inputdim
      cv::resize(sample, sample_temp, cv::Size(inputDim, inputDim));
      if (inGeometry_.width > inputDim || inGeometry_.height > inputDim) {
        LOG(INFO) <<"input size overrange inputdim X inputdim, you can try again"
                  << " by setting preprocess_option value to 0.";
        exit(1);
      }
      left_x = inputDim / 2 - inGeometry_.width / 2;
      top_y = inputDim / 2 - inGeometry_.height / 2;
      break;
    case 1:  // resize source image into inputdim * N, N is bigger than inputdim
      img_w = sample.cols;
      img_h = sample.rows;
      img_scale = img_w < img_h ? (inputDim / img_w) : (inputDim / img_h);
      new_w = std::round(img_w * img_scale);
      new_h = std::round(img_h * img_scale);
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h));
      if (inGeometry_.width > new_w || inGeometry_.height > new_h) {
        LOG(INFO) <<"input size overrange inputdim X N, you can try again"
                  << " by setting preprocess_option value to 0.";
        exit(1);
      }
      left_x = new_w / 2 - inGeometry_.width / 2;
      top_y = new_h / 2 - inGeometry_.height / 2;
      break;
    case 2:  // resize source image into inputdim * n, n is samller than inputdim
      img_w = sample.cols;
      img_h = sample.rows;
      img_scale = img_w < img_h ? (inputDim / img_h) : (inputDim / img_w);
      new_w = std::floor(img_w * img_scale);
      new_h = std::floor(img_h * img_scale);
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h), CV_INTER_LINEAR);
      if (inChannel_ == 3)
        sample_temp_416 = cv::Mat(inGeometry_.height, inGeometry_.height,
                                CV_8UC3, cv::Scalar(128, 128, 128));
      if (inChannel_ == 4)
        sample_temp_416 = cv::Mat(inGeometry_.height, inGeometry_.height,
                                CV_8UC4, cv::Scalar(128, 128, 128, 128));
      sample_temp.copyTo(sample_temp_416(
                        cv::Range((static_cast<float>(inGeometry_.height) - new_h) / 2,
                          (static_cast<float>(inGeometry_.height) - new_h) / 2 + new_h),
                        cv::Range((static_cast<float>(inGeometry_.height) - new_w) / 2,
                          (static_cast<float>(inGeometry_.height) - new_w) / 2 + new_w)));
      //  BGR(A)->RGB(A)
      if (inChannel_ == 3){
        cv::cvtColor(sample_temp_416, sample_temp_bgr, cv::COLOR_BGR2RGB);
        sample_temp_bgr.convertTo(sample_temp, CV_32FC3, 1);
      }
      if (inChannel_ == 4) {
        cv::cvtColor(sample_temp_416, sample_temp_bgr, cv::COLOR_BGRA2RGBA);
        sample_temp_bgr.convertTo(sample_temp, CV_32FC4, 1);
      }
      left_x = 0;
      top_y = 0;
      break;
    default:
      break;
  }
  select = cv::Rect(cv::Point(left_x, top_y), inGeometry_);
  return sample_temp(select);
}

#define INPUT0_NAME "data"
#define OUTPUT_NAME "pool5"
template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::runParallel() {
  Runner<Dtype, Qtype> *runner = static_cast<Runner<Dtype, Qtype>*>(this->runner_);

  setDeviceId(runner->deviceId());
  this->deviceId_ = runner->deviceId();

  Pipeline<Dtype, Qtype>::waitForNotification();
  int number = 1;
  while (this->imageList.size()) {
    Timer preprocessor;
    int batchsize = (number++)%16+1;
    this->inImages_.clear();
    this->imageName_.clear();
    this->readOneBatch(batchsize);
    if (this->inImages_.empty()) break;
    int in_n = inImages_[0].size();
    int in_c = 4;
    // int in_h = inImages_[0][0].rows;
    // int in_w = inImages_[0][0].cols;
    int in_h = 224;   // temporary: HW is not mutable
    int in_w = 224;
    this->inNum_ = in_n;
    this->inChannel_ = in_c;
    this->inHeight_ = in_h;
    this->inWidth_ = in_w;
    this->inGeometry_ = cv::Size(this->inWidth_, this->inHeight_);
    // this->SetMean();
    // create param desc and get it and push to queue
    vector<int> inShape = { in_n, in_c, in_h, in_w};
    auto descs = runner->getIODataInfo(inShape);
    paramDescVector_.push_back(descs);
    std::map<cnrtParamDescArray_t , vector<int>> in_desc;
    std::map<cnrtParamDescArray_t , vector<int>> out_desc;
    vector<int> outShape;
    vector<int> blobNum;
    blobNum.push_back(runner->inBlobNum());
    blobNum.push_back(runner->outBlobNum());
    in_desc[descs] = blobNum;
    outShape.push_back(runner->outNum());
    outShape.push_back(runner->outChannel());
    outShape.push_back(runner->outHeight());
    outShape.push_back(runner->outWidth());
    outShape.push_back(runner->outCounts()[0]);
    out_desc[descs] = outShape;
    runner->pushValidInputParamDesc(in_desc);
    runner->pushValidOutputParamDesc(out_desc);

    // re-allocate memory if needed
    if(allocatedSize_ < in_n * in_c * in_h * in_w) {
      allocateMemory(FLAGS_fifosize, inShape);
      allocatedSize_ = in_n * in_c * in_h * in_w;
    }

    vector<cv::Mat>& rawImages = this->inImages_[0];
    vector<string>& imageNameVec = this->imageName_[0];

    vector<vector<cv::Mat> > preprocessedImages;
    Timer prepareInput;
    this->WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData_[0]));
    this->Preprocess(rawImages, &preprocessedImages);
    prepareInput.log("prepare input data ...");

    void** mluData = runner->popFreeInputData();
    Timer copyin;
    cnrtDataType_t mluDtype = runner->mluInputDtype()[0];
    cnrtDataType_t cpuDtype;
    if (FLAGS_yuv) {
      cpuDtype = CNRT_UINT8;
    } else {
      cpuDtype = CNRT_FLOAT32;
    }
    // since runner's shape is changed by provider, so the shape info latest
    int dim_values[4] = {in_n, in_c, in_h, in_w};
    int dim_order[4] = {0, 2, 3, 1};  // NCHW --> NHWC
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(cpuData_[0]), cpuDtype,
            reinterpret_cast<void*>(syncCpuData_[0]), mluDtype,
            nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(reinterpret_cast<void*>(cpuData_[0]), mluDtype,
            reinterpret_cast<void*>(syncCpuData_[0]), 4, dim_values, dim_order));
    }
    LOG(INFO) << "memcpy size: " << runner->inputSizeArray()[0];
    CNRT_CHECK(cnrtMemcpy(reinterpret_cast<void*>(mluData[0]),
                          reinterpret_cast<void*>(syncCpuData_[0]),
                          runner->inputSizeArray()[0],
                          CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin.log("copyin time ...");
    preprocessor.log("preprocessor time ...");

    runner->pushValidInputDataAndNames(mluData, imageNameVec);
  }

  LOG(INFO) << "DataProvider: no data ...";
}

template<typename Dtype, template <typename> class Qtype>
class PostProcessor {
  public:
  PostProcessor() : threadId_(0), deviceId_(0),
                    initSerialMode(false) {}
  virtual ~PostProcessor() {}
  virtual void runParallel();
  virtual void runSerial() {}
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setRunner(Runner<Dtype, Qtype> *p) { runner_ = p; }
  inline int top1() { return top1_; }
  inline int top5() { return top5_; }
  void readLabels(vector<string>* labels);
  void updateResult(const vector<string>& origin_img,
                    const vector<string>& labels,
                    float* outCputPtr);
  void printClassResult();

  protected:
  int threadId_;
  int deviceId_;
  int total_ = 0;
  int outCount_ = 0;
  int outN_ = 0;
  vector<string> labels;
  int top1_ = 0;
  int top5_ = 0;

  bool initSerialMode;

  Runner<Dtype, Qtype> *runner_;

  private:
  Dtype* outCpuPtrs_;
  void* syncCpuPtrs_;
};

template<typename Dtype, template <typename> class Qtype>
void PostProcessor<Dtype, Qtype>::runParallel() {
  Runner<Dtype, Qtype> *infr = static_cast<Runner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(infr->deviceId());
  this->readLabels(&this->labels);
  cnrtParamDescArray_t out_descs = nullptr;
  vector<int> out_shape;
  int dim_order[4] = {0, 3, 1, 2};
  while (true) {
    map<cnrtParamDescArray_t, vector<int>> out_desc = infr->popValidOutputParamDesc();
    out_descs = out_desc.begin()->first;
    out_shape = out_desc.begin()->second;
    int outNumTemp = out_shape[0];
    int outChannelTemp = out_shape[1];
    int outHeightTemp = out_shape[2];
    int outWidthTemp = out_shape[3];
    int outCountTemp = out_shape[4];
    this->outCount_ = outCountTemp;
    this->outN_ = outNumTemp;

    Dtype* mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) break;  // no more work

    cnrtDataType_t cpuDtype = CNRT_FLOAT32;
    cnrtDataType_t input_mlu_dtype = CNRT_INVALID;
    cnrtDataType_t mluDtype = CNRT_INVALID;
    cnrtGetDataTypeFromParamDesc(out_descs[0], &input_mlu_dtype);
    cnrtGetDataTypeFromParamDesc(out_descs[1], &mluDtype);

    int dim_values[4] = {outNumTemp, outHeightTemp,
                         outWidthTemp, outChannelTemp};
    outCpuPtrs_ = new(Dtype);
    outCpuPtrs_[0] = new float[outCountTemp];
    int64_t outputSize;
    CNRT_CHECK(cnrtGetParamDescSize(out_descs[1], &outputSize));

    syncCpuPtrs_ = malloc(outputSize);
    LOG(INFO) << "post::outsize=" << outputSize;


    Timer copyout;
    CNRT_CHECK(cnrtMemcpy(syncCpuPtrs_, mluOutData[0],
                          outputSize,
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
    if (mluDtype != cpuDtype) {
      CNRT_CHECK(cnrtTransOrderAndCast(syncCpuPtrs_, mluDtype,
            outCpuPtrs_[0], cpuDtype,
            nullptr, 4, dim_values, dim_order));
    } else {
      CNRT_CHECK(cnrtTransDataOrder(syncCpuPtrs_, cpuDtype,
            outCpuPtrs_[0], 4, dim_values, dim_order));
    }
    copyout.log("copyout time ...");

    Timer postProcess;
    infr->pushFreeOutputData(mluOutData);
    float* data = reinterpret_cast<float*>(outCpuPtrs_[0]);

    vector<string> origin_img = infr->popValidInputNames();
    this->updateResult(origin_img, this->labels, data);

    delete [] reinterpret_cast<float*>(outCpuPtrs_[0]);
    outCpuPtrs_[0] = nullptr;
    delete reinterpret_cast<Dtype*>(outCpuPtrs_);
    outCpuPtrs_ = nullptr;
    free(syncCpuPtrs_);
    syncCpuPtrs_ = nullptr;
    postProcess.log("post process time ...");
  }
  this->printClassResult();
}

template <typename Dtype, template <typename> class Qtype>
void PostProcessor<Dtype, Qtype>::readLabels(vector<string>* labels) {
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

template <typename Dtype, template <typename> class Qtype>
void PostProcessor<Dtype, Qtype>::updateResult(const vector<string>& origin_img,
                                    const vector<string>& labels,
                                    float* outCpuPtr) {
  for (int i = 0; i < this->outN_; i++) {
    string image = origin_img[i];
    if (image == "null") break;

    this->total_++;
    if (image.find_last_of(" ") != -1) {
      image = image.substr(0, image.find(" "));
    }
    vector<int> vtrTop5 = getTop5(labels,
                                  image,
                                  outCpuPtr + i * this->outCount_ / this->outN_,
                                  this->outCount_ / this->outN_);
    image = origin_img[i];
    if (image.find(" ") != string::npos) {
      image = image.substr(image.find(" "));
    }

    int labelID = atoi(image.c_str());
    for (int i = 0; i < 5; i++) {
      if (vtrTop5[i] == labelID) {
        this->top5_++;
        if (i == 0)
          this->top1_++;
        break;
      }
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
void PostProcessor<Dtype, Qtype>::printClassResult() {
  LOG(INFO) << "Accuracy thread id : " << this->runner_->threadId();
  LOG(INFO) << "accuracy1: " << 1.0 * this->top1_ / this->total_ << " ("
            << this->top1_ << "/" << this->total_ << ")";
  LOG(INFO) << "accuracy5: " << 1.0 * this->top5_ / this->total_ << " ("
            << this->top5_ << "/" << this->total_ << ")";
}

template <typename Dtype, template <typename> class Qtype>
class RunnerStrategy {
  public:
  virtual void runParallel(Runner<Dtype, Qtype>* runner);
  virtual ~RunnerStrategy() {}
};

template <typename Dtype, template <typename> class Qtype>
class Runner {
  public:
  Runner():initSerialMode(false), simple_flag_(false) {}
  Runner(const cnrtRuntimeContext_t rt_ctx,
            const int& id);
  Runner(const string& offlinemodel,
            const int& id,
            const int& parallel,
            const int& deviceId,
            const int& devicesize);
  virtual ~Runner();

  int n() {return inNum_;}
  int c() {return inChannel_;}
  int h() {return inHeight_;}
  int w() {return inWidth_;}
  void setInDim(const vector<int> inshape) {
    inNum_ = inshape[0];
    inChannel_ = inshape[1];
    inHeight_ = inshape[2];
    inWidth_ = inshape[3];
  }
  void set_n(int n) {inNum_ = n; LOG(INFO) << "New N is set:" << inNum_;}
  void pushValidInputData(Dtype* data) { validInputFifo_.push(data); }
  void pushFreeInputData(Dtype* data) { freeInputFifo_.push(data); }
  Dtype* popValidInputData() { return validInputFifo_.pop(); }
  Dtype* popFreeInputData() { return freeInputFifo_.pop(); }
  void pushValidOutputData(Dtype* data) { validOutputFifo_.push(data);}
  void pushFreeOutputData(Dtype* data) { freeOutputFifo_.push(data);}
  Dtype* popValidOutputData() { return validOutputFifo_.pop(); }
  Dtype* popFreeOutputData() { return freeOutputFifo_.pop(); }
  void pushValidInputNames(vector<string> images) { imagesFifo_.push(images); }
  vector<string> popValidInputNames() { return imagesFifo_.pop(); }
  void pushValidInputDataAndNames(Dtype* data, const vector<string>& names) {
    std::lock_guard<std::mutex> lk(runner_mutex_);
    pushValidInputData(data);
    pushValidInputNames(names);
  }
  void pushValidOutputSyncData(Dtype* data) { validOutputSyncFifo_.push(data);}
  void pushFreeOutputSyncData(Dtype* data) { freeOutputSyncFifo_.push(data);}
  Dtype* popValidOutputSyncData() { return validOutputSyncFifo_.pop(); }
  Dtype* popFreeOutputSyncData() { return freeOutputSyncFifo_.pop(); }
  void pushValidInputSyncData(Dtype* data) { validInputSyncFifo_.push(data); }
  void pushFreeInputSyncData(Dtype* data) { freeInputSyncFifo_.push(data); }
  Dtype* popValidInputSyncData() { return validInputSyncFifo_.pop(); }
  Dtype* popFreeInputSyncData() { return freeInputSyncFifo_.pop(); }
  void pushValidInputSyncTmpData(Dtype* data) { validInputSyncTmpFifo_.push(data); }
  void pushFreeInputSyncTmpData(Dtype* data) { freeInputSyncTmpFifo_.push(data); }
  Dtype* popValidInputSyncTmpData() { return validInputSyncTmpFifo_.pop(); }
  Dtype* popFreeInputSyncTmpData() { return freeInputSyncTmpFifo_.pop(); }

  void pushValidInputParamDesc(map<cnrtParamDescArray_t, vector<int>> desc) { InputParamDescFifo_.push(desc); }
  map<cnrtParamDescArray_t, vector<int>> popValidInputParamDesc() { return InputParamDescFifo_.pop(); }
  void pushValidOutputParamDesc(map<cnrtParamDescArray_t, vector<int>> desc) { OutputParamDescFifo_.push(desc); }
  map<cnrtParamDescArray_t, vector<int>> popValidOutputParamDesc() { return OutputParamDescFifo_.pop(); }

  virtual void runParallel();
  virtual void runSerial() {}
  cnrtFunction_t function() {return function_;}
  cnrtQueue_t queue() {return queue_;}
  void setQueue(const cnrtQueue_t& queue) {queue_ = queue;}

  cnrtRuntimeContext_t runtimeContext() {return rt_ctx_;}
  inline int inBlobNum() { return inBlobNum_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int outNum() { return outNum_; }
  inline int outChannel() { return outChannel_; }
  inline int outHeight() { return outHeight_; }
  inline int outWidth() { return outWidth_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float runTime() { return runTime_; }
  inline void setRunTime(const float& time) {runTime_ = time;}
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor<Dtype, Qtype> *p ) { postProcessor_ = p; }
  inline bool simpleFlag() {return simple_flag_;}

  inline int64_t* inputSizeArray() { return inputSizeArray_; }
  inline int64_t* outputSizeArray() { return outputSizeArray_; }
  inline cnrtDataType_t* mluInputDtype() { return mluInputDtype_; }
  inline cnrtDataType_t* mluOutputDtype() { return mluOutputDtype_; }
  inline vector<int> inCounts() { return inCounts_; }
  inline vector<int> outCounts() { return outCounts_; }
  inline vector<int> inDimNums() { return inDimNums_; }
  inline vector<int> outDimNums() { return outDimNums_; }
  inline vector<int*> inDimValues() { return inDimValues_; }
  inline vector<int*> outDimValues() { return outDimValues_; }

  int size_validInputFifo_(){ return validInputFifo_.size(); }
  int size_validOutputFifo_(){ return validOutputFifo_.size(); }

  private:
  Qtype<Dtype*> validInputFifo_;
  Qtype<Dtype*> freeInputFifo_;
  Qtype<Dtype*> validInputSyncFifo_;
  Qtype<Dtype*> freeInputSyncFifo_;
  Qtype<Dtype*> validInputSyncTmpFifo_;
  Qtype<Dtype*> freeInputSyncTmpFifo_;
  Qtype<Dtype*> validOutputFifo_;
  Qtype<Dtype*> freeOutputFifo_;
  Qtype<Dtype*> validOutputSyncFifo_;
  Qtype<Dtype*> freeOutputSyncFifo_;
  Qtype<vector<string> > imagesFifo_;
  Qtype<map<cnrtParamDescArray_t, vector<int>>> InputParamDescFifo_;
  Qtype<map<cnrtParamDescArray_t, vector<int>>> OutputParamDescFifo_;

  std::mutex runner_mutex_;

  protected:
  int inBlobNum_ = 1;
  int outBlobNum_ = 1;
  unsigned int inNum_, inChannel_, inHeight_, inWidth_;
  unsigned int outNum_, outChannel_, outHeight_, outWidth_;
  int threadId_ = 0;
  int deviceId_;
  int deviceSize_ = 1;
  int Parallel_ = 1;
  float runTime_;
  bool initSerialMode;
  bool simple_flag_;

  int64_t* inputSizeArray_;
  int64_t* outputSizeArray_;
  cnrtDataType_t* mluInputDtype_;
  cnrtDataType_t* mluOutputDtype_;
  vector<int> inCounts_, outCounts_;
  vector<int> inDimNums_, outDimNums_;
  vector<int*> inDimValues_, outDimValues_;

  PostProcessor<Dtype, Qtype> *postProcessor_;

  public:
  cnrtParamDescArray_t getIODataInfo(vector<int> inShape);

  private:
  void loadOfflinemodel(const string& offlinemodel);

  private:
  RunnerStrategy<Dtype, Qtype>* runnerStrategy_;
  cnrtModel_t model_;
  cnrtQueue_t queue_;
  cnrtFunction_t func;
  cnrtFunction_t function_;
  cnrtRuntimeContext_t rt_ctx_;
  cnrtDim3_t dim_;
};

template<typename Dtype, template <typename> class Qtype>
Runner<Dtype, Qtype>::Runner(const cnrtRuntimeContext_t rt_ctx,
                                   const int& id) {
  this->rt_ctx_ = rt_ctx;
  this->threadId_ = id;
  this->runTime_ = 0;
  this->simple_flag_ = true;

  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_FUNCTION,
         reinterpret_cast<void **>(&this->function_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_MODEL_PARALLEL,
         reinterpret_cast<void **>(&this->Parallel_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_DEV_ORDINAL,
         reinterpret_cast<void **>(&this->deviceId_));

  runnerStrategy_ = new RunnerStrategy<Dtype, Qtype>();

  // allocate memory
  this->inCounts_.resize(this->inBlobNum_, 1);
  this->outCounts_.resize(this->outBlobNum_, 1);
  this->inDimNums_.resize(this->inBlobNum_, 0);
  this->outDimNums_.resize(this->outBlobNum_, 0);
  this->inDimValues_.resize(this->inBlobNum_, nullptr);
  this->outDimValues_.resize(this->outBlobNum_, nullptr);
  inputSizeArray_ = (int64_t*)malloc(sizeof(int64_t)*inBlobNum_);
  outputSizeArray_ = (int64_t*)malloc(sizeof(int64_t)*outBlobNum_);
  mluInputDtype_ = (cnrtDataType_t*)malloc(sizeof(cnrtDataType_t)*inBlobNum_);
  mluOutputDtype_ = (cnrtDataType_t*)malloc(sizeof(cnrtDataType_t)*outBlobNum_);
  for(int i = 0; i < inBlobNum_; ++i) {
    inDimValues_.push_back((int*)malloc(4*sizeof(int)));
  }
  for(int i = 0; i < outBlobNum_; ++i) {
    outDimValues_.push_back((int*)malloc(4*sizeof(int)));
  }
}

template<typename Dtype, template <typename> class Qtype>
Runner<Dtype, Qtype>::~Runner() {
  setDeviceId(this->deviceId_);

  for (auto values: outDimValues_)
     free(values);
  for (auto values: inDimValues_)
     free(values);

  delete runnerStrategy_;

  free(inputSizeArray_);
  inputSizeArray_ = nullptr;
  free(outputSizeArray_);
  outputSizeArray_ = nullptr;
  free(mluInputDtype_);
  mluInputDtype_ = nullptr;
  free(mluOutputDtype_);
  mluOutputDtype_ = nullptr;
}

// get function's I/O DataDesc
template<typename Dtype, template <typename> class Qtype>
cnrtParamDescArray_t Runner<Dtype, Qtype>::getIODataInfo(vector<int> inShape) {
  cnrtParamDescArray_t IOParamDescs;
  cnrtCreateParamDescArray(&IOParamDescs, inBlobNum_ + outBlobNum_);
  const int dim_num = 4;
  int in_n = inShape[0];
  int in_c = inShape[1];
  int in_h = inShape[2];
  int in_w = inShape[3];

  int input_shape[dim_num] = {in_n, in_h, in_w, in_c};
  CNRT_CHECK(cnrtSetShapeToParamDesc(IOParamDescs[0], input_shape, 4));

  CNRT_CHECK(cnrtInferFunctionOutputShape(function_, 1, &IOParamDescs[0],
                                          1, &IOParamDescs[1]));
  int n;
  CNRT_CHECK(cnrtGetShapeFromParamDesc(IOParamDescs[0], inDimValues_.data(), &n));
  inDimNums_[0] = n;
  LOG(INFO) <<"cnrtInputShapeFromParam " << n <<" "
            << (inDimValues_[0])[0] <<" "<<(inDimValues_[0])[1] <<" "
            << (inDimValues_[0])[2] <<" "<< (inDimValues_[0])[3];
  CNRT_CHECK(cnrtGetShapeFromParamDesc(IOParamDescs[1], outDimValues_.data(), &n));
  outDimNums_[0] = n;
  LOG(INFO) <<"cnrtOutputShapeFromParam " << n <<" "
            << (outDimValues_[0])[0] <<" "<<(outDimValues_[0])[1] <<" "
            << (outDimValues_[0])[2] <<" "<< (outDimValues_[0])[3];
  size_t input_num;
  size_t output_num;
  CNRT_CHECK(cnrtGetParamElementNum(IOParamDescs[0], &input_num));
  CNRT_CHECK(cnrtGetParamElementNum(IOParamDescs[1], &output_num));
  LOG(INFO) << "IO elem num: " << input_num << " , " << output_num;
  inCounts_[0] = input_num;
  outCounts_[0] = output_num;
  int64_t input_size;
  int64_t output_size;
  CNRT_CHECK(cnrtGetParamDescSize(IOParamDescs[0], &input_size));
  CNRT_CHECK(cnrtGetParamDescSize(IOParamDescs[1], &output_size));
  LOG(INFO) << "IO size:" << input_size << " , " << output_size;
  inputSizeArray_[0] = input_size;
  outputSizeArray_[0] = output_size;

  CNRT_CHECK(cnrtGetDataTypeFromParamDesc(IOParamDescs[0], this->mluInputDtype_)); // TODO??
  LOG(INFO) << "InDT: " << *mluInputDtype_;
  CNRT_CHECK(cnrtGetDataTypeFromParamDesc(IOParamDescs[1], this->mluOutputDtype_)); // TODO??
  LOG(INFO) << "OutDT: " << *mluOutputDtype_;

  this->inNum_ = this->inDimValues_[0][0];
  this->inChannel_ = this->inDimValues_[0][3];
  this->inWidth_ = this->inDimValues_[0][1];
  this->inHeight_ = this->inDimValues_[0][2];

  this->outNum_ = this->outDimValues_[0][0];
  this->outChannel_ = this->outDimValues_[0][3];
  this->outWidth_ = this->outDimValues_[0][1];
  this->outHeight_ = this->outDimValues_[0][2];

  return IOParamDescs;
}

template<typename Dtype, template <typename> class Qtype>
void Runner<Dtype, Qtype>::runParallel() {
  runnerStrategy_->runParallel(this);
}

template<typename Dtype, template <typename> class Qtype>
void RunnerStrategy<Dtype, Qtype>::runParallel(Runner<Dtype, Qtype>* runner) {
#define RES_SIZE 1
  // set device to runtime context binded device
  cnrtSetCurrentContextDevice(runner->runtimeContext());
  cnrtQueue_t queue[RES_SIZE];
  cnrtNotifier_t notifierBeginning[RES_SIZE];
  cnrtNotifier_t notifierEnd[RES_SIZE];

  for (int i = 0; i < RES_SIZE; i++) {
    CHECK(cnrtCreateQueue(&queue[i]) == CNRT_RET_SUCCESS)
          << "CNRT create queue error, thread_id " << runner->threadId();
    cnrtCreateNotifier(&notifierBeginning[i]);
    cnrtCreateNotifier(&notifierEnd[i]);
  }
  float eventInterval[RES_SIZE] = {0};
  Dtype* mluInData[RES_SIZE];
  Dtype* mluOutData[RES_SIZE];
  cnrtParamDescArray_t descs = nullptr;

  auto do_pop = [&](int index, void **param) {
    mluInData[index] = runner->popValidInputData();
    if ( mluInData[index] == nullptr )
      return false;
    mluOutData[index] = runner->popFreeOutputData();
    for (int i = 0; i < runner->inBlobNum(); i++) {
      param[i] = mluInData[index][i];
    }
    for (int i = 0; i < runner->outBlobNum(); i++) {
      param[runner->inBlobNum() + i] = mluOutData[index][i];
    }

    return true;
  };

  auto do_invoke = [&](int index, void** param) {
    cnrtPlaceNotifier(notifierBeginning[index], queue[index]);
    CNRT_CHECK(cnrtInvokeRuntimeContext_V2(runner->runtimeContext(),
               descs, param, queue[index], nullptr));
  };

  auto do_sync = [&](int index) {
    cnrtPlaceNotifier(notifierEnd[index], queue[index]);
    if (cnrtSyncQueue(queue[index]) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning[index],
             notifierEnd[index], &eventInterval[index]);
      runner->setRunTime(runner->runTime() + eventInterval[index]);
      printfMluTime(eventInterval[index]);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }
    runner->pushValidOutputData(mluOutData[index]);
    // cnrtFree(mluInData[index]);
    runner->pushFreeInputData(mluInData[index]);
  };

  while (true) {
    auto indata = runner->popValidInputParamDesc();
    descs = indata.begin()->first;

    void* param[runner->inBlobNum() + runner->outBlobNum()];
    // pop - ping
    if (do_pop(0, static_cast<void **>(param)) == false) {
      break;
    }
    do_invoke(0, static_cast<void **>(param));
    do_sync(0);
  }

  for (int i = 0; i < RES_SIZE; i++) {
    cnrtDestroyNotifier(&notifierBeginning[i]);
    cnrtDestroyNotifier(&notifierEnd[i]);
    cnrtDestroyQueue(queue[i]);
  }

  // tell postprocessor to exit
  runner->pushValidOutputData(static_cast<Dtype*>(nullptr));
}

template <typename Dtype, template <typename> class Qtype>
class Pipeline {
  public:
  Pipeline(DataProvider<Dtype, Qtype> *provider,
           Runner<Dtype, Qtype> *runner,
           PostProcessor<Dtype, Qtype> *postprocessor);
  Pipeline(const vector<DataProvider<Dtype, Qtype>*>& providers,
           Runner<Dtype, Qtype> *runner,
           PostProcessor<Dtype, Qtype> *postprocessor);
  ~Pipeline();
  void runParallel();
  void runSerial();
  inline DataProvider<Dtype, Qtype>* dataProvider() { return data_provider_; }
  inline vector<DataProvider<Dtype, Qtype>*> dataProviders() { return data_providers_; }
  inline Runner<Dtype, Qtype>* runner() { return runner_; }
  inline PostProcessor<Dtype, Qtype>* postProcessor() { return postProcessor_; }
  static void notifyAll();
  static void waitForNotification();

  private:
  DataProvider<Dtype, Qtype> *data_provider_;
  vector<DataProvider<Dtype, Qtype>*> data_providers_;
  Runner<Dtype, Qtype> *runner_;
  PostProcessor<Dtype, Qtype>* postProcessor_;

  static int imageNum;
  static vector<queue<string>> imageList;

  static std::condition_variable condition;
  static std::mutex condition_m;
  static int start;
  static vector<thread*> stageThreads;
  static vector<Pipeline*> pipelines;
};

template <typename Dtype, template <typename> class Qtype>
int Pipeline<Dtype, Qtype>::imageNum = 0;
template <typename Dtype, template <typename> class Qtype>
vector<queue<string>> Pipeline<Dtype, Qtype>::imageList;
template <typename Dtype, template <typename> class Qtype>
std::condition_variable Pipeline<Dtype, Qtype>::condition;
template <typename Dtype, template <typename> class Qtype>
std::mutex Pipeline<Dtype, Qtype>::condition_m;
template <typename Dtype, template <typename> class Qtype>
int Pipeline<Dtype, Qtype>::start = 0;
template <typename Dtype, template <typename> class Qtype>
vector<thread*> Pipeline<Dtype, Qtype>::stageThreads;
template <typename Dtype, template <typename> class Qtype>
vector<Pipeline<Dtype, Qtype>*> Pipeline<Dtype, Qtype>::pipelines;

template <typename Dtype, template <typename> class Qtype>
Pipeline<Dtype, Qtype>::Pipeline(const vector<DataProvider<Dtype, Qtype>*>& providers,
                                 Runner<Dtype, Qtype> *runner,
                                 PostProcessor<Dtype, Qtype> *postprocessor)
                                 : data_provider_(nullptr),
                                   runner_(nullptr),
                                   postProcessor_(nullptr) {
  CHECK(providers.size() > 0) << "[Error]the size of providers should greater than 0.";
  runner_ = runner;
  postProcessor_ = postprocessor;

  postProcessor_->setRunner(runner_);
  runner_->setPostProcessor(postProcessor_);
  postProcessor_->setThreadId(runner_->threadId());

  data_providers_ = providers;
  for (auto data_provider : data_providers_) {
    data_provider->setRunner(runner_);
    data_provider->setThreadId(runner_->threadId());
  }
}

template <typename Dtype, template <typename> class Qtype>
Pipeline<Dtype, Qtype>::~Pipeline() {
  // delete data_providers_ only for simple compile
  for (auto data_provider : data_providers_) {
    delete data_provider;
  }
  // delete data_provider_ only for flexible compile
  if (data_provider_) {
    delete data_provider_;
  }
  if (runner_) {
    delete runner_;
  }
  if (postProcessor_) {
    delete postProcessor_;
  }
}
template <typename Dtype, template <typename> class Qtype>
void Pipeline<Dtype, Qtype>::waitForNotification() {
  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [](){return start;});
  lk.unlock();
}

template <typename Dtype, template <typename> class Qtype>
void Pipeline<Dtype, Qtype>::runParallel() {
  int data_provider_num = (data_providers_.size() == 0) ? 1 : data_providers_.size();
  vector<thread*> threads(data_provider_num + 2, nullptr);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i] = new thread(&DataProvider<Dtype, Qtype>::runParallel,
        data_providers_[i]);
  }

  threads[data_provider_num] = new thread(&Runner<Dtype, Qtype>::runParallel, runner_);
  threads[data_provider_num + 1] = new thread(&PostProcessor<Dtype, Qtype>::runParallel,
                                              postProcessor_);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i]->join();
    delete threads[i];
  }
  // push a nullptr for simple compile when the thread of data provider finished tasks
  runner_->pushValidInputData(nullptr);
  runner_->pushValidInputSyncData(nullptr);
  runner_->pushValidInputSyncTmpData(nullptr);
  cnrtParamDescArray_t descs = nullptr;
  std::map<cnrtParamDescArray_t , vector<int>> in_desc;
  std::map<cnrtParamDescArray_t , vector<int>> out_desc;
  in_desc[descs] = vector<int>(2,0);
  out_desc[descs] = vector<int>(5,0);
  runner_->pushValidInputParamDesc(in_desc);
  runner_->pushValidOutputParamDesc(out_desc);

  for (int i = 0; i < 2; i++) {
    threads[data_provider_num + i]->join();
    delete threads[data_provider_num + i];
  }
}
typedef DataProvider<void*, BlockingQueue> DataProviderT;
typedef Runner<void*, BlockingQueue> RunnerT;
typedef PostProcessor<void*, BlockingQueue> PostProcessorT;
typedef Pipeline<void*, BlockingQueue> PipelineT;

int main(int argc, char* argv[]) {
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
  gflags::SetUsageMessage("Do offline multicore classification.\n"
        "Usage:\n"
        "    clas_offline_multicore [FLAGS] modelfile listfile\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/clas_offline_multicore/");
    return 1;
  }

  SimpleInterface& simpleInterface = SimpleInterface::getInstance();
  int provider_num = 1;
  simpleInterface.setFlag(true);
  // provider_num = SimpleInterface::data_provider_num_;

  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }

  int totalThreads = FLAGS_threads * deviceIds_.size();
  cnrtInit(0);
  simpleInterface.loadOfflinemodel(FLAGS_offlinemodel, deviceIds_, FLAGS_channel_dup, FLAGS_threads);

  ImageReader img_reader(FLAGS_images, totalThreads * provider_num);
  auto&& imageList = img_reader.getImageList();
  int imageNum = img_reader.getImageNum();

  vector<thread*> stageThreads;
  vector<PipelineT* > pipelines;
  vector<DataProviderT*> providers;

  for (int i = 0; i < totalThreads; i++) {
    DataProviderT* provider;
    RunnerT* runner;
    PipelineT* pipeline;

    providers.clear();
    // provider_num is 1 for flexible compile.
    for (int j = 0; j < provider_num; j++) {
      provider = new DataProviderT(FLAGS_meanfile, FLAGS_meanvalue,
                                      imageList[provider_num * i + j]);
      providers.push_back(provider);
    }

    auto postprocessor = new PostProcessorT();

    auto dev_runtime_contexts = simpleInterface.get_runtime_contexts();
    int index = i % deviceIds_.size();
    int thread_id = i / deviceIds_.size();
    runner = new RunnerT(dev_runtime_contexts[index][thread_id], i);
    pipeline = new PipelineT(providers, runner, postprocessor);

    stageThreads.push_back(new thread(&PipelineT::runParallel, pipeline));
    pipelines.push_back(pipeline);
  }

  for (int i = 0; i < stageThreads.size(); i++) {
    pipelines[i]->notifyAll();
  }

  Timer timer;
  for (int i = 0; i < stageThreads.size(); i++) {
    stageThreads[i]->join();
    delete stageThreads[i];
  }
  timer.log("Total execution time");
  float execTime = timer.getDuration();
  float mluTime = 0;
  int acc1 = 0;
  int acc5 = 0;
  for (int i = 0; i < pipelines.size(); i++) {
    acc1 += pipelines[i]->postProcessor()->top1();
    acc5 += pipelines[i]->postProcessor()->top5();
    mluTime += pipelines[i]->runner()->runTime();
  }
  printfAccuracy(imageNum, acc1, acc5);
  printPerf(imageNum, execTime, mluTime, totalThreads, 1);
  saveResult(imageNum, acc1, acc5, (-1), mluTime, execTime, totalThreads, 1);

  for (auto iter : pipelines)
    delete iter;
  simpleInterface.destroyRuntimeContext();
  cnrtDestroy();
}
#else
#include <glog/logging.h>
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             <<" of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // defined(USE_MLU) && defined(USE_OPENCV)