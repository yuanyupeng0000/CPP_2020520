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


#ifdef USE_MLU
#include<caffe/caffe.hpp>
#include<caffe/compile.hpp>
#include<cstdlib>
#include<fstream>
#include<string>
#include<thread>
#include<vector>

DEFINE_string(model, "model", "model file with prototxt type");
DEFINE_string(weights, "", "weights file with .caffemodel");
DEFINE_int32(mode, 0, "select compile mode, 0: save offline model file"
    " 1: save offline model mem 2: verify offline model file with"
    " CPU info");
DEFINE_string(mcore, "1H8", "select core version");
DEFINE_string(offlinemodel, "NULL", "specified verified offline file");
DEFINE_string(mname, "offline", "The name for the offline model to be generated.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
  gflags::SetUsageMessage("Test for compile usage");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "Task_Compile");
    return -1;
  }
  int32_t modeNum = FLAGS_mode;
  std::string prototxtFile = FLAGS_model;
  std::string weightsFile = FLAGS_weights;
  std::string core = FLAGS_mcore;
  std::string outPath = "./";
  cnmlCoreVersion_t type;
  if (core == "1H8") {
    type = CNML_1H8;
  } else if (core == "1H16") {
    type = CNML_1H16;
  } else {
    type = CNML_MLU100;
  }
  if (modeNum == 0) {
    std::vector<std::string> inputs;
    inputs.push_back(prototxtFile);
    inputs.push_back(weightsFile);
    std::string name = FLAGS_mname;
    if (!caffe::compile(0, &inputs, &outPath, type, name)) {
      LOG(ERROR) << "compile failed";
      return -1;
    }
  } else if (modeNum == 1) {
    FILE* protoFp = fopen(prototxtFile.c_str(), "r");
    uint32_t protoSize = 0;
    if (!protoFp) {
      LOG(ERROR)<< "open prototxt file error";
      fclose(protoFp);
      return -1;
    }
    fseek(protoFp, 0L, SEEK_END);
    protoSize = ftell(protoFp);
    fseek(protoFp, 0L, SEEK_SET);
    uint8_t* protoBuf = new uint8_t[protoSize];
    int ret = fread(protoBuf, sizeof(uint8_t), protoSize, protoFp );
    fclose(protoFp);
    if (ret != protoSize) {
      LOG(ERROR) << "read prototxt error";
      return -1;
    }

    FILE* caffeFp = fopen(weightsFile.c_str(), "r");
    uint32_t caffeSize = 0;
    if (!caffeFp) {
      LOG(ERROR) << "open caffe file error";
      fclose(caffeFp);
      return -1;
    }
    fseek(caffeFp, 0L, SEEK_END);
    caffeSize = ftell(caffeFp);
    fseek(caffeFp, 0L, SEEK_SET);
    uint8_t* caffeBuf = new uint8_t[caffeSize];
    ret = fread(caffeBuf, sizeof(uint8_t), caffeSize, caffeFp);
    fclose(caffeFp);
    if (ret != caffeSize) {
      LOG(ERROR) << "read caffe model file error ";
      return -1;
    }
    std::vector<uint8_t*> buffer;
    std::vector<uint32_t> bufferSize;
    buffer.push_back(protoBuf);
    buffer.push_back(caffeBuf);
    bufferSize.push_back(protoSize);
    bufferSize.push_back(caffeSize);

    caffe::Caffe::set_rt_core(type);
    caffe::Caffe::set_mode(caffe::Caffe::MFUS);
    caffe::Caffe::setReshapeMode(caffe::Caffe::ReshapeMode::SETUPONLY);
    caffe::Net<float> *net_ =
    new caffe::Net<float>(protoBuf, protoSize, caffe::TEST);
    net_->CopyTrainedLayersFrom(reinterpret_cast<void*>(caffeBuf), caffeSize);

    net_->genOfflineModel("tmp");

    int offlineRealSize = 0;
    CNRT_CHECK(cnrtGetModelSize("tmp.cambricon", &offlineRealSize));
    LOG(INFO) << "offline model size " << offlineRealSize;
    delete net_;
    uint8_t* buildModel = new uint8_t[offlineRealSize];
    uint32_t buildSize = offlineRealSize;
    uint32_t realModelSize = 0;
    caffe::compile(0, buffer, bufferSize, buildModel,
           buildSize, realModelSize, type);
    if (realModelSize == offlineRealSize) {
      LOG(INFO) << "offline model dump to memory success!";
    }
    delete[] protoBuf;
    delete[] caffeBuf;
    delete[] buildModel;
    return 0;
  } else if (modeNum == 2) {
    if (FLAGS_offlinemodel != std::string("NULL")) {
      std::string offlineModelName = FLAGS_offlinemodel;

      FILE *offlineModelFp = fopen(offlineModelName.c_str(), "r");
      if (!offlineModelFp) {
        LOG(ERROR) << "failed to open offline model file";
        return -1;
      }

      int offlineModelSize = 0;
      cnrtGetModelSize(offlineModelName.c_str(), &offlineModelSize);
      std::ifstream testFp(offlineModelName, std::ios::in
          | std::ios::binary);
      LOG(INFO) << "offline mode size: " << offlineModelSize;
      int fileSize = 0;
      testFp.seekg(0, testFp.end);
      fileSize = testFp.tellg();
      int segSize = fileSize - offlineModelSize - 2;
      if (segSize <= 0) {
        LOG(ERROR) << "Offline model doesn't contain CPU info.";
        return -1;
      }
      LOG(INFO) << "segment real size:" << segSize;
      testFp.seekg(offlineModelSize + 2, testFp.beg);
      char *segBuf = new char[segSize];
      testFp.read(segBuf, segSize);
      testFp.close();

      caffe::SegmentInfo* segInfo = new caffe::SegmentInfo;
      ReadProtoFromBinaryMem(reinterpret_cast<void*>(segBuf),
          segSize, segInfo);

      // output CPU info to prototxt
      WriteProtoToTextFile(*segInfo, "compare_SegmentInfo.prototxt");
      delete[] segBuf;
    } else {
      LOG(ERROR) << "Please specify offline model file path";
    }
    return 0;
  }
}

#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU
