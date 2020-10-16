/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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

#include <caffe/caffe.hpp>
#include <iostream>
#include <set>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>

using std::string;
using std::stringstream;
using std::ofstream;
using std::ifstream;
using std::endl;


DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
DEFINE_int32(mp, 1, "mp");
DEFINE_int32(batchsize, 1, "batchsize");
DEFINE_int32(core_number, 1, "batchsize");

void readYUV(string name, cv::Mat img, int h, int w) {
  std::ifstream fin(name);
  unsigned char a;
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) {
      a = fin.get();
      img.at<char>(i, j) = a;
      fin.get();
    }
  fin.close();
}

cv::Mat yuv420sp2Bgr24(cv::Mat yuv_image) {
  cv::Mat bgr_image(yuv_image.rows / 3 * 2, yuv_image.cols, CV_8UC3);
  cvtColor(yuv_image, bgr_image, CV_YUV420sp2BGR);
  return bgr_image;
}

cv::Mat convertYuv2Mat(string img_name, cv::Size inGeometry) {
  cv::Mat img = cv::Mat(inGeometry, CV_8UC1);
  readYUV(img_name, img, inGeometry.height, inGeometry.width);
  return img;
}

cv::Mat convertYuv2Mat(string img_name, int width, int height) {
  cv::Size inGeometry_(width, height);
  return convertYuv2Mat(img_name, inGeometry_);
}

void rand1(uint8_t* data, int length) {
  for (int i = 0; i < length; i++) {
     data[i] = i % 255;
  }
}

//void rand1(uint8_t* data, int length) {
//  cv::Mat yuv = convertYuv2Mat("0.jpg.yuv", 450, 320);
//  for (int i = 0; i < 450 * 320; i++) {
//     data[i] = yuv.data[i];
//  }
//}

//void rand1(uint8_t* data, int height, int width) {
//  cv::Mat bgr = cv::imread("demo.jpg");
//  cv::Mat resize;
//  cv::resize(bgr, resize, cv::Size(width, height));
//  for (int i = 0; i < resize.cols*resize.rows; i++) {
//      data[i*4] = resize.data[i*3+2];
//      data[i*4+1] = resize.data[i*3+1];
//      data[i*4+2] = resize.data[i*3];
//      data[i*4+3] = 0;
//  }
//}

//void rand2(uint8_t* data, int length) {
//  cv::Mat yuv = convertYuv2Mat("0.jpg.yuv", 1920, 1620);
//  for (int n=0; n<16; n++) {
//  for (int i = 0; i < 1920 * 540; i++) {
//    int index = i + 1920 * 1080;
//    data[i] = yuv.data[index];
//  }
//  data += (1920 * 540);
//  }
//}

void release_resource(int mlu_option) {
#ifdef USE_MLU
  if (mlu_option > 0) {
    caffe::Caffe::freeQueue();
    cnmlExit();
  }
#endif
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 6 && argc != 7 && argc != 8) {
    LOG(ERROR) << "Usage1: " << argv[0] << " proto_file model_file"
              << " mlu_option mcore output_file";
    LOG(ERROR) << "Usage2: " << argv[0] << " proto_file model_file"
              << " mlu_option mcore output_file outputaddr_file"
              << " outputsize_file";
    exit(1);
  }
  std::string proto_file = argv[1];
  std::string model_file = argv[2];

  std::stringstream ss;
  int mlu_option;
  ss << argv[3];
  ss >> mlu_option;
  if (mlu_option < 0 || mlu_option > 2) {
    LOG(ERROR) << "Unrecognized mlu option: " << mlu_option
              << "Available options: 0(cpu), 1(mlu), 2(mfus: mlu with fusion).";
    exit(1);
  }

  std::string mcore = argv[4];
  if ((mcore != std::string("VENTI")) &&
      (mcore != std::string("MLU270")) &&
      (mcore != std::string("MLU220"))) {
    LOG(ERROR) << "Unrecognized mcore option: " << mcore
              << "Available options: MLU220, MLU270";
    exit(1);
  }

#ifdef USE_MLU
  LOG(INFO) << ">>> test forward >>> mlu option ???" << mlu_option;
  if (mlu_option > 0) {
    cnmlInit(0);
    caffe::Caffe::set_rt_core(argv[4]);
    caffe::Caffe::set_mlu_device(FLAGS_mludevice);
    caffe::Caffe::setDetectOpMode(1);
    caffe::Caffe::setSimpleFlag(true);
    caffe::Caffe::setBatchsize(FLAGS_batchsize);
    caffe::Caffe::setCoreNumber(FLAGS_core_number);
    caffe::Caffe::set_mode(caffe::Caffe::MLU);
    if (mlu_option == 2) {
      caffe::Caffe::set_mode(caffe::Caffe::MFUS);
    }
    caffe::Caffe::setReshapeMode(caffe::Caffe::ReshapeMode::SETUPONLY);
  } else {
    LOG(INFO) << "Use CPU.";
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
#else
  LOG(INFO) << "Use CPU.";
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

  caffe::Net<float>* net = new caffe::Net<float>(proto_file, caffe::TEST);
  if (model_file != std::string("NULL")) {
    net->CopyTrainedLayersFrom(model_file);
  }

  int k = 0;  
  //rand1((uint8_t*)net->input_blobs()[k]->mutable_cpu_data(),
  //    net->input_blobs()[k]->height(), net->input_blobs()[k]->width());
  rand1((uint8_t*)net->input_blobs()[k]->mutable_cpu_data(),
      net->input_blobs()[k]->count());

  if (argc == 7) {
    LOG(WARNING) << "This usage is not supported any more. Use other appropriate ones.";
    release_resource(mlu_option);
    exit(2);
  } else {
    net->Forward();
  }
  LOG(INFO) << "Forward finished.";
  if (std::string(argv[5]) != std::string("NULL") && argc != 8) {
    std::ofstream fout(argv[5], std::ios::out);
    for (auto blob : net->output_blobs()) {
      // transpose
      //void* dst_ptr = (void*)malloc(blob->count() * sizeof(uint8_t));
      //int dimvalues[4] = {blob->num(), blob->height(), blob->channels(), blob->width()};
      //int dimorder[4] = {0, 2, 3, 1};
      //cnrtTransDataOrder((void*)blob->cpu_data(), CNRT_UINT8, dst_ptr, 4, dimvalues, dimorder); 
      for (int i = 0; i < blob->count(); i++) {
        fout <<(int) ((uint8_t*)blob->cpu_data())[i] << std::endl;
        //fout <<blob->cpu_data()[i] << std::endl;
      }
      LOG(INFO) << "size: " << blob->channels() << ", " << blob->width();
      cv::Mat outcv(blob->height(), blob->width(), CV_8UC4, (uint8_t*)blob->cpu_data());
      std::stringstream ss;
      ss << "test.jpg";
      cv::imwrite(ss.str(), outcv);
    }
    fout << std::flush;
    fout.close();
  } else {
    if (std::string(argv[6]) != std::string("NULL")) {
      if (std::string(argv[7]) != std::string("NULL")) {
        std::ofstream fout1(argv[6], std::ios::out);
        std::ofstream fout2(argv[7], std::ios::out);
        for (int k = 0; k < net->output_blobs().size(); k++) {
          fout1 << net->output_blobs()[k]->cpu_data() << std::endl;
          fout2 << net->output_blobs()[k]->count() << std::endl;
        }
        fout1 << std::flush;
        fout2 << std::flush;
        fout1.close();
        fout2.close();
      } else {
        LOG(INFO) << "Usage: " << argv[0] << " proto_file model_file mlu_option mcore"
                  << " output_file outputaddr_file outputsize_file";
        release_resource(mlu_option);
        exit(1);
      }
    }
  }

  release_resource(mlu_option);
  return 0;
}
