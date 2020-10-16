#include <assert.h>
#include <caffe/caffe.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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
  if (argc != 6) {
    LOG(ERROR) << "Usage1: ./test_roi_pooling_on proto_file model_file"
    << " mlu_option mcore output_file";
    exit(1);
  }

  std::string proto_file = argv[1];
  std::string model_file = argv[2];

  std::stringstream ss;
  int mlu_option;
  ss << argv[3];
  ss >> mlu_option;
  if (mlu_option < 0 || mlu_option > 2) {
    LOG(INFO) << "Unrecognized mlu option.";
    LOG(INFO) << "Avaliable option: 0(cpu), 1(mlu),"
    << "2(mlu with layer combination)";
    exit(1);
  }

#ifdef USE_MLU
  LOG(INFO) << ">>>>>> test forward >>> mlu option ???" << mlu_option;
  if (mlu_option > 0) {
    // FIXME: set
    // core version: MLU100, 1H16, 1H8
    cnmlInit(0);
    caffe::Caffe::set_rt_core(argv[4]);
    caffe::Caffe::set_mlu_device(0);
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


  int num_rois = 112;
  int raw_count = 1*256*32*32;
  float* data_raw = static_cast<float*>(malloc(raw_count*sizeof(float)));
  for (int i = 0; i < raw_count; i++) {
    data_raw[i] = i * 0.01;
  }


  float* cx = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* cy = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* w = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* h = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* idx = static_cast<float*>(malloc(num_rois*sizeof(float)));
  memset(idx, 0, num_rois * sizeof(float)); // NOLINT
  pad_uniform_distribution_data(cx, num_rois, 1000, 5, 32);
  pad_uniform_distribution_data(cy, num_rois, 1000, 5, 32);
  pad_uniform_distribution_data(w, num_rois, 1000, 0, 10);
  pad_uniform_distribution_data(h, num_rois, 1000, 0, 10);

  float* x1 = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* y1 = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* x2 = static_cast<float*>(malloc(num_rois*sizeof(float)));
  float* y2 = static_cast<float*>(malloc(num_rois*sizeof(float)));
  for (int i = 0; i < num_rois; i++) {
    x1[i] = cx[i] - w[i] / 2;
    x1[i] = std::min(x1[i], static_cast<float>(32));
    y1[i] = cy[i] - h[i] / 2;
    y1[i] = std::min(y1[i], static_cast<float>(32));
    x2[i] = cx[i] + w[i] / 2;
    x2[i] = std::min(x2[i], static_cast<float>(32));
    y2[i] = cy[i] + h[i] / 2;
    y2[i] = std::min(y2[i], static_cast<float>(32));
  }

  free(cx);
  free(cy);
  free(w);
  free(h);

  int unit_num = 5;
  // float* input_data = net->input_blobs()[0]->mutable_cpu_data();
  if (mlu_option == 0) {
    float* concat_data = static_cast<float*>
      (malloc(num_rois * unit_num * sizeof(float)));
    for (int i = 0; i < num_rois; i++) {
      concat_data[i * unit_num] = idx[i];
      concat_data[i * unit_num + 1] = x1[i];
      concat_data[i * unit_num + 2] = y1[i];
      concat_data[i * unit_num + 3] = x2[i];
      concat_data[i * unit_num + 4] = y2[i];
    }
    net->input_blobs()[0]->set_cpu_data(data_raw);
    net->input_blobs()[1]->set_cpu_data(concat_data);
  } else {
    float* rois_conc_data = static_cast<float*>
      (malloc(num_rois * unit_num * sizeof(float)));
    for (int i = 0; i < num_rois; i++) {
      rois_conc_data[i * unit_num] = x1[i];
      rois_conc_data[i * unit_num + 1] = y1[i];
      rois_conc_data[i * unit_num + 2] = x2[i];
      rois_conc_data[i * unit_num + 3] = y2[i];
      rois_conc_data[i * unit_num + 4] = idx[i];
    }
    net->input_blobs()[0]->set_cpu_data(data_raw);
    net->input_blobs()[1]->set_cpu_data(rois_conc_data);
  }
  net->Forward();
  std::ofstream fout(argv[5], std::ios::out);
  for (auto blob : net->output_blobs()) {
    for (int i = 0; i < blob->count(); i++) {
      fout << blob->cpu_data()[i] << std::endl;
    }
  }
  fout << std::flush;
  fout.close();

  free(x1);
  free(y1);
  free(x2);
  free(y2);
  free(idx);
#ifdef USE_MLU
  if (mlu_option > 0) {
    caffe::Caffe::freeQueue();
    cnmlExit();
  }
#endif
  return 0;
}
