#include <caffe/caffe.hpp>
#include <iostream>
#include <set>
#include <string>
#include "test_yolo_input_data.hpp"

int main(int argc, char** argv) {
  // ::google::InitGoogleLogging(argv[0]);
  if (argc != 6 && argc != 7 && argc != 8) {
    LOG(ERROR) << "Usage1: "<< argv[0] <<" proto_file model_file"
    << " mlu_option mcore output_file";
    // in this mode, mlu_option must be 1, before the layer,
    // which the layer name is, net forwards in CPU,
    // after the layer, net forwards in MLU.
    LOG(ERROR) << "Usage2: "<< argv[0] <<" proto_file model_file"
     << " 1 mcore output_file layer_name";
    LOG(ERROR) << "Usage3: "<< argv[0] << " proto_file model_file"
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

  // float* input_data = net->input_blobs()[0]->mutable_cpu_data();
  for (int k = 0 ; k < net->input_blobs().size(); k++) {
    float* input_data = net->input_blobs()[k]->mutable_cpu_data();
    for (int i = 0; i < net->input_blobs()[k]->count(); ++i) {
      input_data[i]= yolo_input_data::input_cpu_data[i];
    }
  }

  if (argc == 7) {
    if (mlu_option != 1) {
        LOG(INFO) << "just support mode 1";
        return 0;
    }
    std::string start_layer_name = argv[6];
    int layer_idx = 0;
    bool find = false;
    for (int i = 0; i < net->layers().size(); ++i) {
      if (net->layers()[i]->layer_param().name() == start_layer_name) {
        layer_idx = i;
        find = true;
        break;
      }
    }
    if (find == false) {
      LOG(INFO) << "WARNNING: do not find the layer name,"
      << "please verify the name!!";
      return 0;
    }
    if (layer_idx == 0) {
      net->Forward();
    } else {
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
      net->ForwardFromTo(0, layer_idx - 1);
      caffe::Caffe::set_mode(caffe::Caffe::MLU);
      net->ForwardFromTo(layer_idx, net->layers().size() - 1);
    }

  } else {
    net->Forward();
  }
  LOG(INFO) << "Forward finish.";
  if (std::string(argv[5]) != std::string("NULL") && argc != 8) {
    std::ofstream fout(argv[5], std::ios::out);
    for (auto blob : net->output_blobs()) {
      for (int i = 0; i < blob->count(); i++) {
        fout << blob->cpu_data()[i] << std::endl;
      }
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
        LOG(INFO) << argv[0] << " proto_file model_file mlu_option"
        << "mcore output_file outputaddr_file outputsize_file";
        exit(1);
      }
    }
  }

#ifdef USE_MLU
  if (mlu_option > 0) {
    caffe::Caffe::freeQueue();
    cnmlExit();
  }
#endif
  return 0;
}
