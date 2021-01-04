#include <assert.h>
#include <caffe/caffe.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>


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
  if (argc != 6) {
    LOG(ERROR) << "Usage1: "<< argv[0] << " proto_file model_file"
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
    << "2(mlu with layer combination)" << std::endl;
    exit(1);
  }

#ifdef USE_MLU
  LOG(INFO) << ">>>>>> test forward >>> mlu option ???" << mlu_option;
  if (mlu_option > 0) {
    // FIXME: set
    // core version: MLU100, 1H16, 1H8
    cnmlInit(0);
    caffe::Caffe::set_rt_core(argv[4]);
    caffe::Caffe::set_mlu_device(FLAGS_mludevice);
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
    NParray* np_rois = new NParray(rois_conc_data, rois_shape);
    net->input_blobs()[0]->set_cpu_data(np_data->data);
    net->input_blobs()[1]->set_cpu_data(np_rois->data);
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
