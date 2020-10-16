/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <cmath>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::string;
using std::vector;
using std::map;
using std::sqrt;

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "The pretrained weights to initialize finetuning, separated by ','."
    "Cannot be set simultaneously with snapshot.");

void layer_weights_copy(caffe::LayerParameter* dst,
                        const caffe::LayerParameter src) {
  dst->clear_blobs();
  caffe::BlobProto* proto;
  for (int i = 0; i < src.blobs_size(); i++) {
    // add shape
    proto = dst->add_blobs();
    proto->clear_shape();
    for (int j = 0; j < src.blobs(i).shape().dim_size(); j++) {
      proto->mutable_shape()->add_dim(src.blobs(i).shape().dim(j));
    }
    // add data
    if (src.blobs(i).data_size()) {
      for (int j = 0; j < src.blobs(i).data_size(); j++) {
        proto->add_data(src.blobs(i).data(j));
      }
    }
  }
}

vector<vector<float> > get_bn_alphabeta(caffe::LayerParameter layer) {
  vector<vector<float> > alphabeta(2);
  alphabeta[0].resize(layer.blobs(0).data_size(), 1);
  alphabeta[1].resize(layer.blobs(0).data_size(), 0);
  float scale = layer.blobs(2).data(0) == 0 ? 0 : 1 / layer.blobs(2).data(0);
  auto abs = [](double data) -> double { return data >= 0 ? data : -data; };
  for (int i = 0; i < alphabeta[0].size(); i++) {
    // NOTE: we must regard extremely small number e.g 1.234e-20 as 0
    // so that MLU calculation will not overflow
    if (scale != 0 && abs(layer.blobs(1).data(i)) > 1.0e-10) {
      alphabeta[0][i] = sqrt(1. / (scale * layer.blobs(1).data(i)));
    } else {
      alphabeta[0][i] = 0;
    }
  }

  for (int i = 0; i < alphabeta[0].size(); i++) {
    if (alphabeta[0][i] != 0)
      alphabeta[1][i] = -layer.blobs(0).data(i) * scale * alphabeta[0][i];
  }

  if (layer.blobs_size() > 3) {
    for (int i = 0; i < alphabeta[0].size(); i++) {
      alphabeta[0][i] *= layer.blobs(3).data(i);
    }
    for (int i = 0; i < alphabeta[1].size(); i++) {
      alphabeta[1][i] += layer.blobs(4).data(i);
    }
  }

  return alphabeta;
}

vector<vector<float> > get_scale_alphabeta(caffe::LayerParameter layer) {
  vector<vector<float> > alphabeta(2);
  alphabeta[0].resize(layer.blobs(0).data_size(), 1);
  alphabeta[1].resize(layer.blobs(0).data_size(), 0);
  for (int i = 0; i < layer.blobs(0).data_size(); i++) {
    alphabeta[0][i] = layer.blobs(0).data(i);
  }
  if (layer.blobs_size() >= 2) {
    for (int i = 0; i < layer.blobs(1).data_size(); i++) {
      alphabeta[1][i] = layer.blobs(1).data(i);
    }
  }
  return alphabeta;
}

void update_conv_weights(caffe::LayerParameter* layer,
                         const vector<vector<float> >& alphabeta) {
  vector<int> shape;
  layer->mutable_convolution_param()->clear_bias_term();
  caffe::BlobProto* proto = layer->mutable_blobs(0);
  if (proto->has_num() || proto->has_channels() || proto->has_height() ||
      proto->has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    shape.resize(4);
    shape[0] = proto->num();
    shape[1] = proto->channels();
    shape[2] = proto->height();
    shape[3] = proto->width();
  } else {
    shape.resize(proto->shape().dim_size());
    for (int i = 0; i < proto->shape().dim_size(); ++i) {
      shape[i] = proto->shape().dim(i);
    }
  }
  CHECK_EQ(shape[0], alphabeta[0].size()) << "Output channels should be equal!";
  int inner_size = 1;
  for (int i = 1; i < shape.size(); i++) {
    inner_size *= shape[i];
  }

  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < inner_size; j++) {
      if (proto->double_data_size() > 0) {
        proto->set_double_data(
            i * inner_size + j,
            proto->double_data(i * inner_size + j) * alphabeta[0][i]);
      } else {
        proto->set_data(
            i * inner_size + j,
            proto->data(i * inner_size + j) * alphabeta[0][i]);
      }
    }
  }

  if (layer->blobs_size() == 1) {
    proto = layer->add_blobs();
    for (int i = 0; i < shape[0]; i++) {
      if (layer->blobs(0).data_size())
        proto->add_data(0);
      else
        proto->add_double_data(0);
    }
  }

  proto = layer->mutable_blobs(1);
  for (int i = 0; i < shape[0]; i++) {
    if (proto->double_data_size() > 0) {
      if (alphabeta[0][i] != 0) {
        proto->set_double_data(
            i, proto->double_data(i) * (alphabeta[0][i]) + alphabeta[1][i]);
      }
    } else {
      if (alphabeta[0][i] != 0) {
        proto->set_data(
            i, proto->data(i) * (alphabeta[0][i]) + alphabeta[1][i]);
      }
    }
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  gflags::SetUsageMessage(
      "command line brew \n"
      "usage: optimize_net \n\n"
      "-model net.prototxt \n"
      "--weights net.caffemodel");
  // Google flags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Google logging.
  google::InitGoogleLogging(argv[0]);
  // Provide a backtrace on segfault.
  google::InstallFailureSignalHandler();

  CHECK_GT(FLAGS_model.size(), 0) << "Need model file.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need weights file.";

  caffe::NetParameter model_param, weights_param;
  ReadProtoFromTextFileOrDie(FLAGS_model, &model_param);
  ReadNetParamsFromBinaryFileOrDie(FLAGS_weights, &weights_param);

  // copy weights
  for (int i = 0; i < model_param.layer_size(); i++) {
    for (int j = 0; j < weights_param.layer_size(); j++) {
      if (weights_param.layer(j).name() == model_param.layer(i).name()) {
        layer_weights_copy(model_param.mutable_layer(i),
                           weights_param.layer(j));
      }
    }
  }

  // get layer info
  vector<vector<int> > layer_optimized(model_param.layer_size());
  vector<int> layer_passed(model_param.layer_size(), 0);
  for (int i = 0; i < model_param.layer_size(); i++) {
    string blob_name, blob_next;
    if (model_param.layer(i).type() == "Convolution") {
      blob_name = model_param.layer(i).top(0);
      blob_next = model_param.layer(i).top(0);
      for (int j = i + 1; j < model_param.layer_size(); j++) {
        if (model_param.layer(j).type() == "BatchNorm" &&
            model_param.layer(j).bottom(0) == blob_name &&
            model_param.layer(j).batch_norm_param().use_global_stats()) {
          layer_optimized[i].push_back(j);
          blob_next = model_param.layer(j).top(0);
        } else if (model_param.layer(j).type() == "Scale" &&
                   model_param.layer(j).bottom(0) == blob_next &&
                   model_param.layer(j).bottom_size() == 1 &&
                   model_param.layer(j).scale_param().axis() == 1 &&
                   model_param.layer(j).scale_param().num_axes() == 1) {
          layer_optimized[i].push_back(j);
          blob_next = model_param.layer(j).top(0);
        } else if (model_param.layer(j).bottom(0) == blob_name &&
                   blob_name != blob_next) {
          layer_optimized[i].clear();
          break;
        } else if (model_param.layer(j).bottom(0) == blob_name &&
                   blob_name == blob_next) {
          break;
        }
      }
    }
  }

  // print info
  for (int i = 0; i < model_param.layer_size(); i++) {
    if (layer_optimized[i].size()) {
      LOG(INFO) << "=================conv layer "
                << model_param.layer(i).name();
      for (int j = 0; j < layer_optimized[i].size(); j++) {
        int k = layer_optimized[i][j];
        LOG(INFO) << model_param.layer(k).name();
      }
      LOG(INFO) << "=====================";
    }
  }

  // update weigths
  for (int i = 0; i < model_param.layer_size(); i++) {
    if (layer_optimized[i].size()) {
      for (int j = 0; j < layer_optimized[i].size(); j++) {
        int index = layer_optimized[i][j];
        layer_passed[index] = 1;
        vector<vector<float> > alphabeta;
        if (model_param.layer(index).type() == "BatchNorm") {
          alphabeta = get_bn_alphabeta(model_param.layer(index));
        } else if (model_param.layer(index).type() == "Scale") {
          alphabeta = get_scale_alphabeta(model_param.layer(index));
        } else {
          LOG(FATAL) << "Wrong layer type!";
        }
        update_conv_weights(model_param.mutable_layer(i), alphabeta);
      }
    }
  }

  // delete optimized layer
  weights_param.CopyFrom(model_param);
  weights_param.clear_layer();
  for (int i = 0; i < model_param.layer_size(); i++) {
    if (!layer_passed[i]) {
      caffe::LayerParameter* layer_param = weights_param.add_layer();
      layer_param->CopyFrom(model_param.layer(i));
    } else {
      if (model_param.layer(i).bottom(0) != model_param.layer(i).top(0)) {
        for (int j = i + 1; j < model_param.layer_size(); j++) {
          for (int k = 0; k < model_param.layer(j).bottom_size(); k++) {
            if (model_param.layer(j).bottom(k) == model_param.layer(i).top(0)) {
              model_param.mutable_layer(j)->set_bottom(
                  k, model_param.layer(i).bottom(0));
              for (int p = 0; p < model_param.layer(j).top_size(); p++) {
                if (model_param.layer(i).top(0) == model_param.layer(j).top(p)) {
                  model_param.mutable_layer(j)->set_top(p,
                      model_param.layer(i).bottom(0));
                }
              }
            }
          }
        }
      }
    }
  }

  WriteProtoToBinaryFile(weights_param, "optimized.caffemodel");
  for (int i = 0; i < weights_param.layer_size(); ++i) {
    weights_param.mutable_layer(i)->clear_blobs();
  }
  WriteProtoToTextFile(weights_param, "optimized.prototxt");
}
