/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#ifndef INCLUDE_CAFFE_LAYERS_REGION_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_REGION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class RegionLayer : public Layer<Dtype> {
  public:
  explicit RegionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline const char* type() const { return "Region"; }
  virtual inline void set_int8_context(bool int8_mode) { int8_context = int8_mode; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  protected:
  bool int8_context;

  private:
  bool bias_match_;
  bool output_square_;
  bool softmax_;
  int num_;
  int classes_;
  int coords_;
  Dtype ratio_;
  Dtype box_thresh_;
  Dtype iou_thresh_;
  std::vector<Dtype> anchors_;

  private:
  Dtype sigmoid(Dtype x) {return 1./(1. + exp(-x));}
  Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
    Dtype l1 = x1 - w1/2;
    Dtype l2 = x2 - w2/2;
    Dtype left = l1 > l2 ? l1 : l2;
    Dtype r1 = x1 + w1/2;
    Dtype r2 = x2 + w2/2;
    Dtype right = r1 < r2 ? r1 : r2;
    return right - left;
  }
  Dtype box_intersection(Dtype* a, Dtype* b) {
    Dtype w = overlap(a[1], a[3], b[1], b[3]);
    Dtype h = overlap(a[2], a[4], b[2], b[4]);
    if (w < 0 || h < 0)
      return 0;
    Dtype area = w * h;
    return area;
  }

  Dtype box_iou(Dtype* a, Dtype* b) {
    Dtype i = box_intersection(a, b);
    Dtype u = a[3] * a[4] + b[3] * b[4] - i;
    return i / u;
  }
  void do_nms(int total, Dtype *probs, Dtype iou_thresh) {
    total *= 5;
    for (int i = 0; i < total; i += 5) {
      if (probs[i] < 0.1f) continue;
      for (int j = i + 5; j < total; j += 5) {
        if (probs[j] < 0.1f) continue;
        if (box_iou(probs + i, probs + j) > iou_thresh * probs[i] * probs[j]) {
          if (probs[i] < probs[j]) {
            probs[i] = 0.f;
          } else {
            probs[j] = 0.f;
          }
        }
      }
    }

    int k = 0;
    for (int i = 0; i < total; i += 5) {
      if (probs[i] < 0.1f) continue;
      memcpy(probs + k, probs + i, sizeof(Dtype) * 5);
      k += 5;
    }
    probs[k] = 0.;
  }

  void make_square(Dtype* probs) {
    for (int i = 0; probs[i] > 0.1f; i += 5) {
      Dtype w = probs[i + 3];
      Dtype h = probs[i + 4];
      Dtype s = sqrt(w * h);
      w = w < h ? w : h;
      s = w + (s - w) * 0.5;
      probs[i + 3] = probs[i + 4] = w;
    }
  }
};

}  // namespace caffe

#endif  // INCLUDE_CAFFE_LAYERS_REGION_LAYER_HPP_
