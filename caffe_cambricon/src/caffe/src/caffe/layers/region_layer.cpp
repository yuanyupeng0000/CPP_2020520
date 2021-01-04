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

#include <algorithm>
#include <vector>

#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void RegionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  RegionParameter region_param = this->layer_param_.region_param();
  bias_match_ = region_param.bias_match();
  output_square_ = region_param.output_square();
  softmax_ = region_param.softmax();
  num_ = region_param.num();
  classes_ = region_param.classes();
  coords_ = region_param.coords();
  ratio_ = region_param.ratio();
  iou_thresh_ = region_param.iou_thresh();
  box_thresh_ = region_param.box_thresh();
  box_thresh_ = -log(1.0f / box_thresh_ - 1);

  int anchor_size = num_ * 2;
  if (bias_match_) {
    CHECK_EQ(anchor_size, region_param.anchors().size());
    for (int i = 0; i < anchor_size; i++)
      anchors_.push_back(region_param.anchors(i));
  } else {
    for (int i = 0; i < anchor_size; i++)
      anchors_.push_back(1.0);
  }

  CHECK_LE(classes_, 1);  // only support: classes <= 1
  CHECK_LE(coords_, 4);   // only support: coords == 4
  // CHECK_EQ(num_ * (coords_ + classes_ + 1), bottom[0]->channels());
}

template <typename Dtype>
void RegionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  int batch = bottom[0]->num();
  int w = bottom[0]->width();
  int h = bottom[0]->height();
  int grid = w * h;
  int n = coords_ + classes_;
  int output_jump = (n + 1) * grid;
  int input_jump = n * grid;

  Dtype* input = bottom[0]->mutable_cpu_data();
  Dtype* output = top[0]->mutable_cpu_data();
  input += input_jump;

  for (int b = 0, boxes = 0; b < batch; b++, output += output_jump) {
    Dtype* box = output;
    for (int k = 0; k < num_; k++, input += input_jump) {
      for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++, input++) {
          if (*input >= box_thresh_) {
            *box++ = sigmoid(*input);
            *box++ = ratio_ * (sigmoid(input[-n*grid]) + i);
            *box++ = ratio_ * (sigmoid(input[-(n-1)*grid]) + j);
            *box++ = ratio_ * exp(input[-(n-2)*grid]) * anchors_[2*k];
            *box++ = ratio_ * exp(input[-(n-3)*grid]) * anchors_[2*k+1];
            boxes++;
          }
        }
      }
    }

    if (boxes == num_ * grid) boxes--;
    do_nms(boxes, output, iou_thresh_);

    if (output_square_) make_square(output);
  }
}

template <typename Dtype>
void RegionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(RegionLayer);
}  // namespace caffe
