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
#include <algorithm>
#include <vector>

#include "caffe/layers/strided_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StridedSliceLayer<Dtype>::LayerSetUp(
                                     const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const StridedSliceParameter& stridedslice_param =
                                        this->layer_param_.stridedslice_param();
  if (stridedslice_param.has_n_begin()) {
    n_begin_ = stridedslice_param.n_begin();
  }
  if (stridedslice_param.has_c_begin()) {
    c_begin_ = stridedslice_param.c_begin();
  }
  if (stridedslice_param.has_h_begin()) {
    h_begin_ = stridedslice_param.h_begin();
  }
  if (stridedslice_param.has_w_begin()) {
    w_begin_ = stridedslice_param.w_begin();
  }
  if (stridedslice_param.has_n_end()) {
    n_end_ = stridedslice_param.n_end();
  } else {
    n_end_ = bottom[0]->num();
  }
  if (stridedslice_param.has_c_end()) {
    c_end_ = stridedslice_param.c_end();
  } else {
    c_end_ = bottom[0]->channels();
  }
  if (stridedslice_param.has_h_end()) {
    h_end_ = stridedslice_param.h_end();
  } else {
    h_end_ = bottom[0]->height();
  }
  if (stridedslice_param.has_w_end()) {
    w_end_ = stridedslice_param.w_end();
  } else {
    w_end_ = bottom[0]->width();
  }
  if (stridedslice_param.has_n_stride()) {
    n_stride_ = stridedslice_param.n_stride();
  }
  if (stridedslice_param.has_c_stride()) {
    c_stride_ = stridedslice_param.c_stride();
  }
  if (stridedslice_param.has_h_stride()) {
    h_stride_ = stridedslice_param.h_stride();
  }
  if (stridedslice_param.has_w_stride()) {
    w_stride_ = stridedslice_param.w_stride();
  }
}
template <typename Dtype>
void StridedSliceLayer<Dtype>::Reshape(
                                    const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  int n_len = (n_end_ - n_begin_) > 0 ? n_end_ - n_begin_ : n_begin_ - n_end_;
  int h_len = (h_end_ - h_begin_) > 0 ? h_end_ - h_begin_ : h_begin_ - h_end_;
  int w_len = (w_end_ - w_begin_) > 0 ? w_end_ - w_begin_ : w_begin_ - w_end_;
  int c_len = (c_end_ - c_begin_) > 0 ? c_end_ - c_begin_ : c_begin_ - c_end_;
  int n_out = n_len / n_stride_ + (n_len % n_stride_ > 0 ? 1 : 0);
  int h_out = h_len / h_stride_ + (h_len % h_stride_ > 0 ? 1 : 0);
  int w_out = w_len / w_stride_ + (w_len % w_stride_ > 0 ? 1 : 0);
  int c_out = c_len / c_stride_ + (c_len % c_stride_ > 0 ? 1 : 0);

  vector<int> shape(4, 1);
  shape[0]= n_out;
  shape[1]= c_out;
  shape[2]= h_out;
  shape[3]= w_out;
  top[0]->Reshape(shape);
}
template <typename Dtype>
void StridedSliceLayer<Dtype>::Forward_cpu(
                                        const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  int ni = bottom[0]->num();
  int ci = bottom[0]->channels();
  int hi = bottom[0]->height();
  int wi = bottom[0]->width();
  int no = top[0]->num();
  int co = top[0]->channels();
  int ho = top[0]->height();
  int wo = top[0]->width();
  int ns = n_stride_;
  int cs = c_stride_;
  int hs = h_stride_;
  int ws = w_stride_;
  int nb = n_begin_;
  int cb = c_begin_;
  int hb = h_begin_;
  int wb = w_begin_;

  auto *input = bottom[0]->cpu_data();
  auto *output = top[0]->mutable_cpu_data();

  for (int nn = 0; nn < no; ++nn) {
    for (int cc = 0; cc < co; ++cc) {
      for (int hh = 0; hh < ho; ++hh) {
        for (int ww = 0; ww < wo; ++ww) {
          int nni = nb + nn * ns;
          if (nni < 0)
            nni = ni + nni;
          int hhi = hb + hh * hs;
          if (hhi < 0)
            hhi = hi + hhi;
          int wwi = wb + ww * ws;
          if (wwi < 0)
            wwi = wi + wwi;
          int cci = cb + cc * cs;
          if (cci < 0)
            cci = ci + cci;
          output[nn * co * ho * wo + cc * ho * wo + hh * wo + ww] =
                  input[nni * ci * hi * wi + cci * hi * wi + hhi * wi + wwi];
        }
      }
    }
  }
}

INSTANTIATE_CLASS(StridedSliceLayer);

}  // namespace caffe
