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

#ifdef USE_MLU

#include <vector>
#include "caffe/layers/mlu_image_crop_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLUImagecropLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ImagecropParameter crop_param = this->layer_param().image_crop_param();
  crop_x_ = crop_param.crop_x();
  crop_y_ = crop_param.crop_y();
  crop_w_ = crop_param.crop_w();
  crop_h_ = crop_param.crop_h();
  if (crop_w_ <= 1) {
    crop_w_ = bottom[0]->width();
  }
  if (crop_h_ <= 1) {
    crop_h_ = bottom[0]->height();
  }
  CHECK_LE(crop_x_ + crop_w_, bottom[0]->width())
      << "crop_x + crop_w should less than bottom width";
  CHECK_LE(crop_y_ + crop_h_, bottom[0]->height())
      << "crop_y + crop_h should less than bottom height";
  bottom[0]->set_preprocess(false);
  top[0]->set_preprocess(false);
}

template <typename Dtype>
void MLUImagecropLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  BaseDataType cpu_dtype = DT_UINT8;
  BaseDataType mlu_dtype = DT_UINT8;
  bottom[0]->set_cpu_type(cpu_dtype);
  bottom[0]->set_mlu_type(mlu_dtype);
  vector<int> top_shape(4, 1);
  top_shape[0] = bottom[0]->num();
  top_shape[1] = 4;
  top_shape[2] = crop_h_;
  top_shape[3] = crop_w_;
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUImagecropLayer<Dtype>::MLUDestroyOp() {
  if (crop_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyGrepOpParam(&crop_param_ptr_));
    crop_param_ptr_ = nullptr;
  }
  if (crop_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&crop_op_ptr_));
    crop_op_ptr_ = nullptr;
  }
}

template <typename Dtype>
void MLUImagecropLayer<Dtype>::MLUCompileOp() {
  MLU_CHECK(cnmlCompileBaseOp(crop_op_ptr_, Caffe::rt_core(),
                              Caffe::core_number()));
}

template <typename Dtype>
void MLUImagecropLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  fuser->fuse(crop_op_ptr_);
}

template <typename Dtype>
void MLUImagecropLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlCreateGrepOpParam(&crop_param_ptr_,
                                  0,
                                  crop_y_,
                                  crop_x_,
                                  0));
  MLU_CHECK(cnmlCreateGrepOp(&crop_op_ptr_,
                             crop_param_ptr_,
                             bottom[0]->mlu_tensor(),
                             top[0]->mlu_tensor()));
}

template <typename Dtype>
void MLUImagecropLayer<Dtype>::Forward_mlu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  MLU_CHECK(cnmlComputeGrepOpForward_V4(crop_op_ptr_,
                                        nullptr,
                                        bottom[0]->mutable_mlu_data(),
                                        nullptr,
                                        top[0]->mutable_mlu_data(),
                                        Caffe::queue(),
                                        nullptr));
}

INSTANTIATE_CLASS(MLUImagecropLayer);

}  // namespace caffe

#endif  // USE_MLU
