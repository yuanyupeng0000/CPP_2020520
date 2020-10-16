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
#include "caffe/layers/mlu_permute_layer.hpp"
#include "caffe/mlu/data_trans.hpp"

namespace caffe {

template <typename Dtype>
void MLUPermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  PermuteLayer<Dtype>::LayerSetUp(bottom, top);
  // if the order is 0,1,2,3, there is no need to transpose it.
  if (this->permute_order_.cpu_data()[0] == 0 &&
      this->permute_order_.cpu_data()[1] == 1 &&
      this->permute_order_.cpu_data()[2] == 2) {
    transpose_ = false;
  }
}

template <typename Dtype>
void MLUPermuteLayer<Dtype>::Reshape_tensor(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  // The shape of the output is determined based on the parameter order.
  vector<int> top_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(this->permute_order_.cpu_data()[i]));
  }
  BaseDataType cpu_dtype = sizeof(Dtype) == 4 ? DT_FLOAT32 : DT_DOUBLE;
  BaseDataType mlu_dtype = bottom[0]->mlu_type();
  if (this->layer_param_.permute_param().use_image()) {
    cpu_dtype = mlu_dtype = DT_UINT8;
    bottom[0]->set_cpu_type(cpu_dtype);
    bottom[0]->set_mlu_type(mlu_dtype);
    bottom[0]->set_preprocess(false);
    top[0]->set_preprocess(false);
  }  
  top[0]->Reshape(top_shape, cpu_dtype, mlu_dtype, CNML_TENSOR);
}

template <typename Dtype>
void MLUPermuteLayer<Dtype>::MLUCreateOpBindData(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (transpose_) {
    int length = bottom[0]->mlu_shape().size();

    vector<int> dim(length, 0);
    vector<int> input_order(length, 0);
    input_order[0] = 0;
    input_order[length - 1] = 1;
    for (int i = 1; i < length - 1; i++) {
      input_order[i] = i + 1;
    }
    const int* order_ptr = this->permute_order_.cpu_data();
    vector<int> permute_order(order_ptr, order_ptr + length);
    vector<int> mlu_order = to_mlu_shape(permute_order);
    for (int i = 0; i < length; i++) {
      vector<int>::iterator iter = find(input_order.begin(),
          input_order.end(), mlu_order[i]);
      dim[i] = std::distance(input_order.begin(), iter);
    }
    /* TransposeProOp */
    MLU_CHECK(cnmlCreateNdTransposeOpParam(&transpose_op_param_ptr_,
                                           dim.data(),
                                           length));
    MLU_CHECK(cnmlCreateNdTransposeProOp(&transpose_pro_op_ptr_,
                                          bottom[0]->mlu_tensor(),
                                          top[0]->mlu_tensor(),
                                          transpose_op_param_ptr_));
  } else {
    // DeviceMemcpyOp
    //   if nothing is done, the data simply needs to
    //   be copied from the bottom to the top.
    MLU_CHECK(cnmlCreateDeviceMemcpyOp(&memcpy_op_ptr_,
                                       bottom[0]->mlu_tensor(),
                                       top[0]->mlu_tensor()));
  }
}

template <typename Dtype>
void MLUPermuteLayer<Dtype>::MLUCompileOp() {
  if (transpose_) {
    MLU_CHECK(cnmlCompileBaseOp(transpose_pro_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  } else {
    MLU_CHECK(cnmlCompileBaseOp(memcpy_op_ptr_,
                                Caffe::rt_core(),
                                Caffe::core_number()));
  }
}

template <typename Dtype>
void MLUPermuteLayer<Dtype>::Forward_mlu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  if (transpose_) {
    MLU_CHECK(cnmlComputeNdTransposeProOpForward(transpose_pro_op_ptr_,
                                                 bottom[0]->mutable_mlu_data(),
                                                 top[0]->mutable_mlu_data(),
                                                 Caffe::forward_param(),
                                                 Caffe::queue()));
  } else {
    MLU_CHECK(cnmlComputeDeviceMemcpyOpForward_V3(memcpy_op_ptr_,
                                                  bottom[0]->mutable_mlu_data(),
                                                  top[0]->mutable_mlu_data(),
                                                  Caffe::forward_param(),
                                                  Caffe::queue()));
  }
}

template <typename Dtype>
void MLUPermuteLayer<Dtype>::fuse(MFusion<Dtype>* fuser) {
  if (transpose_) {
    fuser->fuse(transpose_pro_op_ptr_);
  } else {
    fuser->fuse(memcpy_op_ptr_);
  }
}

template <typename Dtype>
void MLUPermuteLayer<Dtype>::MLUDestroyOp() {
  if (transpose_pro_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&transpose_pro_op_ptr_));
    transpose_pro_op_ptr_ = nullptr;
  }
  if (transpose_op_param_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyNdTransposeOpParam(&transpose_op_param_ptr_));
    transpose_op_param_ptr_ = nullptr;
  }
  if (memcpy_op_ptr_ != nullptr) {
    MLU_CHECK(cnmlDestroyBaseOp(&memcpy_op_ptr_));
    memcpy_op_ptr_ = nullptr;
  }
}

INSTANTIATE_CLASS(MLUPermuteLayer);

}  // namespace caffe
#endif  // USE_MLU
