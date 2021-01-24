// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>

#include <torch/torch.h>

#include "nms/nms.h"
#include "roi_align/roi_align.h"
#include "deformable_conv/deform_conv_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "NMS");
  m.def("roi_align_forward", &roi_align_forward, "roi_align_forward");
  m.def("roi_align_backward", &roi_align_backward, "roi_align_backward");
  m.def("deform_conv_forward_cuda", &deform_conv_forward_cuda, "deform_conv_forward_cuda");
  m.def("deform_conv_backward_input_cuda", &deform_conv_backward_input_cuda, "deform_conv_backward_input_cuda");
  m.def("deform_conv_backward_parameters_cuda", &deform_conv_backward_parameters_cuda, "deform_conv_backward_parameters_cuda");
}
