"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch

from ..extensions._EXTRA import nms as nms_impl


class NMSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes, scores, threshold):
        if scores.dim() == 1:
            scores = scores.unsqueeze(1)
        keep = nms_impl(torch.cat((boxes, scores), dim=1), threshold)
        return boxes[keep, :], scores.squeeze(1)[keep]

    @staticmethod
    def backward(ctx, boxes):
        raise NotImplementedError


nms = NMSFunction.apply


# # Original author: Francisco Massa:
# # https://github.com/fmassa/object-detection.torch
# # Ported to PyTorch by Max deGroot (02/01/2017)
# def nms(boxes, scores, overlap=0.5, top_k=200):
#     """Apply non-maximum suppression at test time to avoid detecting too many
#     overlapping bounding boxes for a given object.
#     Args:
#         boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
#         scores: (tensor) The class predscores for the img, Shape:[num_priors].
#         overlap: (float) The overlap thresh for suppressing unnecessary boxes.
#         top_k: (int) The Maximum number of box preds to consider.
#     Return:
#         The indices of the kept boxes with respect to num_priors.
#     """
#
#     scores = scores.view(-1)
#     keep = torch.zeros_like(scores, dtype=torch.long)
#     # keep = scores.new(scores.size(0)).zero_().long()
#     if boxes.numel() == 0:
#         return keep, 0
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     area = torch.mul(x2 - x1, y2 - y1)
#     v, idx = scores.sort()  # sort in ascending order
#     # I = I[v >= 0.01]
#     idx = idx[-top_k:]  # indices of the top-k largest vals
#     xx1 = boxes.new()
#     yy1 = boxes.new()
#     xx2 = boxes.new()
#     yy2 = boxes.new()
#     w = boxes.new()
#     h = boxes.new()
#
#     # keep = torch.Tensor()
#     count = 0
#     while idx.numel() > 0:
#         i = idx[-1]  # index of current largest val
#         # keep.append(i)
#         keep[count] = i
#         count += 1
#         if idx.size(0) == 1:
#             break
#         idx = idx[:-1]  # remove kept element from view
#         # load bboxes of next highest vals
#         torch.index_select(x1, 0, idx, out=xx1)
#         torch.index_select(y1, 0, idx, out=yy1)
#         torch.index_select(x2, 0, idx, out=xx2)
#         torch.index_select(y2, 0, idx, out=yy2)
#         # store element-wise max with next highest score
#         xx1 = torch.clamp(xx1, min=x1[i])
#         yy1 = torch.clamp(yy1, min=y1[i])
#         xx2 = torch.clamp(xx2, max=x2[i])
#         yy2 = torch.clamp(yy2, max=y2[i])
#         w.resize_as_(xx2)
#         h.resize_as_(yy2)
#         w = xx2 - xx1
#         h = yy2 - yy1
#         # check sizes of xx1 and xx2.. after each iteration
#         w = torch.clamp(w, min=0.0)
#         h = torch.clamp(h, min=0.0)
#         inter = w * h
#         # IoU = i / (area(a) + area(b) - i)
#         rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
#         union = (rem_areas - inter) + area[i]
#         IoU = inter / union  # store result in iou
#         # keep only elements with an IoU <= overlap
#         idx = idx[IoU.le(overlap)]
#     return keep, count
