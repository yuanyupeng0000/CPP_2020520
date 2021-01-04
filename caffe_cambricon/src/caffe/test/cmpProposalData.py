#! /usr/bin/env python
from __future__ import division
import sys
# this tool calculate the ratio of iou > 0.5 box in all the boxs
# the closer to 1 the better the similarity of two input

# note that for now(2018/11/1)
# caffe's proposal output is (n,x1,y1,x2,y2)
# but cnml's output is (x1,y1,x2,y2)
# they are working on a new mode to adapt caffe
def sort_box(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    new_box = [min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)]
    return new_box

def is_intersect(box_a,box_b):
    xa1 = box_a[0]
    ya1 = box_a[1]
    xa2 = box_a[2]
    ya2 = box_a[3]
    xb1 = box_b[0]
    yb1 = box_b[1]
    xb2 = box_b[2]
    yb2 = box_b[3]
    return  ((xa1 <= xb1 <= xa2) or (xa1 <= xb2 <= xa2)) and \
            ((ya1 <= yb1 <= ya2) or (ya1 <= yb2 <= ya2))

def iou(box_a,box_b):
    box_a = sort_box(box_a)
    box_b = sort_box(box_b)
    if not is_intersect(box_a, box_b):
        return 0
    xA = max(box_a[0],box_b[0])
    yA = max(box_a[1],box_b[1])
    xB = min(box_a[2],box_b[2])
    yB = min(box_a[3],box_b[3])
    intersect = abs(xB - xA) * abs(yB - yA)
    S_box_a = abs(box_a[0] - box_a[2]) * abs(box_a[1] - box_a[3])
    S_box_b = abs(box_b[0] - box_b[2]) * abs(box_b[1] - box_b[3])
    if S_box_a == 0 or S_box_b == 0:
        return 0
    return intersect / (S_box_a + S_box_b - intersect)


if len(sys.argv) != 3:
    print('usage: cmpProposalData.py data_a data_b')
data_a = []
data_b = []
get_data = lambda x:float(x.split('\n')[0])
with open(sys.argv[1],'r') as f:
    content = f.readlines()
    data_a = map(get_data,content)
with open(sys.argv[2],'r') as f:
    content = f.readlines()
    data_b = map(get_data,content)

assert len(data_a) % 5 == 0
assert len(data_b) % 5 == 0
assert len(data_a) == len(data_b)
total_dis = 0.
chunk_data = lambda l,n:[l[i:i+n] for i in range(0,len(l),n)]
data_a = chunk_data(data_a,5)
data_b = chunk_data(data_b,5)
paired_num = 0
for pos_a in data_a:
    max_iou = 0
    for pos_b in data_b:
        tmp_iou = iou(pos_a[1:],pos_b[0:4])
        if tmp_iou > max_iou:
            max_iou = tmp_iou
    if max_iou > 0.7:
        paired_num += 1
print paired_num/len(data_a)
