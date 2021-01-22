#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <gflags/gflags.h>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
//#include <ext_list.hpp>
#include "intel_dldt.h"

#define yolo_scale_13 13
#define yolo_scale_26 26
#define yolo_scale_52 52
#define yolo_scale_19 19
#define yolo_scale_38 38
#define yolo_scale_76 76
#define yolo_scale_27 27
#define yolo_scale_54 54
#define yolo_scale_108 108

using namespace InferenceEngine;
/*struct DetectedObject
{
    int left, top, right, bottom;
    float confidence;
    int objType;
    std::string name;
};*/

struct ibox
{
    float x, y, w, h;
};

struct indexsort
{
    int iclass;
    int index;
    int channel;
    float* prob;
};
/*struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
};*/
void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName);
void ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects);
void ParseYOLOV3TinyNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects);
void ParseYOLOV3TinyNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, const unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objects);
void ParseYOLOV3TinyNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, const unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objectsconst, const unsigned int id_offset = 0);
void ParseYOLOV5SOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objects);
void ParseYOLOV3TinyNcsOutputHW(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, const unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objects);
void ParseSSDNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects);
double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2);
void ChangeMotorLPR2VeichleLPR(cv::Mat& motor_lpr, cv::Mat& veichle_lpr);

int indexsort_comparator(const void *pa, const void *pb);

float logistic_activate(float x);
void transpose(float *src, float* tar, int k, int n);
void softmax(float *input, int n, float temp, float *output);
float overlap(float x1, float w1, float x2, float w2);
float box_intersection(ibox a, ibox b);
float box_union(ibox a, ibox b);
float box_iou(ibox a, ibox b);
int max_index(float *a, int n);

int cvPutChineseTextTest();


void ShowVec(const std::vector<int>& valList);

