/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include "librknn.h"

using namespace std;
using namespace cv;

#define MAX_OUT_TENSOR 3
#define MAX_IN_TENSOR 3
rknn_context ctx;
rknn_input_output_num io_num;
rknn_tensor_attr output_attrs[MAX_OUT_TENSOR];
rknn_tensor_attr input_attrs[MAX_IN_TENSOR];
bool b_is_first_time = true;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}
static bool IsFile(std::string &filename) {
    struct stat   buffer;
    return (stat (filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

static bool IsDir(std::string &filefodler) {
    struct stat   buffer;
    return (stat (filefodler.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}
void ListDir(std::string &input_dir, std::vector<std::string> &files)
{
    DIR* dir = opendir(input_dir.c_str());
    dirent* p = NULL;
    while((p = readdir(dir)) != NULL)//开始逐个遍历
    {
        //这里需要注意，linux平台下一个目录中有"."和".."隐藏文件，需要过滤掉
        if(p->d_name[0] != '.')//d_name是一个char数组，存放当前遍历到的文件名
        {
            string name = input_dir + "/" + string(p->d_name);
            std::cout<< name << std::endl;
            files.push_back(name);
        }
    }
    closedir(dir);
}

static float logistic_activate(float x)
{
    return 1./(1. + exp(-x));
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

static double IntersectionOverUnion(DetectionObject &box_1, DetectionObject &box_2){

    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

static void NMS(std::vector<DetectionObject> &objects, const float iou_thresh){
    // Filtering overlapping boxes
    std::sort(objects.begin(), objects.end());
    for (int i = 0; i < objects.size(); ++i) {
        if (objects[i].confidence == 0)
            continue;
        for (int j = i + 1; j < objects.size(); ++j)
            if (IntersectionOverUnion(objects[i], objects[j]) >= iou_thresh)
                objects[j].confidence = 0;
    }
    for(std::vector<DetectionObject>::iterator iter=objects.begin(); iter!=objects.end(); )
    {
         if(iter->confidence == 0){
             iter = objects.erase(iter);
         }
         else{
             iter++;
         }
    }
}

static void ParseYOLOV3TinyRknnOutputs(const rknn_output outputs[],
                                       const rknn_tensor_attr output_attrs[],
                                       const rknn_tensor_attr input_attrs[],
                                       const rknn_input_output_num &io_num,
                                       const unsigned long original_im_h,
                                       const unsigned long original_im_w,
                                       const double threshold,
                                       const double iou_thresh,
                                       std::vector<DetectionObject> &objects,
                                       const unsigned int id_offset) {
    if(io_num.n_input != 1){
        throw std::runtime_error("YOLOV3 tiny inputs number != 1");
    }
    unsigned long resized_im_h = input_attrs[0].dims[1];
    unsigned long resized_im_w = input_attrs[0].dims[0];


    for(int i=0; i<io_num.n_output; i++){
        // --------------------------- Validating output parameters -------------------------------------
        const int out_blob_c = output_attrs[i].dims[2];
        int out_blob_h = output_attrs[i].dims[1];
        int out_blob_w = output_attrs[i].dims[0];
        if (out_blob_h != out_blob_w)
            throw std::runtime_error("Invalid size of output It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
            ", current W = " + std::to_string(out_blob_h));
        // --------------------------- Extracting layer parameters -------------------------------------

        int num = 3;
        int coords = 4;
        int classes = out_blob_c/num - coords - 1;
        std::vector<float> anchors;
        io_num.n_output==2 ? (anchors = {10,25,  20,50,  30,75, 50,125,  80,200,  150,150}):
                             (anchors = {10.0,13.0, 16.0,30.0, 33.0,23.0, 30.0,61.0, 62.0,45.0, 59.0,119.0, 116.0,90.0, 156.0,198.0, 373.0,326.0});
        auto side = out_blob_h;
        int anchor_offset = 0;
        anchor_offset = 6*(io_num.n_output-1-i);

        auto side_square = side * side;
        const float *output_blob = static_cast<float*>(outputs[i].buf);
        // --------------------------- Parsing YOLO Region output -------------------------------------
        for (int i = 0; i < side_square; ++i) {
            int row = i / side;
            int col = i % side;
            for (int n = 0; n < num; ++n) {
                int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
                int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);

                float scale = logistic_activate(output_blob[obj_index]);
                if (scale < threshold)
                    continue;
                double x = (col + logistic_activate(output_blob[box_index + 0 * side_square])) / side * resized_im_w;
                double y = (row + logistic_activate(output_blob[box_index + 1 * side_square])) / side * resized_im_h;
                double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
                double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
                //std::cout << "x:" << x << " y:"<< y << " h:" << height << " w:" << width << " resized_im_w:" << resized_im_w << std::endl;
                for (int j = 0; j < classes; ++j) {
                    int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                    float prob = scale * logistic_activate(output_blob[class_index]);
                    if (prob < threshold)
                        continue;
                    DetectionObject obj(x, y, height, width, j+id_offset, prob,
                            static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                            static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                    objects.push_back(obj);
                }
            }
        }
    }
    NMS(objects, iou_thresh);
}

void DrawBoxes(std::vector<DetectionObject> &objs, cv::Mat &frame_show, std::string &img_full_name, float thresh=0.3){
    for (auto &object : objs) {
        if (object.confidence < thresh){
            continue;
        }
        float confidence = object.confidence;
        if (confidence >= thresh) {
            //printf("xmin=%d,ymin=%d;xmax=%d,ymax=%d\n", object.xmin, object.ymin, object.xmax, object.ymax);
            cv::rectangle(frame_show, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(0, 0, 255), 2);
        }
    }
    cv::imwrite(img_full_name.substr(0, img_full_name.size()-strlen(".jpg")) + "_result.jpg", frame_show);
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
struct RknnDetParam {

    std::string model_rknn;
    float thresh_confidence, thresh_iou;

    RknnDetParam(std::string model_rknn="",
                   float thresh_confidence=0.3, float thresh_iou=0.5, int number_infer_requests=4) {
        this->model_rknn = model_rknn;
        this->thresh_confidence = thresh_confidence;
        this->thresh_iou = thresh_iou;
    }
};

static bool intel_dldt_query_str(const std::string& src, const std::string& query, std::string& target){
    size_t pos_start = src.find(query, 0);
    if(pos_start == std::string::npos){
        std::cout << "[ ERRO ] " << query << " not find ." << std::endl;
        return false;
    }
    else{
        size_t pos_end = src.find("\n", pos_start + query.size());
        if(pos_end == std::string::npos){
            std::cout << "[ ERRO ] " << query << " not find ." << std::endl;
            return false;
        }
        else{
            std::string value(src.substr(pos_start + query.size(), pos_end-pos_start-query.size()));
            if(value.empty()){
                std::cout << "[ ERRO ] " << query << " not find ." << std::endl;
                return false;
            }
            std::cout << "[ INFO ] " << query << value << std::endl;
            target = value;
            return true;
        }
    }
}
static void rknn_read_config(const std::string& config, RknnDetParam& rknn_detect_param){
    std::ifstream cfg(config);
    std::stringstream buffer;
    buffer << cfg.rdbuf();
    std::string contents(buffer.str());
    std::string target;
    if(intel_dldt_query_str(contents, "model_rknn=", target)){
        rknn_detect_param.model_rknn = target;
    }
    if(intel_dldt_query_str(contents, "thresh=", target)){
        rknn_detect_param.thresh_confidence = std::stof(target);
    }
    if(intel_dldt_query_str(contents, "nms=", target)){
        rknn_detect_param.thresh_iou = std::stof(target);
    }
}

int librknn_ai_init(const std::string& config){
    printf("[ INFO ] call rknn_ai_init\n");
    RknnDetParam rknn_det_param;
    rknn_read_config(config, rknn_det_param);

    ///rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = rknn_det_param.model_rknn.c_str();

    // Load RKNN Model
    printf("[ INFO ] call load_model\n");
    model = load_model(model_path, &model_len);
    printf("[ INFO ] load_model end\n");
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    ///rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    ///rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    ///rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }
    return 0;
}
int librknn_ai_detect(const unsigned char* pucframe, std::vector<DetectionObject>& objs){

    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    int ret = -1;
    const int MODEL_IN_WIDTH = 416;
    const int MODEL_IN_HEIGHT = 416;
    const int MODEL_IN_CHANNELS = 3;

    rknn_input inputs[io_num.n_input];
    rknn_output outputs[io_num.n_output];

    /*
    string img_path("test.jpg");
    std::string input_path(img_path);
    std::vector<std::string> imgs;
    if(IsFile(input_path)){
        imgs.push_back(input_path);
    }
    else if(IsDir(input_path)){
        ListDir(input_path, imgs);
    }*/

    for(int i=0; i<1/*imgs.size()*/; i++){
        // Load image
#if 0
        cv::Mat orig_img = imread(imgs[i], cv::IMREAD_COLOR);
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", imgs[i]);
            return -1;
        }
#else
        Mat orig_img = Mat(416, 416, CV_8UC3, (unsigned char*)pucframe, 0);
        //cv::Mat orig_img = cv::imread("17_55_551.jpg", cv::IMREAD_COLOR);
        //cv::imwrite("librknn_saved.jpg", orig_img);

#endif
        auto t22 = std::chrono::high_resolution_clock::now();
        cv::Mat img = orig_img.clone();
        auto t222 = std::chrono::high_resolution_clock::now();
        ms detection22 = std::chrono::duration_cast<ms>(t222 - t22);
        std::cout << "clone ms:" << detection22.count() << std::endl;
        if(orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
            printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
            auto t00 = std::chrono::high_resolution_clock::now();
            cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
            auto t11 = std::chrono::high_resolution_clock::now();
            ms detection0 = std::chrono::duration_cast<ms>(t11 - t00);
            std::cout << "resize ms:" << detection0.count() << std::endl;
        }

        //cv::cvtColor(img, img, COLOR_BGR2RGB);

        // Set Input Data
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = img.cols*img.rows*img.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = img.data;

        auto t0 = std::chrono::high_resolution_clock::now();
        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }

        // Run
        printf("rknn_run\n");
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // Get Output
        memset(outputs, 0, sizeof(outputs));
        for(int i=0; i<io_num.n_output; i++){
            outputs[i].want_float = 1;
        }
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }

        // Post Process
        ///std::vector<DetectionObject> objs;
        float confidence = 0.3;
        float iou_thresh = 0.5;
        ParseYOLOV3TinyRknnOutputs(outputs, output_attrs, input_attrs, io_num, orig_img.rows, orig_img.cols, confidence, iou_thresh, objs, 0);
        /*for (int i = 0; i < output_attrs[0].n_elems; i++) {
            float val = ((float*)(outputs[0].buf))[i];
            if (val > 0.01) {
                printf("%d - %f\n", i, val);
            }
        }*/
        auto t1 = std::chrono::high_resolution_clock::now();
        ms detection = std::chrono::duration_cast<ms>(t1 - t0);
        std::cout << "ms:" << detection.count() << std::endl;
        printf("nboxes = %d\n", objs.size());
        std:string save_name("test.jpg");
        DrawBoxes(objs, orig_img, save_name, confidence);

        // Release rknn_outputs
        rknn_outputs_release(ctx, io_num.n_output, outputs);
    }
    return 0;
}

void rknn_ai_alg(void *p, std::vector<DetectionObject> &objs){
    if(b_is_first_time){
        printf("This is a calling test, will call librknn_ai_init func\n");
        string config("rknn_config/rknn_config.ini");
        librknn_ai_init(config);
        printf("after calling librknn_ai_init\n");
        b_is_first_time = false;
    }
    librknn_ai_detect((const unsigned char*)p, objs);
}
int main(int argc, char** argv)
{
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    const int MODEL_IN_WIDTH = 416;
    const int MODEL_IN_HEIGHT = 416;
    const int MODEL_IN_CHANNELS = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    rknn_input inputs[io_num.n_input];
    rknn_output outputs[io_num.n_output];

    std::string input_path(img_path);
    std::vector<std::string> imgs;
    if(IsFile(input_path)){
        imgs.push_back(input_path);
    }
    else if(IsDir(input_path)){
        ListDir(input_path, imgs);
    }

    for(int i=0; i<imgs.size(); i++){
        // Load image
        cv::Mat orig_img = imread(imgs[i], cv::IMREAD_COLOR);
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", imgs[i]);
            return -1;
        }

        cv::Mat img = orig_img.clone();
        if(orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
            printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
            auto t00 = std::chrono::high_resolution_clock::now();
            cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
            auto t11 = std::chrono::high_resolution_clock::now();
            ms detection0 = std::chrono::duration_cast<ms>(t11 - t00);
            std::cout << "resize ms:" << detection0.count() << std::endl;
        }

        cv::cvtColor(img, img, COLOR_BGR2RGB);

        // Set Input Data
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = img.cols*img.rows*img.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = img.data;

        auto t0 = std::chrono::high_resolution_clock::now();
        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }

        // Run
        printf("rknn_run\n");
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // Get Output
        memset(outputs, 0, sizeof(outputs));
        for(int i=0; i<io_num.n_output; i++){
            outputs[i].want_float = 1;
        }
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }

        // Post Process
        std::vector<DetectionObject> objs;
        float confidence = 0.3;
        float iou_thresh = 0.5;
        ParseYOLOV3TinyRknnOutputs(outputs, output_attrs, input_attrs, io_num, orig_img.rows, orig_img.cols, confidence, iou_thresh, objs, 0);
        /*for (int i = 0; i < output_attrs[0].n_elems; i++) {
            float val = ((float*)(outputs[0].buf))[i];
            if (val > 0.01) {
                printf("%d - %f\n", i, val);
            }
        }*/
        auto t1 = std::chrono::high_resolution_clock::now();
        ms detection = std::chrono::duration_cast<ms>(t1 - t0);
        std::cout << "ms:" << detection.count() << std::endl;
        printf("nboxes = %d\n", objs.size());
        DrawBoxes(objs, orig_img, imgs[i]);
    }

    // Release rknn_outputs
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
