#ifndef RECOGNIZER_H
#define RECOGNIZER_H
#include <stdio.h>
#include <string>
#include <iostream>
//#include <samples/ocv_common.hpp>
#include <inference_engine.hpp>
#include "Common.h"
using namespace InferenceEngine;
#define LPR_MAX_INFER_SIZE 48
class Recognizer
{
public:
    Recognizer(const std::string& inputXml,
               const std::string& inputBin,
               const std::string& inputDevice,
               const float thresh=0.3,
               const float iou=0.7,
               const int nireq=4);
    int Recognize(int idx, const cv::Mat frame, std::vector<DetectionObject>& objects);
    void InitIdxMatMap();
    bool FindClassObjects(std::vector<DetectionObject>& objects,
                                      std::vector<int>& target_idxes,
                                      const int target_class_id);
    bool GetTargetFrames(cv::Mat frame, std::vector<DetectionObject>& objects, std::vector<int>& plate_idxes, std::vector<cv::Mat>& plate_frames);
    void FakeProvinceFeild(std::string& lpr_txt);
    bool CropObjectRegion(DetectionObject& object, cv::Mat frame, cv::Mat& object_region);
    void fillSeqBlob(InferRequest::Ptr);
    std::string GetLicencePlateText(InferRequest::Ptr);
    bool dobinaryzation(cv::Mat gray, cv::Mat& dst);
    bool ReLocateLicensePlate(cv::Mat& plate);
    bool ReLocateLicensePlates(std::vector<cv::Mat>& plate_frames);
    void CaculateXYLocation(cv::Mat& edges, cv::Point& left_top, cv::Point& right_bottom);
    ///CNNNetReader netReader;
    CNNNetwork cnnNetwork;
    //Core ie;
    ICNNNetwork::InputShapes inputShapes;
    InputsDataMap inputInfo;
    OutputsDataMap outputInfo;
    std::string outputName;
    const int maxSequenceSizePerPlate = 88;
    std::vector<std::string> labels;
    InferRequest::Ptr IfReqs[LPR_MAX_INFER_SIZE];
    float thresh;
    int nireq;
    int current_request_id;
    float iou;
    std::string input_xml;

    //TODO: change to multi cameras more than 4
    std::queue<cv::Mat> queue0;
    std::queue<cv::Mat> queue1;
    std::queue<cv::Mat> queue2;
    std::queue<cv::Mat> queue3;
    std::map<int, std::queue<cv::Mat>> idx_mat_map;
    /**
     * @brief A destructor
     */
    virtual ~Recognizer() {}

};

#endif // RECOGNIZER_H
