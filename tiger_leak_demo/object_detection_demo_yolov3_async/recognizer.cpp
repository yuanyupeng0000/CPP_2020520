#include "recognizer.h"
#include "Common.h"
#include <samples/slog.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <stdio.h>
#include <gflags/gflags.h>
#include <samples/common.hpp> //#include <samples/ocv_common.hpp>
//#include <ext_list.hpp>
#include <chrono>
#define ONE_LINE_LPR_CLASS_ID 6
#define TWO_LINE_LPR_CLASS_ID 5
#define PROVINCE "<GuiLin>"
extern Core ie;
Recognizer::Recognizer(const std::string& inputXml, const std::string& inputBin, const std::string& inputDevice,
                       const float thresh, const float iou, const int nireq):input_xml(inputXml), thresh(thresh), iou(iou), nireq(nireq){
        try {
            // --------------------------- 1. Load Plugin for inference engine -------------------------------------
            slog::info << "Loading plugin" << slog::endl;
            ///InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(inputDevice);
            //printPluginVersion(plugin, std::cout);

            /**Loading extensions to the plugin **/

            /** Loading default extensions **/
            if (inputDevice.find("CPU") != std::string::npos) {
                /**
                 * cpu_extensions library is compiled from the "extension" folder containing
                 * custom CPU layer implementations.
                **/
                ////plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
            }
            // -----------------------------------------------------------------------------------------------------

            // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
            slog::info << "Loading network files" << slog::endl;
            cnnNetwork = ie.ReadNetwork(inputXml);
            slog::info << "Batch size is forced to  1." << slog::endl;
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- 3. Configuring input and output -----------------------------------------
            // --------------------------------- Preparing input blobs ---------------------------------------------
            /** LPR network should have 2 inputs (and second is just a stub) and one output **/
            // ---------------------------Check inputs ------------------------------------------------------
            slog::info << "Checking LPR Network inputs" << slog::endl;
            inputInfo = InputsDataMap(cnnNetwork.getInputsInfo());
#define LPR_ONE_INPUT
#ifndef LPR_ONE_INPUT
            if (inputInfo.size() != 2) {
                throw std::logic_error("LPR should have 2 inputs");
            }
            InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
            ////inputInfoFirst->setInputPrecision(Precision::U8);
            if (/*FLAGS_auto_resize*/false) {
                inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
            } else {
                inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
            }
            //auto inputName = inputInfo.begin()->first;
            auto sequenceInput = (++inputInfo.begin());
            //auto inputSeqName = sequenceInput->first;
            if (sequenceInput->second->getTensorDesc().getDims()[0] != maxSequenceSizePerPlate) {
                throw std::logic_error("LPR post-processing assumes certain maximum sequences");
            }
#else
            if (inputInfo.size() != 1) {
                throw std::logic_error("LPR should have 1 inputs");
            }
            InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
            auto inputName = inputInfo.begin()->first;
            slog::info << "inputName:" << inputName << slog::endl;
            inputInfoFirst->setPrecision(Precision::U8);
            inputShapes = cnnNetwork.getInputShapes();
            SizeVector& inSizeVector = inputShapes.begin()->second;
            inSizeVector[0] = 1;  // set batch to 1
            cnnNetwork.reshape(inputShapes);
            if (/*FLAGS_auto_resize*/false) {
                inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
            } else {
                inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
            }
#endif
            // -----------------------------------------------------------------------------------------------------


            // --------------------------------- Preparing output blobs -------------------------------------------
            slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
            outputInfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
            outputName = outputInfo.begin()->first;
            if (outputInfo.size() != 1) {
                throw std::logic_error("This net only accepts networks with one layers");
            }
            for (auto &output : outputInfo) {
                output.second->setPrecision(Precision::FP32);
                output.second->setLayout(Layout::NCHW);
            }
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- 4. Loading model to the plugin ------------------------------------------
            slog::info << "Loading recogzizer model to the plugin" << slog::endl;
            ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, inputDevice);

            // -----------------------------------------------------------------------------------------------------

            // --------------------------- 5. Creating infer request -----------------------------------------------
            for(int i=0; i<this->nireq; i++){
                this->IfReqs[i] = network.CreateInferRequestPtr();
            }
            //async_infer_request_next = network.CreateInferRequestPtr();
            //async_infer_request_curr = network.CreateInferRequestPtr();
            // -----------------------------------------------------------------------------------------------------
        }
        catch (const std::exception& error) {
            std::cerr << "[ ERROR ] " << error.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        }

}

std::string Recognizer::GetLicencePlateText(InferRequest::Ptr request) {
    slog::info << "start static defination" << slog::endl;
#define CHINESE
#ifndef CHINESE
    static std::vector<std::string> items = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
            "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
            "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
            "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
            "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
            "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
            "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
            "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
            "<Zhejiang>", "<police>",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z"
    };
#else
    static std::vector<std::string> items = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "皖", "京", "渝", "闽",
            "甘", "粤", "桂", "贵",
            "琼", "冀", "黑", "豫",
            "港", "鄂", "湘", "蒙",
            "苏", "赣", "吉", "辽",
            "澳", "宁", "青", "陕",
            "鲁", "沪", "晋", "川",
            "津", "台", "新", "云",
            "浙", "<WJ>",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z"
    };
#endif
    // up to 88 items per license plate, ended with "-1"
    slog::info << "start request->GetBlob" << slog::endl;
    const auto data = request->GetBlob(outputName)->buffer().as<float*>();
    std::string result;
    for (int i = 0; i < maxSequenceSizePerPlate; i++) {
        ///slog::info << data[i] << slog::endl;
        if (data[i] < 0)
            break;
        slog::info << items[data[i]] << slog::endl;
        result += items[data[i]];
    }

    return result;
}

void Recognizer::fillSeqBlob(InferRequest::Ptr inferRequest) {
    auto sequenceInput = (++inputInfo.begin());
    auto inputSeqName = sequenceInput->first;
    Blob::Ptr seqBlob = inferRequest->GetBlob(inputSeqName);
    int maxSequenceSizePerPlate = seqBlob->getTensorDesc().getDims()[0];
    // second input is sequence, which is some relic from the training
    // it should have the leading 0.0f and rest 1.0f
    float* blob_data = seqBlob->buffer().as<float*>();
    blob_data[0] = 0.0f;
    std::fill(blob_data + 1, blob_data + maxSequenceSizePerPlate, 1.0f);
}

bool Recognizer::FindClassObjects(std::vector<DetectionObject>& objects,
                                  std::vector<int>& target_idxes,
                                  const int target_class_id){
    bool bRet = false;
    size_t len = objects.size();
    for(size_t i=0; i<len; i++){
        if((objects[i].class_id == target_class_id) && (objects[i].confidence > this->thresh)){
            target_idxes.push_back(i);
            bRet = true;
        }
    }
    return bRet;
}

bool Recognizer::dobinaryzation(cv::Mat gray, cv::Mat& dst){
    double max, min;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(gray, &min, &max, &min_loc, &max_loc);

    float x = max - ((max-min) / 2);
    cv::threshold(gray, dst, x, 255, cv::THRESH_BINARY);
    return true;
}

void Recognizer::CaculateXYLocation(cv::Mat& edges, cv::Point& left_top, cv::Point& right_bottom){
    left_top.x = 0;
    left_top.y = 0;
    unsigned int h = edges.size().height;
    unsigned int w = edges.size().width;
    cv::Mat canny;
    edges.copyTo(canny);
    slog::info << "h:" << h << " w:" << w <<slog::endl;
    //初始化一个跟图像高一样长度的数组，用于记录每一行的黑点个数
    unsigned int statistics_h[h] = {0};
    for(unsigned int i=0; i<h; i++){
        for(unsigned int j=0; j<w; j++){
            if(edges.at<uchar>(i,j) == 0){
                statistics_h[i] += 1;
                edges.at<uchar>(i,j) = 255;//将其改为白点
            }
        }
    }
    //从该行应该变黑的最左边的点开始向最右边的点设置黑点
    for(int i=0; i<h; i++){
        for(int j=0; j<statistics_h[i]; j++){
            edges.at<uchar>(i,j) = 0;
        }
    }
    cv::Mat erosion;
    cv::Mat kernel_horizion = cv::Mat::ones(cv::Size(1,7),CV_8UC1);
    cv::erode(edges, erosion, kernel_horizion);
    memset(statistics_h, 0, sizeof(statistics_h));
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            if(edges.at<uchar>(i,j) == 0){
                statistics_h[i] += 1;
            }
        }
    }

    std::vector<int> vec_front(statistics_h, statistics_h+(h/3));
    std::vector<int> vec_back(statistics_h+(h/3*2), statistics_h+h-1);
    //ShowVec(vec_front);
    std::reverse(vec_front.begin(), vec_front.end());
    //ShowVec(vec_front);
    std::vector<int>::iterator biggest_front = std::max_element(std::begin(vec_front), std::end(vec_front));
    std::vector<int>::iterator biggest_back  = std::max_element(std::begin(vec_back), std::end(vec_back));
    unsigned int ymin_position = h/3 - int(biggest_front-std::begin(vec_front));
    unsigned int ymax_position = h/3*2 + int(biggest_back-std::begin(vec_back));

    //some refine operation
    if((w - (*biggest_front)) >= 4){
        ymin_position = 0;
    }
    if((w - *biggest_back) >= 4){
        ymax_position = h - 1;
    }

    slog::info << "ymin pos:" << ymin_position << slog::endl;
    //slog::info << "ymin val:" << *biggest_front << slog::endl;
    slog::info << "ymax pos" << ymax_position << slog::endl;
    cv::Mat close, bk;
    cv::Mat kernel = cv::Mat::ones(cv::Size(5,5),CV_8UC1);
    cv::morphologyEx(canny, close, cv::MORPH_CLOSE, kernel);
    close.copyTo(bk);
    //初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
    unsigned int statistics_w[w] = {0};
    for(unsigned int i=0; i<w; i++){
        for(unsigned int j=0; j<h; j++){
            if(close.at<uchar>(j,i) == 0){
                statistics_w[i] += 1;
                close.at<uchar>(j,i) = 255;//将其改为白点
            }
        }
    }
    //从该列应该变黑的最顶部的开始向最底部设为黑点
    for(int i=0; i<w; i++){
        for(int j=0; j<h-statistics_w[i]; j++){
            close.at<uchar>(j,i) = 0; //设为黑点
        }
    }
    cv::Mat erod;
    cv::Mat kernel_virtical = cv::Mat::ones(cv::Size(5,1),CV_8UC1);
    cv::erode(bk, erod, kernel_virtical);

    //初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
    memset(statistics_w, 0, sizeof(statistics_w));
    for(unsigned int i=0; i<w; i++){
        for(unsigned int j=0; j<h; j++){
            if(erod.at<uchar>(j,i) == 0){
                statistics_w[i] += 1;
            }
        }
    }
    std::vector<int> vec_ft(statistics_w, statistics_w+(w/4));
    std::vector<int> vec_bk(statistics_w+(w/4*3), statistics_w+w-1);
    //ShowVec(vec_ft);
    std::reverse(vec_ft.begin(), vec_ft.end());
    //ShowVec(vec_ft);
    std::vector<int>::iterator biggest_ft = std::max_element(std::begin(vec_ft), std::end(vec_ft));
    std::vector<int>::iterator biggest_bk  = std::max_element(std::begin(vec_bk), std::end(vec_bk));
    unsigned int xmin_position = w/4 - int(biggest_ft-std::begin(vec_ft));
    unsigned int xmax_position = w/4*3 + int(biggest_bk-std::begin(vec_bk));
    if((h - *biggest_ft) >= 5){
        xmax_position = w - 1;
    }
    if((h - *biggest_bk) > 5){
        xmin_position = 0;
    }
    slog::info << "xmin pos:" << xmin_position << slog::endl;
    //slog::info << "ymin val:" << *biggest_front << slog::endl;
    slog::info << "xmax pos" << xmax_position << slog::endl;
    left_top.x = xmin_position;
    left_top.y = ymin_position;
    right_bottom.x = xmax_position;
    right_bottom.y = ymax_position;
}
bool Recognizer::ReLocateLicensePlate(cv::Mat& plate){
    bool bRet = false;
    int SHRINK =100;
    cv::Mat resized, gray, binary, canny, close;
    cv::resize(plate, resized, cv::Size(SHRINK, SHRINK*plate.size().height/plate.size().width));
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    slog::info << "plate size height:" << plate.size().height << " plate size width:" << plate.size().width << slog::endl;
    dobinaryzation(gray, binary);
    cv::Canny(binary, canny, binary.size().height, binary.size().width); // TODO:how to set the thresh1 and thresh2 ?
    cv::Point left_top, right_bottom;
    CaculateXYLocation(canny, left_top, right_bottom);
    cv::Rect roi;
    roi.x = left_top.x;
    roi.y = left_top.y;
    roi.width = right_bottom.x - left_top.x;
    roi.height = right_bottom.y - left_top.y;
    slog::info << "roi.x:" << roi.x << " roi.y:" << roi.y << " roi.width:" << roi.width << " roi.height:"<< roi.height << slog::endl;
    plate = resized(roi);
    return bRet;

}
bool Recognizer::ReLocateLicensePlates(std::vector<cv::Mat>& plate_frames){
    bool bRet = false;
    for (int i=0; i<plate_frames.size(); i++){
        bRet = ReLocateLicensePlate(plate_frames[i]);
    }
    return bRet;
}

bool Recognizer::CropObjectRegion(DetectionObject& object, cv::Mat frame, cv::Mat& object_region){
    cv::Rect roi ;
    int min_height = 30;
    int width = object.xmax - object.xmin;
    int height = object.ymax - object.ymin;
    if((float(width/height) < 1.5) /*|| (object.ymax < frame.rows/2)*/){
        object.confidence = 0;
        return false;
    }
    int expand_w = std::max(width, int(min_height*2.5)) * 0.1;
    int expand_h = std::max(height, min_height) * 0.15;
    roi.x = std::max(0, object.xmin - expand_w);
    roi.y = std::max(0, object.ymin - expand_h);
    roi.width = std::min(std::max(int(min_height*2.5), object.xmax - object.xmin + 2*expand_w), frame.cols - roi.x);
    roi.height = std::min(std::max(min_height, object.ymax - object.ymin + 2*expand_h), frame.rows - roi.y);
    object.xmin = roi.x;
    object.ymin = roi.y;
    object.xmax = roi.x + roi.width -1;
    object.ymax = roi.y + roi.height -1;
    std::cout << "roi.x:" << roi.x << " roi.y:" << roi.y << " roi.width:" << roi.width << " roi.height:"<< roi.height << std::endl;
    object_region = frame(roi);
    return true;
}

bool Recognizer::GetTargetFrames(cv::Mat frame, std::vector<DetectionObject>& objects, std::vector<int>& plate_idxes, std::vector<cv::Mat>& plate_frames){
    bool bRet = false;
    for(int i=0; i<plate_idxes.size(); i++){
        cv::Mat region;        
        this->CropObjectRegion(objects[plate_idxes[i]], frame, region);
        plate_frames.push_back(region);
        //ReLocateLicensePlates(plate_frames);
        bRet=true;
    }
    return bRet;
}

void Recognizer::FakeProvinceFeild(std::string& lpr_txt){
    int iPos1 = lpr_txt.find('<');
    int iPos2 = lpr_txt.find('>');
    //std::string str_target = lpr_txt.substr(iPos1, iPos2-iPos1 + 1);
    if((iPos1 != lpr_txt.npos) && (iPos2 != lpr_txt.npos)){
        lpr_txt.replace(iPos1, iPos2-iPos1+1, PROVINCE);
    }
    else{
        lpr_txt.insert(0, PROVINCE);
    }
    //lpr_txt.replace(lpr_txt.find("<police>"), strlen("<police>"), "");
}

int Recognizer::Recognize(int idx, const cv::Mat coresponding_frame, std::vector<DetectionObject>& objects){
    std::vector<int> plate_idxes;
    //slog::info << "into lpr recognize " << slog::endl;
    //int class_plate_id = 6;
    std::vector<cv::Mat> plate_frames;
    if(this->FindClassObjects(objects, plate_idxes, ONE_LINE_LPR_CLASS_ID)){
        this->GetTargetFrames(coresponding_frame, objects, plate_idxes, plate_frames);
        try {
            for(int i=0; i<plate_frames.size(); i++){
                //slog::info << "Start lpr inference " << slog::endl;
                cv::Mat frame = plate_frames[i];
                if(frame.size().width <= 0 || frame.size().height <=0){
                    continue;
                }
                /*cv::Mat temp;
                ChangeMotorLPR2VeichleLPR(frame, temp);
                frame = temp;*/
                //cv::imwrite("one_line_lpr.jpg", frame);
                // --------------------------- 6. Doing inference ------------------------------------------------------
                slog::info << "Start lpr inference " << slog::endl;
                typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
                auto t0 = std::chrono::high_resolution_clock::now();
                auto inputName = inputInfo.begin()->first;
                //Main sync point:
                slog::info << "inputName:" << inputName << slog::endl;
                FrameToBlob(frame, IfReqs[idx], inputName);
#ifndef LPR_ONE_INPUT
                this->fillSeqBlob(IfReqs[idx]);
#endif
                IfReqs[idx]->StartAsync();
                if (OK == IfReqs[idx]->Wait(IInferRequest::WaitMode::RESULT_READY)){
                    std::string str_plate = GetLicencePlateText(IfReqs[idx]);
                    //FakeProvinceFeild(str_plate);
                    slog::info << "plate: " << str_plate << slog::endl;
                    ///cv::imwrite("./lpr_result_dir/"+str_plate+".jpg", frame);
                    objects[plate_idxes[i]].set_text(str_plate);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                ms recognization = std::chrono::duration_cast<ms>(t1 - t0);
                slog::info << "recognization duration : " << recognization.count() << slog::endl;
            }

        }
        catch (const std::exception& error) {
            std::cerr << "[ ERROR ] " << error.what() << std::endl;
            return -100;
        }
        catch (...) {
            std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
            return -100;
        }
        //slog::info << "Execution successful" << slog::endl;
    }

    return 0;
}
