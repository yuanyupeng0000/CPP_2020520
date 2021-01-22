// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov3_async/main.cpp
* \example object_detection_demo_yolov3_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include "Common.h"
#include "object_detection_demo_yolov3_async.hpp"
//#include <ext_list.hpp>
#include "detector.h"
#include "recognizer.h"
#include "intel_dldt.h"
using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty() && FLAGS_j.empty()) {
        throw std::logic_error("Parameter -i and -j is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

int main(int argc, char *argv[]){
    /** This demo covers a certain topology and cannot be generalized for any object detection **/
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // ------------------------------ Parsing and validating the input arguments ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }
    slog::info << "Reading input" << slog::endl;
    cv::VideoCapture cap;
    // read input (video) frame
    cv::Mat frame;
    cv::Mat next_frame;
    if(FLAGS_j == ""){
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        cap >> frame;
    }

    //const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    /*if (!cap.grab()) {
        throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                               "Failed to get next frame from the " + FLAGS_i);
    }*/
    // -----------------------------------------------------------------------------------------------------
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    ///Detect
    /// or detector(FLAGS_m, binFileName, FLAGS_d, FLAGS_t, FLAGS_iou_t, FLAGS_nireq);
    ///Recognizer Recognizer()
    intel_dldt_init("FP16/vpu_config.ini");
    ///
    //slog::info << "Start inference " << slog::endl;

    bool isLastFrame = false;
    int N = 0;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::queue<cv::Mat> frame_que;
    cv::VideoWriter writer(FLAGS_s, CV_FOURCC('X', 'V', 'I', 'D'), 15, cv::Size(1280, 960));
    while (true) {
        N += 1;

        // Here is the first asynchronous point:
        // in the Async mode, we capture frame to populate the NEXT infer request
        // in the regular mode, we capture frame to the CURRENT infer request
        if(FLAGS_j != ""){
            frame = cv::imread(FLAGS_j);
        }
        else{
            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }
        }

        //detector.Detect(frame);
        frame_que.push(frame);
        //std::cout << "[ INFO ] Qeue size:" << frame_que.size() << std::endl;
        std::vector<DetectionObject> objects;

        int ret_code = intel_dldt_detect(frame, 1, objects);
        ///int ret_code = detector.Detect(frame, objects);
        std::cout << "[ INFO ] ret_code " << ret_code << std::endl;
        if(ret_code < 0 && ret_code != -100){
            continue;
        }

        //std::cout << "[ INFO ] nboxes = " << objects.size() << std::endl;
        // Drawing boxes
        cv::Mat frame_show = frame_que.front();
        for (auto &object : objects) {
            if (object.confidence < FLAGS_t/*detector.thresh*/){
                //std::cout << "[INFO]: confidence = " << object.confidence << std::endl;
                continue;
            }
            auto label = object.class_id;
            float confidence = object.confidence;
            if (true) {
                std::cout << "[" << label << "] element, prob = " << confidence <<
                          "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                          << ((confidence > FLAGS_t/*detector.thresh*/) ? " WILL BE RENDERED!" : "") << std::endl;
            }
            if (confidence > FLAGS_t/*detector.thresh*/) {
                /** Drawing only objects when >confidence_threshold probability **/
                std::ostringstream conf;
                conf << ":" << std::fixed << std::setprecision(3) << confidence;
                cv::putText(frame_show,
                        (object.class_id == 6 ? object.text: /*label < detector.labels.size() ? detector.labels[label] : */
                                                std::string("#") + std::to_string(label)) /*+ conf.str()*/,
                            cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                            cv::Scalar(0, 0, 255));
                cv::rectangle(frame_show, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(0, 0, 255), 2);
            }
        }
        cv::Mat dst;
        cv::resize(frame_show, dst, cv::Size(640, 480));
        cv::imshow("Detection results", dst);
        if(!FLAGS_s.empty()){
            writer.write(dst);
        }
        //cv::imwrite("saved_result.jpg", frame_show);
        frame_que.pop();
        auto t1 = std::chrono::high_resolution_clock::now();
        ms detection = std::chrono::duration_cast<ms>(t1 - t0);
        slog::info << "Duration fps:" << 1000*N/detection.count() << slog::endl;
        frame = next_frame;
        next_frame = cv::Mat();
        const int key = cv::waitKey(2);
        if (27 == key)  // Esc
            break;
        if(FLAGS_j != ""){
            const int key = cv::waitKey(5000);
            if (27 == key)  // Esc
                break;
        }
    }
    //writer.release();
    slog::info << "exit." << slog::endl;
}
/*
int main_bak(int argc, char *argv[]) {
    try {
        //This demo covers a certain topology and cannot be generalized for any object detection
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

        // read input (video) frame
        cv::Mat frame;  cap >> frame;
        cv::Mat next_frame;

        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed to get next frame from the " + FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({"/data/github_repos/yolov3-tiny-fit-ncs/ncs2/OpenVINO/inference_engine/lib/ubuntu_16.04/intel64", ""}).getPluginByDevice(FLAGS_d);
        //printPluginVersion(plugin, std::cout);

        //Loading extensions to the plugin

        //Loading default extensions
        if (FLAGS_d.find("CPU") != std::string::npos) {

             //* cpu_extensions library is compiled from the "extension" folder containing
             //* custom CPU layer implementations.

            ////plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            plugin.AddExtension(extension_ptr);
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        }

        //Per-layer metrics
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        //Reading network model
        netReader.ReadNetwork(FLAGS_m);
        //etting batch size to 1
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        //Extracting the model name and loading its weights
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        //Reading labels (if specified)
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        //YOLOV3-based network should have one input and three output
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }
        // --------------------------------- Preparing output blobs -------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 3) {
            throw std::logic_error("This demo only accepts networks with three layers");
        }
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        ExecutableNetwork network = plugin.LoadNetwork(netReader.getNetwork(), {});

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Creating infer request -----------------------------------------------
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_decode_time = 0, ocv_render_time = 0;
        //float FLAGS_t = 0.3;

        while (true) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the Async mode, we capture frame to populate the NEXT infer request
            // in the regular mode, we capture frame to the CURRENT infer request
            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }
            if (isAsyncMode) {
                if (isModeChanged) {
                    FrameToBlob(frame, async_infer_request_curr, inputName);
                }
                if (!isLastFrame) {
                    FrameToBlob(next_frame, async_infer_request_next, inputName);
                }
            } else if (!isModeChanged) {
                FrameToBlob(frame, async_infer_request_curr, inputName);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
            // in the regular mode, we start the CURRENT request and wait for its completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();
                std::ostringstream out;
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (ocv_decode_time + ocv_render_time) << " ms";
                cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                out.str("");
                out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                cv::putText(frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
                if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                    out.str("");
                    out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                                cv::Scalar(255, 0, 0));
                }

                // ---------------------------Processing output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                unsigned long resized_im_h = inputInfo.begin()->second.get()->getDims()[0];
                unsigned long resized_im_w = inputInfo.begin()->second.get()->getDims()[1];
                std::vector<DetectionObject> objects;
                // Parsing outputs
                for (auto &output : outputInfo) {
                    auto output_name = output.first;
                    CNNLayerPtr layer = netReader.getNetwork().getLayerByName(output_name.c_str());
                    Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                    ParseYOLOV3Output(layer, blob, resized_im_h, resized_im_w, height, width, FLAGS_t, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end());
                for (int i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (int j = i + 1; j < objects.size(); ++j)
                        if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                            objects[j].confidence = 0;
                }
                // Drawing boxes
                for (auto &object : objects) {
                    if (object.confidence < FLAGS_t)
                        continue;
                    auto label = object.class_id;
                    float confidence = object.confidence;
                    if (true) {
                        std::cout << "[" << label << "] element, prob = " << confidence <<
                                  "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }
                    if (confidence > FLAGS_t) {
                        //Drawing only objects when >confidence_threshold probability
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frame,
                                (label < labels.size() ? labels[label] : std::string("label #") + std::to_string(label))
                                    + conf.str(),
                                    cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    cv::Scalar(0, 0, 255));
                        cv::rectangle(frame, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(0, 0, 255));
                    }
                }
            }
            cv::imshow("Detection results", frame);

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }


            // Final point:
            // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
            frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            }
        }
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;

        //Showing performace results
        if (FLAGS_pc) {
            //printPerformanceCounts(*async_infer_request_curr, std::cout);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}*/
