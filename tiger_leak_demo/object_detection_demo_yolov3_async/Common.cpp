#include "Common.h"

int indexsort_comparator(const void *pa, const void *pb)
{
    float proba = ((indexsort *)pa)->prob[((indexsort *)pa)->index * ((indexsort *)pa)->channel + ((indexsort *)pa)->iclass];
    float probb = ((indexsort *)pb)->prob[((indexsort *)pb)->index * ((indexsort *)pb)->channel + ((indexsort *)pb)->iclass];

    float diff = proba - probb;
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float logistic_activate(float x)
{
    return 1./(1. + exp(-x));
}

void transpose(float *src, float* tar, int k, int n)
{
    int i, j, p;
    float *tmp = tar;
    for(i = 0; i < n; ++i)
    {
        for(j = 0, p = i; j < k; ++j, p += n)
        {
            *(tmp++) = src[p];
        }
    }
}
void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(ibox a, ibox b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(ibox a, ibox b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(ibox a, ibox b)
{
    return box_intersection(a, b)/box_union(a, b);
}
int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}


void ShowVec(const std::vector<int>& valList)
{
    std::for_each(valList.cbegin(), valList.cend(), [](const int& val)->void{std::cout << val << std::endl; });
}



void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (/*FLAGS_auto_resize*/false) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        std::cout << "FrameToBlob before inferRequest->GetBlob" << std::endl;
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        std::cout << "FrameToBlob before matU8ToBlob" << std::endl;
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

static int EntryIndex(int side_h, int side_w, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side_h * side_w);
    int loc = location % (side_h * side_w);
    return n * side_h * side_w * (lcoords + lclasses + 1) + entry * side_h * side_w + loc;
}

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
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

void ParseYOLOV3Output(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------
    auto num = layer->GetParamAsInt("num");
    try { num = layer->GetParamAsInts("mask").size(); } catch (...) {}
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    int anchor_offset = 0;
    switch (side) {
        case yolo_scale_13:
            anchor_offset = 2 * 6;
            break;
        case yolo_scale_26:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_52:
            anchor_offset = 2 * 0;
            break;
        case yolo_scale_19:
            anchor_offset = 2 * 6;
            break;
        case yolo_scale_38:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_76:
            anchor_offset = 2 * 0;
            break;
        case yolo_scale_27:
            anchor_offset = 2 * 6;
            break;
        case yolo_scale_54:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_108:
            anchor_offset = 2 * 0;
            break;
        default:
            throw std::runtime_error("Invalid output size");
    }
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void ParseSSDNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects){

    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    /*std::cout << "[INFO] dims:"
              << blob->dims().at(0)
              << blob->dims().at(1)
              << blob->dims().at(2)
              << std::endl;*/
    for(int i=0; i<100; i++){
        if(output_blob[i*7 + 2] < threshold){
            continue;
        }
        else{
            double xmin = (output_blob[i*7 + 3] * original_im_w);
            double ymin = (output_blob[i*7 + 4] * original_im_h);
            double xmax = (output_blob[i*7 + 5] * original_im_w);
            double ymax = (output_blob[i*7 + 6] * original_im_h);
            float confidence = output_blob[i*7 + 2];
            int class_id = (output_blob[i*7 + 1]);
            DetectionObject obj(xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2, ymax-ymin, xmax-xmin, class_id, confidence, 1.0, 1.0);
            objects.push_back(obj);
        }
    }
}

void ParseYOLOV3TinyNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    /*if (layer->type != "RegionYolo")
        throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");*/
    const int out_blob_c = static_cast<int>(blob->getTensorDesc().getDims()[1]);
    //std::cout << "[ INFO ] Output blob chanel = " << out_blob_c << std::endl;
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------
    /*auto num = layer->GetParamAsInt("num");
    try { num = layer->GetParamAsInts("mask").size(); } catch (...) {}
    auto coords = layer->GetParamAsInt("coords");
    auto classes = layer->GetParamAsInt("classes");
    */
    int num = 3;
    int coords = 4;
    int classes = out_blob_c/num - coords - 1;
    //std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
    //                             156.0, 198.0, 373.0, 326.0};
    std::vector<float> anchors = {10,25,  20,50,  30,75, 50,125,  80,200,  150,150};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    int anchor_offset = 0;
    switch (side) {
        case yolo_scale_13:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_26:
            anchor_offset = 2 * 0;
            break;
        case yolo_scale_19:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_38:
            anchor_offset = 2 * 0;
            break;
        case yolo_scale_27:
            anchor_offset = 2 * 3;
            break;
        case yolo_scale_54:
            anchor_offset = 2 * 0;
            break;
        default:
            throw std::runtime_error("Invalid output size");
    }
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            #if 0
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];

            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
            #else
            float scale = logistic_activate(output_blob[obj_index]);
            if (scale < threshold)
                continue;
            double x = (col + logistic_activate(output_blob[box_index + 0 * side_square])) / side * resized_im_w;
            double y = (row + logistic_activate(output_blob[box_index + 1 * side_square])) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];

            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * logistic_activate(output_blob[class_index]);
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
            #endif
        }
    }
}

void ParseYOLOV3TinyNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, const unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    const int out_blob_c = static_cast<int>(blob->getTensorDesc().getDims()[1]);
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    int num = 3;
    int coords = 4;
    int classes = out_blob_c/num - coords - 1;
    //std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
    //                             156.0, 198.0, 373.0, 326.0};
    std::vector<float> anchors = {10,25,  20,50,  30,75, 50,125,  80,200,  150,150};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    int anchor_offset = 0;

    anchor_offset = 2*(1-layer_order_id)*3;
    //std::cout << "[ INFO ] layer_order_id = " << layer_order_id << std::endl;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
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
            //height = std::min(height, resized_im_h - y);
            //width = std::min(width, resized_im_w - x);

            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * logistic_activate(output_blob[class_index]);
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }

        }
    }
}

void ParseYOLOV3TinyNcsOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, const unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objects, const unsigned int id_offset) {
    // --------------------------- Validating output parameters -------------------------------------
    const int out_blob_c = static_cast<int>(blob->getTensorDesc().getDims()[1]);
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    // --------------------------- Extracting layer parameters -------------------------------------

    int num = 3;
    int coords = 4;
    int classes = out_blob_c/num - coords - 1;
    std::vector<float> anchors = {10,25,  20,50,  30,75, 50,125,  80,200,  150,150};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    int anchor_offset = 0;
    anchor_offset = 2*(1-layer_order_id)*3;

    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
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


void ParseYOLOV3TinyNcsOutputHW(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, const unsigned long layer_order_id,
                       const double threshold, std::vector<DetectionObject> &objects) {
    // --------------------------- Validating output parameters -------------------------------------
    const int out_blob_c = static_cast<int>(blob->getTensorDesc().getDims()[1]);
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);

    int num = 3;
    int coords = 4;
    int classes = out_blob_c/num - coords - 1;
    //std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
    //                             156.0, 198.0, 373.0, 326.0};
    std::vector<float> anchors = {10,25,  20,50,  30,75, 50,125,  80,200,  150,150};
    try { anchors = layer->GetParamAsFloats("anchors"); } catch (...) {}
    auto side = out_blob_h;
    auto side_h = out_blob_h;
    auto side_w = out_blob_w;
    int anchor_offset = 0;

    anchor_offset = 2*(1-layer_order_id)*3;
    //std::cout << "[ INFO ] layer_order_id = " << layer_order_id << std::endl;
    auto side_square = side_h * side_w; //auto side_square = side * side;

    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side_w; //side;
        int col = i % side_w; //side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side_h, side_w, coords, classes, n * side_h * side_w + i, coords);
            int box_index = EntryIndex(side_h, side_w, coords, classes, n * side_h * side_w + i, 0);

            float scale = logistic_activate(output_blob[obj_index]);
            if (scale < threshold)
                continue;
            double x = (col + logistic_activate(output_blob[box_index + 0 * side_square])) / side_w * resized_im_w;
            double y = (row + logistic_activate(output_blob[box_index + 1 * side_square])) / side_h * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n];

            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side_h, side_w, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * logistic_activate(output_blob[class_index]);
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }

        }
    }
}

void ParseYOLOV5SOutput(const CNNLayerPtr &layer, const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w, unsigned long layer_order_id,
                        const double threshold, std::vector<DetectionObject> &objects){
    ///std::cout << "thresh in parse yolov5:" << threshold << std::endl;
    // --------------------------- Validating output parameters -------------------------------------
    const int out_blob_c = static_cast<int>(blob->getTensorDesc().getDims()[1]);
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + layer->name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));
    int num = 3;
    int coords = 4;
    int classes = out_blob_c/num - coords - 1;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                 156.0, 198.0, 373.0, 326.0};
    auto side = out_blob_h;
    int anchor_offset = 0;
    switch (resized_im_h/side) {
    case 8:
        layer_order_id = 0;
        break;
    case 16:
        layer_order_id = 1;
        break;
    case 32:
        layer_order_id = 2;
        break;
    default:
        break;
    }
    anchor_offset = 2 * 3 * layer_order_id;
    //std::cout << "[ INFO ] layer_order_id = " << layer_order_id << std::endl;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
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
            double x = (col + 2*logistic_activate(output_blob[box_index + 0 * side_square]) - 0.5) * resized_im_w / side;
            double y = (row + 2*logistic_activate(output_blob[box_index + 1 * side_square]) - 0.5) * resized_im_h / side;
            double height = std::pow(2*(logistic_activate(output_blob[box_index + 3 * side_square])), 2) * anchors[anchor_offset + 2 * n + 1];
            double width = std::pow(2*(logistic_activate(output_blob[box_index + 2 * side_square])), 2) * anchors[anchor_offset + 2 * n];

            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * logistic_activate(output_blob[class_index]);
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void ChangeMotorLPR2VeichleLPR(cv::Mat& motor_lpr, cv::Mat& veichle_lpr){
    int h = motor_lpr.rows;
    int w = motor_lpr.cols;
    cv::Rect rect_top_middle = cv::Rect(w/5, 0, 3*w/5, h/2);
    cv::Rect rect_bottom = cv::Rect(0, h/2, w, h/2);
    cv::hconcat(motor_lpr(rect_top_middle), motor_lpr(rect_bottom), veichle_lpr);
    //veichle_lpr = motor_lpr(rect_bottom);
}

int cvPutChineseTextTest(){
    /*
    cv::Mat img=cv::imread("/home/图片/Plates/abc.jpg");

    string text="这次肯定能Put上中文！";

    int fontHeight=60;
    int thickness=-1;
    int linestyle=8;
    int baseline=0;

    cv::Ptr<cv::freetype::FreeType2> ft2;
    ft2=cv::freetype::createFreeType2();
    ft2->loadFontData("/usr/share/fonts/winFonts/simkai.ttf",0);

    cv::Size textSize=ft2->getTextSize(text,fontHeight,thickness,&baseline);

    if (thickness>0) baseline+=thickness;

    // center the text
    cv::Point textOrg((img.cols - textSize.width) / 2,
              (img.rows + textSize.height) / 2);
    // draw the box
    cv::rectangle(img, textOrg + cv::Point(0, baseline),
          textOrg + cv::Point(textSize.width, -textSize.height),
          cv::Scalar(0,255,0),1,8);
    // ... and the baseline first
    cv::line(img, textOrg + cv::Point(0, thickness),
     textOrg + cv::Point(textSize.width, thickness),
     cv::Scalar(0, 0, 255),1,8);
    // then put the text itself
    ft2->putText(img, text, textOrg, fontHeight,
             cv::Scalar(255,0,0), thickness, linestyle, true );

    cv::imshow("效果",img);
    if (cv::waitKey(0)==27)*/ return 0;
}
