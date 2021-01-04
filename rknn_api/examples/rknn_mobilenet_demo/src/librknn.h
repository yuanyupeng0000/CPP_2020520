#ifndef LIBRKNN_H
#define LIBRKNN_H
#include <iostream>

//Detection object
struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;
    std::string text;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = std::max(0, static_cast<int>((x - w / 2) * w_scale));
        this->ymin = std::max(0, static_cast<int>((y - h / 2) * h_scale));
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
        //std::cout << "xmin:" << xmin << " ymin:" << ymin << " xmax:" << xmax << " ymax:"<< ymax << std::endl;

    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
    void set_text(std::string& text){
        this->text = text;
    }
};

int librknn_ai_init(const std::string& config);
int librknn_ai_detect(const unsigned char* pucframe, std::vector<DetectionObject>& objs);

void rknn_ai_alg(void *p, std::vector<DetectionObject>& objs);


#endif //LIBRKNN_H
