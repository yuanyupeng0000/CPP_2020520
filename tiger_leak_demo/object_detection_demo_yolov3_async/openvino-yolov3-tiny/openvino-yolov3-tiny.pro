DISTFILES += \
    ../README.md \
    ../CMakeLists.txt
HEADERS += \
    ../Common.h \
    ../object_detection_demo_yolov3_async.hpp \
    ../detector.h \
    ../intel_dldt.h \
    ../recognizer.h

SOURCES += \
    ../Common.cpp \
    ../main.cpp \
    ../detector.cpp \
    ../intel_dldt.cpp \
    ../main.cpp \
    ../recognizer.cpp
INCLUDEPATH += /opt/intel/openvino/deployment_tools/inference_engine/include \
    /opt/intel/openvino/deployment_tools/inference_engine/samples/cpp/common/ \
    /opt/intel/openvino/deployment_tools/inference_engine/src/extension \
    /opt/intel/openvino/inference_engine/include/
    ../
