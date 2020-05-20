TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

HEADERS += \
    ../../cam_alg.h \
    ../../cam_codec.h \
    ../../cam_net.h \
    ../../camera_service.h \
    ../../client_file.h \
    ../../client_net.h \
    ../../client_obj.h \
    ../../common.h \
    ../../csocket.h \
    ../../file_op.h \
    ../../g_define.h \
    ../../glplayer.h \
    ../../h264_stream_file.h \
    ../../hw_vaapi.h \
    ../../IPCNetSDK.h \
    ../../Net_param.h \
    ../../Net.h \
    ../../sig_service.h \
    ../../vaapi.h \
    ../../tcp_server.h

SOURCES += \
    ../../cam_alg.c \
    ../../cam_codec.c \
    ../../cam_net.c \
    ../../camera_service.c \
    ../../client_config_main.c \
    ../../client_file.c \
    ../../client_net.c \
    ../../client_obj.c \
    ../../common.c \
    ../../csocket.c \
    ../../file_op.c \
    ../../glplayer.c \
    ../../h264_stream_file.c \
    ../../hw_vaapi.c \
    ../../main.c \
    ../../sig_service.c \
    ../../tcp_server.c
