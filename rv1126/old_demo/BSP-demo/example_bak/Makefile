hide := @
ECHO := echo

GCC :=/rv1126/rv1126_rv1109/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc-8.3.0
CFLAGS := -I../include \
			-I../include/rkmedia \
			-I../include/rkaiq/common \
			-I../include/rkaiq/xcore \
			-I../include/rkaiq/uAPI \
			-I../include/rkaiq/algos \
			-I ../include/rkaiq/iq_parser

LIB_FILES := -L../lib

LD_FLAGS := -lpthread -leasymedia -ldrm -lrockchip_mpp \
	        -lavformat -lavcodec -lswresample -lavutil \
			-lasound -lv4l2 -lv4lconvert -lrga \
			-lRKAP_ANR -lRKAP_Common -lRKAP_AEC \
			-lmd_share -lrkaiq -lod_share

CFLAGS += -DRKAIQ

SAMPLE_COMMON := common/sample_common_isp.c
all:
	$(GCC) rk_sample.c $(SAMPLE_COMMON) $(CFLAGS) $(LIB_FILES) $(LD_FLAGS) -o rk_sample
	$(hide)$(ECHO) "Build Done ..."

