# this is a makefile for caffe  
TARGET := classify  
SLIB := libclassify.so
# caffe directory  
CAFFE_DIR := /opt/caffe
# caffe include  
CAFFE_INCLUDE := $(CAFFE_DIR)/include
# caffe build   
CAFFE_BUILD := $(CAFFE_DIR)/build
# caffe build lib  
CAFFE_BUILD_LIB := $(CAFFE_BUILD)/lib  
# caffe build src  
CAFFE_BUILD_SRC := $(CAFFE_BUILD)/src  
#
CUDA_INCLUDE := /usr/local/cuda-8.0/include

# dependency libraries and caffe linking libs  
LFLAGS := -pthread -lnsl -lX11  
LFLAGS += -lcaffe -lglog -lgflags -lopencv_flann -lprotobuf -lboost_system -lboost_filesystem  -lboost_thread -lboost_coroutine -lboost_context -lboost_regex -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb  -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_videoio -lstdc++ -lcblas -latlas -lcuda -lcudart -lcublas -lcurand -lcudnn

LFLAGS += -Wl,-rpath=$(CAFFE_BUILD_LIB) -L/usr/lib/x86_64-linux-gnu/hdf5/serial
LFLAGS += -L/root/source/opencv-3.2.0/build/__install/lib/ -lopencv_core -lopencv_imgcodecs -lopencv_highgui
LFLAGS += -L/usr/local/cuda-8.0/lib64

# rules  
CFLAGS := -g -I . -I $(CUDA_INCLUDE) -I $(CAFFE_INCLUDE) -I $(CAFFE_DIR)/src -I $(CAFFE_BUILD_SRC) -I/usr/include/hdf5/serial -I ./ -Wall -DUSE_OPENCV  
CC := g++  

SRC +=  attribute_recognition.cpp 
INCLUDE += /
## all .o based on all .c
OBJS        := $(SRC:.cpp=.o)

#$(TARGET):$(SRC)  
#	$(CC) -o $(TARGET) -O0 -std=c++11 $(SRC) $(CFLAGS) $(LFLAGS) -L$(CAFFE_BUILD_LIB) -Wno-sign-compare  
$(SLIB):$(SRC)  
	$(CC) -shared -fpic -o $(SLIB) -O0 -std=c++11 -I $(INCLUDE) $(SRC) $(CFLAGS) $(LFLAGS) -L$(CAFFE_BUILD_LIB) -Wno-sign-compare 

#all:build
#build:
#	$(CC) -o $(TARGET) -O0 -std=c++11 $(SRC) $(CFLAGS) $(LFLAGS) -L$(CAFFE_BUILD_LIB) -Wno-sign-compare
#	$(CC) -fPIC -shared -o $(TARGET) $(OBJS)
#.PHONY:clean

clean :  
	@rm -f $(SLIB) $(TARGET) $(OBJS)     
