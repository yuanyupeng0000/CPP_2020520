CPP     = g++ -std=c++11
RM      = rm -rf
## debug flag
DBG_ENABLE   = 1
DUSE_GPU = 0
DDETECT_ATTRI = 0
DDETECT_PLATE = 0
DUSE_PYTHON = 0
## source file path
#SRC_PATH   := $(PWD)
#SRC_PATH=$(PWD)
SRC_PATH=./

## target exec file name
TARGET     := libzenalg.so

## get all source files
SRCS         += $(wildcard $(SRC_PATH)/*.cpp)

## all .o based on all .c
OBJS        := $(SRCS:.cpp=.o)

## need libs, add at here
LIBS :=

## used headers  file path
INCLUDE_PATH += ../include
INCLUDE_PATH += /usr/include/python3.5/
INCLUDE_PATH += $(SRC_PATH)/include
INCLUDE_PATH += $(SRC_PATH)
INCLUDE_PATH += /usr/local/include
## used include librarys file path
LIBRARY_PATH := ../lib/opencv3.3 /lib
LDFLAGS += -L/usr/local/lib -lopencv_core -lopencv_ml -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_dnn
LDFLAGS +=     \
    -Wl,--rpath=./ \
	-Wl,--rpath=./lpr/ 
ifeq (1, ${DUSE_GPU})
	INCLUDE_PATH += /usr/local/cuda/include/
	LIBRARY_PATH += /usr/local/cuda/lib64 
	LDFLAGS += -L./ -ldarknet -lcuda -lcudart -lcublas -lcurand
	CFLAGS += -DGPU -DOPENCV -DDETECT_GPU
endif
ifeq (1, ${DDETECT_PLATE})
	CFLAGS += -DDETECT_PLATE
	LDFLAGS += -L./lpr/ -lPlateRecognize
endif
## debug for debug info, when use gdb to debug
ifeq (1, ${DBG_ENABLE})
	CFLAGS += -D_DEBUG -O0 -g -DDEBUG=1
endif

## get all include path
CFLAGS  += $(foreach dir, $(INCLUDE_PATH), -I$(dir))  -g
CFLAGS += -fopenmp 
ifeq (1, ${DUSE_PYTHON})
	CFLAGS += -DUSE_PYTHON
endif
## get all library path
LDFLAGS += $(foreach lib, $(LIBRARY_PATH), -L$(lib)) -g 

## get all librarys
LDFLAGS += $(foreach lib, $(LIBS), -l$(lib))
LDFLAGS += -lgomp -lpthread 


all:build
build:
	$(CPP) -fPIC -c $(CFLAGS) $(SRCS) $(LDFLAGS) -g
	$(CPP) -fPIC -shared -o $(TARGET) $(OBJS) $(LDFLAGS)
.PHONY:clean
clean:
	$(RM) $(OBJS) $(TARGET)
