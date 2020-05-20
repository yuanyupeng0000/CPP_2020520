CPP     = g++ -std=c++11
RM      = rm -rf
## debug flag
DBG_ENABLE   = 1
## source file path
#SRC_PATH   := $(PWD)
#SRC_PATH=$(PWD)
SRC_PATH= src

## target exec file name
TARGET     := libPlateRecognize.so

## get all source files
SRCS         += $(wildcard $(SRC_PATH)/*.cpp)

## all .o based on all .c
OBJS        := $(SRCS:.cpp=.o)

## need libs, add at here
LIBS :=

## used headers  file path
INCLUDE_PATH += /usr/include/python3.5/
INCLUDE_PATH += include
INCLUDE_PATH += $(SRC_PATH)
INCLUDE_PATH += ../../include #/usr/local/include
## used include librarys file path
LIBRARY_PATH := /lib


## debug for debug info, when use gdb to debug
ifeq (1, ${DBG_ENABLE})
	CFLAGS += -D_DEBUG -O0 -g -DDEBUG=1
endif

## get all include path
CFLAGS  += $(foreach dir, $(INCLUDE_PATH), -I$(dir))  -Wall -g 
CFLAGS += -fopenmp 

## get all library path
LDFLAGS += $(foreach lib, $(LIBRARY_PATH), -L$(lib)) -g

## get all librarys
LDFLAGS += $(foreach lib, $(LIBS), -l$(lib))
LDFLAGS += -lgomp -lpthread 
LDFLAGS += -L../../lib/opencv3.3 -lopencv_core -lopencv_ml -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_dnn
#LDFLAGS += `pkg-config --libs opencv`


all:build
build:
	$(CPP) -fPIC -c $(CFLAGS) $(SRCS) 
	cp *.o src
	$(CPP) -fPIC -shared -o $(TARGET) $(OBJS) $(LDFLAGS)
.PHONY:clean
clean:
	$(RM) $(OBJS) $(TARGET) ./*.o
