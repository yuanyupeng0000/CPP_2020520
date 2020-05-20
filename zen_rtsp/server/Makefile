CFLAGS=-g -Wall
CC=gcc
LDFLAGS=-lpthread
## used headers  file path
INCLUDE_PATH := ../include 
## used include librarys file path
LIBRARY_PATH :=

## debug for debug info, when use gdb to debug
ifeq (1, ${DBG_ENABLE})
	CFLAGS += -D_DEBUG -O0 -g -DDEBUG=1
endif

## get all include path
CFLAGS  += $(foreach dir, $(INCLUDE_PATH), -I$(dir))

## get all library path
LDFLAGS += $(foreach lib, $(LIBRARY_PATH), -L$(lib))
TARGET=ipsearch
SRC_PATH   := .
SRCS         += $(wildcard $(SRC_PATH)/*.c)
OBJS        := $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
.PHONY:clean
clean:
	rm -f *.o *.a  $(TARGET)
	
