SUBDIR=.
SOURCE=$(foreach n, $(SUBDIR),$(wildcard $(n)/*.c))

INCDIR=include
INCLUDE_FILE=$(foreach n, $(INCDIR),$(wildcard $(n)/*.h))
DBG_ENABLE=1
DUSE_GPU = 0
DDETECT_PLATE = 0
DDUSE_PYTHON = 0
ORACLE_VS = 0
#exclude_files:=client_config_main.c
#SOURCE:=$(filter-out $(exclude_files),$(SOURCE))
SOURCE:= cam_alg.c         cam_net.c             common.c             sig_service.c\
cam_codec.c       client_obj.c              csocket.c  h264_stream_file.c  tcp_server.c \
camera_service.c    file_op.c  main.c client_file.c client_net.c udp_network.c \
./ini/inifile.c ./ini/fvdconfig.c ./fvdmysql/fvdmysql.c ./camera/camera_rtsp.c g_fun.c ./websocket/ws_service.c \
./queue/queue.c cvxtext.c cam_event.c url_downfile.c ptserver/non_motor_server.c

ifeq (1, ${ORACLE_VS})
	SOURCE +=  ./fvdmysql/fvdoracle.c
endif

OBJS:=$(SOURCE:%.c=%.o)

#CFLGS +=-I./
#LFLGS +=-L/usr/lib -lpthread
#-DUSE_FILE -DPLAY_BACK
#exclude_dirs=bin lib include config rtp server $(ALG_SRC)
#dirs:=$(shell find . -maxdepth 1 -type d)
#dirs:=$(basename $(patsubst ./%,%,$(dirs)))
#dirs:=$(filter-out $(exclude_dirs),$(dirs))
CFLGS +=-I./include -I./     #  -DHW_DECODE       #  -DUSE_FILE  -DPLAY_BACK
#CFLGS +=-I/usr/local/include
CFLGS += -I/fvd_extra_include  -I./alg-mutilthread/ -I./include/libssh2 -I./ptserver
CFLGS += -I./include/fvd_extra_include/GL
CFLGS += -I./include/fvd_extra_include/
CFLGS += -I./ini  -I./camera 
CFLGS += -I./websocket -I./include/websocket -I./include/websocket/libwebsockets
CFLGS += -I./queue -I/usr/include/freetype2
CFLGS += -I/usr/include/oracle/11.2/client64/
CFLGS += -fopenmp -std=c++11
ifeq (1, ${DUSE_PYTHON})
	CFLGS += -DUSE_PYTHON
endif
CFLGS_EXTRA +=-I../fvd_extra_include  -I../fvd_extra_include/GL
LIBS_EXTRA=./lib
LFLGS +=  -Wl,-rpath=./lib/opencv3.3 -L./lib/opencv3.3 \
        -Wl,-rpath=$(LIBS_EXTRA)/va/lib -L./lib/va/lib -lmysqlclient  \
	-Wl,-rpath=/opt/intel/libva/lib \
	-Wl,-rpath=$(LIBS_EXTRA)/ffmpeg -L$(LIBS_EXTRA)/ffmpeg \
	-Wl,-rpath=./ \
	-Wl,-rpath=./lib \
	-lavcodec -lavformat -lavutil -lswresample -lswscale -lfreetype \
    -export-dynamic -w    \
 	-lpthread -L./lib -lzenalg -lssh2 -lobject_detection_yolov3_async -lwebsockets \
 	-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_flann 
#LFLGS +=`pkg-config --libs opencv` 
ifeq (1, ${ORACLE_VS})
	LFLGS += -Wl,-rpath=/usr/lib/oracle/11.2/client64/lib -L/usr/lib/oracle/11.2/client64/lib -lclntsh 
endif

ifeq (1, ${DUSE_GPU})
	CFLGS += -I/usr/local/cuda/include/ -DGPU -DOPENCV -DDETECT_GPU
	LFLGS += -L./alg-mutilthread/ -ldarknet -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand 
endif
ifeq (1, ${DDETECT_PLATE})
	CFLGS += -DDETECT_PLATE
	LFLGS +=     \
    -Wl,--rpath=./alg-mutilthread/lpr/
	LFLGS += -L./alg-mutilthread/lpr/ -lPlateRecognize
endif
ifeq (1, ${DBG_ENABLE})
	CFLGS += -D_DEBUG -O0 -g -DDEBUG=1 -w
endif
LFLGS += -lgomp
all: zenith 
fvd-config:
	g++  file_op.c  client_config_main.c  -w  common.c -lpthread csocket.c  client_file.c client_obj.c -o fvd-config
zenith:$(OBJS) $(INCLUDE_FILE)
	make -C alg-mutilthread/
	cp alg-mutilthread/libzenalg.so lib
	g++ $(OBJS) -o $@   $(LFLGS) ./lib/libNet.a -lpython3.5m

$(OBJS): %.o: %.c 
#	g++ -c $(CFLAGS) $< -o $@ 	$(CFLGS)  $(CFLGS_EXTRA)   -DHANHUI
	g++ -c $(CFLAGS) $< -o $@ 	$(CFLGS)  $(CFLGS_EXTRA)
clean:
	rm -rf *.o camera/*.o ini/*.o fvdmysql/*.o websocket/*.o queue/*.o
	cd alg-mutilthread;make clean;cd ..
	
install:
	mkdir .install
	cp -r fvd .install/zenith
	cp server/ipsearch .install
	cp bin/wardens .install
	cp bin/wardens.service .install
	cp bin/set_up_fedora20.sh .install
	tar cvzf  .install/lib.tar.gz lib
	cp -r cfg  .install
	mkdir .install/log
