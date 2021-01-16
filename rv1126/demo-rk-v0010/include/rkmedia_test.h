#ifndef RK_MEDIA_TEST_H
#define RK_MEDIA_TEST_H

#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>   
#include <syslog.h> 
#include <libgen.h>
#include <unistd.h>
#include <ctype.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/ipc.h>   
#include <sys/shm.h> 
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/types.h> 
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <net/if.h>

#include "rkmedia_api.h"
#include "rkmedia_venc.h"
#include "rkmedia_common.h"


#ifndef PACKED
#define PACKED		__attribute__((packed, aligned(1)))
#define PACKED4		__attribute__((packed, aligned(4)))
#endif

#define IVE_BT1120_VI_WIDTH           	   1920
#define IVE_BT1120_VI_HEIGTH               1080

#define FACE_SNM_MAX_NUM                   64
#define FD_RW_FACE_COUNT_MAX    		   128

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

typedef struct _RW_FACE_INFO
{
    unsigned short x;
    unsigned short y;
    unsigned short w;
    unsigned short h;
    unsigned int trackID;	
    unsigned char type;
    unsigned char quality;
    unsigned char confidence;
    unsigned char release;	
}PACKED RW_FACE_INFO;

typedef struct _FACE_SHM_UNIT
{
	int flag;
	unsigned int s_time;
	RK_U64 u64pts;
	int face_num;
	RW_FACE_INFO face_unit[FD_RW_FACE_COUNT_MAX];
}PACKED FACE_SHM_UNIT;

typedef struct _FACE_SHM_PROFILES
{
	unsigned int read_index;
	unsigned int write_index;
	FACE_SHM_UNIT sf_unit[FACE_SNM_MAX_NUM];
}PACKED FACE_SHM_PROFILES;

typedef struct _AREA_RECT_S
{
    short x;
    short y;
    short w;
    short h;
}PACKED AREA_RECT_S;

typedef struct _URL_PORT_S
{
    char addr[64];
    unsigned int IpAddr;
    unsigned int port;
}PACKED URL_PORT_S;

typedef struct _USER_PWD_S
{
    char username[32];
    char password[32];
}PACKED USER_PWD_S;

typedef struct _FACE_HTTP_INFO_S
{
	char			httpPath[128];
	char			httpID[32];
	char			szUserName[32];		//”√ªß√˚
	char			szPassword[32];		//√‹¬Î
	unsigned short port;
}PACKED FACE_HTTP_INFO_S;


typedef struct _FaceRec_Detect_Param
{
	unsigned short	restartipc;
	unsigned short	restartface;
	unsigned char	showfaceoverlay;
	unsigned char	snapenable;
	unsigned char 	heartbeat;
	unsigned char	backpicupload;
	unsigned char 	faceoverlay;
	unsigned char	snapthreshold;
	unsigned char	pictype;
	unsigned char	snapmode;
	unsigned char	snapinterval;
	unsigned char	backpicquality;
	unsigned char	facepicquality;
	unsigned short	minifacepixel;
	int    			sendtofaceserver;
	URL_PORT_S	    stserverurl;
	unsigned char	facepicuploadmodel;
	URL_PORT_S		stUrlPort;
	char			szftppath[128];
	USER_PWD_S		stuserpwd;
	AREA_RECT_S	    stdectectarea[9];
	URL_PORT_S		stsdkrlport;
	USER_PWD_S		stsdkuserpwd;
	unsigned long   inttotal;
	unsigned long   outtotal;
	FACE_HTTP_INFO_S sthttpinfo;

	unsigned int writeflag;
	unsigned int readflag;
	
} PACKED FaceRec_Detect_Param;

typedef struct _Face_Upgrade_Param_S
{
    unsigned int size;
    unsigned int baudrate;
    unsigned int dowloadmode;
    unsigned char protocolversion[64];
    unsigned char softwareversion[64];
    unsigned char fimwareversion[64];
} Face_Upgrade_Param;

typedef struct DEVICE_INFO_S
{
	char deviceid[16];
	int faceexposureenable;
	int faceexposurevalue;
} DEVICE_INFO_S;

#endif


