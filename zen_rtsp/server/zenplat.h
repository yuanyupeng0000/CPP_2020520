#ifndef PLAT_CMN_H
#define PLAT_CMN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifndef uint32
#define uint32 unsigned int
#endif

#ifndef uint16
#define uint16 unsigned short
#endif

#ifndef uint8
#define uint8 unsigned char
#endif

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE  0
#endif

#ifdef _DEBUG_
#define DBG(fmt, args...) fprintf(stderr, "[Debug]: " fmt, ## args)
#else
#define DBG(fmt, args...)
#endif

typedef int ERET;
enum{
  ERET_LIMT  =-5,//达到限制，溢出
  ERET_EXST  =-4,//已存在
  ERET_PARA  =-3,//参数错误
  ERET_EXCP  =-2,//不可预知的错误
  ERET_FAIL  =-1,
  ERET_OK    = 0
};

#ifdef WIN32
    #include <winsock2.h>

#define MKDIR(x,y)            _mkdir(x)
#define MSLEEP(x)             Sleep(x)
#ifndef MUTEX_TYPE
#define MUTEX_TYPE            CRITICAL_SECTION
#define MUTEX_LOCK(x)         EnterCriticalSection(x)
#define MUTEX_UNLOCK(x)       LeaveCriticalSection(x)
#define MUTEX_INIT(x)         InitializeCriticalSection(x)
#endif
#else
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <sys/socket.h>
    #include <unistd.h>
    #include <pthread.h>


    #define MKDIR(x,y)        mkdir(x,y)
    #define MSLEEP(x)         usleep((x)*1000)
    #ifndef MUTEX_TYPE
    #define MUTEX_TYPE        pthread_mutex_t
    #define MUTEX_LOCK(x)     pthread_mutex_lock(x)
    #define MUTEX_UNLOCK(x)   pthread_mutex_unlock(x)
    #define MUTEX_INIT(x)     pthread_mutex_init((x),NULL)
    #endif
#endif

#endif // PLAT_CMN_H

