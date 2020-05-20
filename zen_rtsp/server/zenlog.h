#ifndef __ZEN_LOGGER_H_
#define __ZEN_LOGGER_H_

#include "zenplat.h"

#include <sys/stat.h>
#include <stdarg.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/time.h>


#define MAX_LEN_DIRNAME  128
#define MAX_LEN_FILENAME 256

#define LOG_LEN_MAX          512//
#define LOG_DIR_LEN_MAX      256//
#define LOG_FILENAME_LEN_MAX 256//


#define MOD_UNDIFINE  "MOD_UNDIFINE"
#define MOD_NETWORK   "MOD_NETWORK"
#define MOD_MAIN      "MOD_MAIN"
#define MOD_SYSTEM    "MOD_SYSTEM"


#ifdef USING_DMX_MALLOC
    #define malloc(x)         ding_malloc(x,__FILE__,__LINE__,__FUNCTION__)
    #define free(x)           ding_free(x)
    #define DING_MALLOC_TRACE ding_malloc_trace();
    #define DING_MALLOC_END   ding_malloc_end();
#else
    #define DING_MALLOC_TRACE
    #define DING_MALLOC_END
#endif

//classics level,4 or 5
typedef enum {
	LevHIGHEST=0x01,
    LevFAULT=0x02,
    LevWARN=0x04,
    LevINFO=0x08,
    LevDEBUG=0x10,
    LevTRACE=0x20,

    // next are terminal display levels,they should not be used as log level
    LevDispWARN=0x07,
    LevDispINFO=0x0F,
    LevDispDEBUG=0x1F, //debug above
    LevDispTRACE=0x3F
    //but disp level could be all,considering sometimes only want to disp ones,like debug.that is why merged.
}LogLev;

typedef struct{
    int year;
    int month;
    int mday;
    int hour;
    int min;
    int sec;
    int msec;
    unsigned long tv_sec;
    unsigned long tv_usec;
    unsigned long reseved[3];
}SDATE_TIME;

typedef enum{
  EVarI32,
  EVarU32,
  EVarStr
}EType;
typedef struct{
    int           iVar;
    unsigned long uVar;
    char          strVar[256];
    EType         type;
}SVariant;

#ifdef __cplusplus
extern "C"
{
#endif


//initializate logger ,only surpport absolute path except "."

void setLog_init();

void setLog_local(char* logpath,unsigned int uSizeMax,LogLev lev);

//set the terminal display,nothing about init.
void setLog_display(LogLev dispLev);

void setLog_level(LogLev level);
//set log to diff file by module name.
//0,no  1,yes

void setLog_classify(int bclassify);

ERET setLog_udp(const char* ip,int port,LogLev lev,uint row);

void Log(const char* module,LogLev level,const char* fmt,...);
//while c do not support overloading,and this is wrote for c more than c++,
//convenient fuction cast,except this one

//level is trace
void Log0(const char* fmt,...);

SDATE_TIME* getDateTime(SDATE_TIME* dt);

unsigned long getFileSize(char* filepath);
//not contain the subdir,get ext type file total size
unsigned long getDirectorySize(char* dirpath,int* fileNum,const char * ext);

//查找文件依据修改时间  （包含目录 ext为NULL或0时，所有非隐藏文件）
ERET findFileByModTime(const char * path, const char * ext, char * filename);

//查找文件依据依字母顺序（包含目录 ext为NULL或0时，所有非隐藏文件）
ERET findFileByAlphaSort(const char * path, const char * ext, char * filename);


// 输入参数：pAppName-段落名，pKeyName-关键字，pDefStr-默认值，pRetStr-返回值（OUT），nSize-pRetStr最大长度，sCfgName-配置文件名称
// 输出参数：unsigned long-返回值的长度
int  readINIString(const char*  section, const char* pKeyName, const char* pDefStr, char* pRetStr, int nSize, const char* sCfgName);

// 函数名：WritePrivateProfileString，向配置文件中写入指定配置
// 输入参数：pAppName-段落名，pKeyName-关键字，pWriteStr-写入字符串，sCfgName-配置文件名称
// 输出参数：TRUE-成功，FALSE-失败
ERET writeINIString(const char* section, const char* pKeyName, const char* pWriteStr, const char* sCfgName);

char *appAbsDirectory(char *buffer, int length);

void  ding_malloc_trace();
void  ding_malloc_end();
void* ding_malloc(size_t size,const char* file,int line,const char* func);
void  ding_free(void* ptr);

#ifdef __linux__

void check_mem_state(int   pid);
int check_disk_state(char* dir);
ERET startdDetachThread(void *(*THREAD_ENTITY)(void*), void *arg);

#endif

#ifdef __cplusplus
}
#endif



#endif // ZEN_LOGGER_H

