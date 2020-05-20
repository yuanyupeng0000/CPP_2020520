#include "zenlog.h"

typedef struct{
    int logSock;
    struct sockaddr_in logAddr;
    unsigned char bClassify;
    LogLev localLev;
    char logPath[LOG_DIR_LEN_MAX];
    unsigned int uSizeMax_K;
    LogLev terminalDisp;
}SLogSet;
static SLogSet sLog={-1,{0},0,(LogLev)0,{0},0,LevDispTRACE};
//static SLogSet sLog;
static MUTEX_TYPE g_LogMutex;

typedef struct SmemMgrNode{
        int addr;
        int size;
        int line;
        char file[128];
        char func[32];
        struct SmemMgrNode* next;
}SmemMgrNode;
static SmemMgrNode g_cmemMgr={0,0,0,{0},{0},0};
static MUTEX_TYPE g_cmemMutex;
char g_memLogName[64]={0};


unsigned long getFileSize(char* filepath)
{	//stat only can get the file size ,none dir
    if(filepath==NULL){return -1;}
    struct stat dirstat;
    memset(&dirstat,0,sizeof(dirstat));
    if(stat(filepath,&dirstat)==-1)
    {
            printf("get file state fail:%d\n",errno);
            return -1;
    }
    return dirstat.st_size;
}

//auto make a line
static inline void mkline(char* buf)
{
        int end;
        if(buf==NULL||((end=strlen(buf))==0)){return;}
        //assure last two \n\0
        if(buf[end-1]!='\n'){buf[end]='\n';}
        buf[LOG_LEN_MAX-2]='\n';
        buf[LOG_LEN_MAX-1]=0;
}

SDATE_TIME* getDateTime(SDATE_TIME* dt)
{
    struct timeval tv;
    time_t tm;
    struct tm * ptm;

    gettimeofday(&tv, NULL);
    tm = tv.tv_sec;
    ptm = localtime(&tm);


    dt->year=ptm->tm_year+1900;
    dt->month=ptm->tm_mon+1;
    dt->mday= ptm->tm_mday;
    dt->hour= ptm->tm_hour;
    dt->min=ptm->tm_min;
    dt->sec=ptm->tm_sec;
    dt->msec=tv.tv_usec % 1000;
    dt->tv_sec =tv.tv_sec;
    dt->tv_usec=tv.tv_usec;
    return dt;
}

static inline int makeLogHeader(char* buf,const char *module, LogLev level)
{
    struct timeval tv;
    struct tm * ptm;
    const char *levelstr=NULL;
    switch(level){
            case 32:
            levelstr="trace";break;
            case 16:
            levelstr="debug";break;
            case 8:
            levelstr="info";break;
            case 4:
            levelstr="warning";break;
            case 2:
            levelstr="error";break;
            case 1:
            levelstr="highest";break;
            default:
            levelstr="trace";break;
    }
    char timestr[40]={0};
    snprintf(timestr,30,"%s",module);
    timestr[15]=0;
    sprintf(buf,"[%-15s]",timestr);


    gettimeofday(&tv, NULL);
    ptm = localtime(&tv.tv_sec);

    sprintf(timestr, "%02d:%02d:%02d.%03ld",
                ptm->tm_hour,
                ptm->tm_min,
                ptm->tm_sec,
                tv.tv_usec % 1000);

    sprintf(buf+17,"[%-7s][%12s] :",levelstr,timestr);
    return 42;
}

ERET findFileByModTime(const char * path, const char * ext, char * filename)
{
    char pathBuff[LOG_DIR_LEN_MAX];
    memset(pathBuff,0,LOG_DIR_LEN_MAX);
    struct stat st;
    time_t ct_min=0x7FFFFFFF;
    DIR* dirfd=opendir(path);
    int findFlag=ERET_FAIL;
    if(dirfd==NULL){
            printf("open dir error:%s,%s\n",path,strerror(errno));
            return ERET_EXCP;
    }

    struct dirent* dp;
    ////readdir auto point to the next
    while( (dp=readdir(dirfd))!=NULL )
    {            
        if(dp->d_name[0]=='.')//check posion?
        {
            continue;
        }
        if(NULL!=ext&&0!=ext[0])
        {
            if(strstr(dp->d_name,ext)==NULL)
            {
                continue;
            }
        }
        memset(&st,0,sizeof(st));
        sprintf(pathBuff, "%s/%s", path, dp->d_name);
        //printf("%s\n",pathBuff);
        if(stat(pathBuff,&st)==0)
        {
                findFlag=ERET_OK;
                if(st.st_mtime<ct_min)
                {
                    ct_min=st.st_mtime;
                    strcpy(filename,dp->d_name);
                    //printf("%s\n",filename);
                }
        }
    }
    closedir(dirfd);

    return findFlag;
}

//not contain the subdir
unsigned long getDirectorySize(char* dirpath,int* logNum,const char * ext)
{
        *logNum=0;
        unsigned long dirSize=0;
        char pathBuff[LOG_DIR_LEN_MAX];
        memset(pathBuff,0,LOG_DIR_LEN_MAX);

        DIR* dirfd=opendir(dirpath);
        if(dirfd==NULL){
                printf("open dir error:%s,%s\n",dirpath,strerror(errno));
                return -1;
        }

        struct dirent* dp;
        while( (dp=readdir(dirfd))!=NULL )
        {
                sprintf(pathBuff, "%s/%s", dirpath, dp->d_name);
                if(ext==NULL||!strcmp(ext,"*"))
                {
                    dirSize+=getFileSize(pathBuff);
                    (*logNum)++;
                    continue;
                }
                if(strstr(dp->d_name,ext))//check posion?
                {
                        dirSize+=getFileSize(pathBuff);
                        (*logNum)++;
                }
        }
        closedir(dirfd);
        return dirSize;
}

static void logtofile(char* logdir,char* filename,char* buf)
{
    int iNew = 0;	//
    int FileNum;
    char szDelFile[LOG_FILENAME_LEN_MAX] = { 0 };
    char szFileExt[8] = { 0 };

    MUTEX_LOCK(&g_LogMutex);
    FILE *fd = fopen(filename, "r");
    if(fd == 0)
            iNew = 1;
    else
            fclose(fd);
            
    fd = fopen(filename, "a");
    if(fd == 0 )
    {
        fprintf(stderr,"[%s] open log file  failed: %s,error:%s\n",__FUNCTION__, filename,strerror(errno));
        MUTEX_UNLOCK(&g_LogMutex);
        return;
    }
    if(fseek(fd, 0, SEEK_END) ==-1)
    {
        fprintf(stderr,"[%s] fseek log file failed: %s,error:%s\n",__FUNCTION__, filename,strerror(errno));
        fclose(fd);
        MUTEX_UNLOCK(&g_LogMutex);
        return;
    }
    else
    {
       if(fputs(buf, fd)==-1)
	   {
            fprintf(stderr,"[%s] fputs log file failed: %s,error:%s\n",__FUNCTION__, filename,strerror(errno));
	   }
	   fclose(fd);
    }
    MUTEX_UNLOCK(&g_LogMutex);
    
    if(iNew == 1){
		sprintf(szFileExt, ".log");
		do{
			unsigned long dirSize=getDirectorySize(logdir,&FileNum,".log");
			Log(MOD_SYSTEM,LevINFO,"SizeCheck in [%s],Size[%uK/%uK]",logdir,dirSize/1000,sLog.uSizeMax_K);
			if(sLog.uSizeMax_K>0&&dirSize>sLog.uSizeMax_K*1000)
			{
		        if(findFileByModTime(logdir,szFileExt , szDelFile) == 0)
		        {
	                char pathname[LOG_DIR_LEN_MAX];
	                sprintf(pathname,"%s/%s",logdir,szDelFile);
	                if(unlink(pathname)==-1){
	                        Log(MOD_SYSTEM,LevWARN,"unlink [%s]failed,error:\n",pathname,strerror(errno));break;
	                }
	                Log(MOD_SYSTEM,LevINFO,"deleted logfile[%s]",pathname);
		        }
		        else{printf("[%s]FindFile error!",__FUNCTION__);break;}
			}
			else{
				break;
			}
        }while(1);
    }
}

void setLog_init()
{
    char pwdpath[LOG_DIR_LEN_MAX];
    getcwd(pwdpath,LOG_DIR_LEN_MAX);
    strcpy(sLog.logPath,pwdpath);
    MUTEX_INIT(&g_LogMutex);
    MUTEX_INIT(&g_cmemMutex);

}

void setLog_local(char* logpath,unsigned int uSizeMax,LogLev lev)
{
        sLog.uSizeMax_K=uSizeMax;
        sLog.localLev=lev;
        char pwdpath[LOG_DIR_LEN_MAX];
        getcwd(pwdpath,LOG_DIR_LEN_MAX);
        if(strlen(logpath)>=LOG_DIR_LEN_MAX||logpath==NULL||strlen(logpath)==0||strcmp(logpath,".")==0)
        {
            strcpy(sLog.logPath,pwdpath);
        }
        else
        {
            strcpy(sLog.logPath,logpath);
        }
        if( access(sLog.logPath, F_OK) == -1 )
        {
                if( MKDIR(sLog.logPath, S_IXUSR) == -1 )
                {
                        strcpy(sLog.logPath,pwdpath);
                        Log(MOD_SYSTEM,LevWARN,"create App's logdir[%s] failed,change log to[%s]",logpath,pwdpath);

                }
        }
        Log(MOD_SYSTEM,LevHIGHEST,"init success,logdir:%s",sLog.logPath);


}

static void LogNet(const char* buf)
{
    if(buf==NULL||sLog.logSock == -1)return;
    MUTEX_LOCK(&g_LogMutex);
	//printf("*****************\n");
    sendto(sLog.logSock,buf,strlen(buf),0,(struct sockaddr*)&sLog.logAddr,sizeof(sLog.logAddr));
    MUTEX_UNLOCK(&g_LogMutex);
}

void Log(const char* module,LogLev level,const char* fmt,...)
{
    struct timeval tv;
    struct tm * ptm;
    if(level&sLog.terminalDisp ||(sLog.localLev>=level))
    {
        char buf[LOG_LEN_MAX]={0};
        int pos=makeLogHeader(buf,module,level);

        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf+pos,LOG_LEN_MAX-pos-1,fmt, ap);
        va_end(ap);


        mkline(buf);

        if(sLog.terminalDisp&level){printf("%s",buf);}

        if(sLog.localLev>=level)
        {
            char szFileName[LOG_FILENAME_LEN_MAX] = { 0 };
            char szFilepath[LOG_DIR_LEN_MAX]={0};
            gettimeofday(&tv, NULL);
            ptm = localtime(&tv.tv_sec);
            sprintf(szFileName, "%04d%02d%02d",
                    ptm->tm_year+1900,
                    ptm->tm_mon+1,
                    ptm->tm_mday);
            szFileName[8]=0;
            if(sLog.bClassify)
            {
                sprintf(szFilepath,"%s/%s%s.log",sLog.logPath,module,szFileName);
            }else{
                sprintf(szFilepath,"%s/%s.log",sLog.logPath,szFileName);
            }//logtoProxy(buf);

            if(sLog.logSock==-1)
			{
                logtofile(sLog.logPath,szFilepath,buf);
			}else{
				LogNet(buf);
			}
		}
    }
}

void Log0(const char* fmt,...)
{
    struct timeval tv;
    struct tm * ptm;
    if(LevTRACE&sLog.terminalDisp||(sLog.localLev>=LevTRACE))
    {
        char buf[LOG_LEN_MAX+128]={0};
        int pos=makeLogHeader(buf,MOD_UNDIFINE,LevTRACE);

        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf+pos,LOG_LEN_MAX,fmt, ap);
        va_end(ap);

        mkline(buf);
        if(sLog.terminalDisp&LevTRACE){printf("%s",buf);}
        if(sLog.localLev>=LevTRACE)
        {
            //logtoProxy(buf);
            char szFileName[LOG_FILENAME_LEN_MAX] = { 0 };
            char szFilepath[LOG_DIR_LEN_MAX]={0};
            gettimeofday(&tv, NULL);
            ptm = localtime(&tv.tv_sec);
            sprintf(szFileName, "%04d%02d%02d",
                    ptm->tm_year+1900,
                    ptm->tm_mon+1,
                    ptm->tm_mday);
            szFileName[8]=0;
            if(sLog.bClassify)
            {
                sprintf(szFilepath,"%s/%s%s.log",sLog.logPath,MOD_UNDIFINE,szFileName);
            }
            else
            {
                sprintf(szFilepath,"%s/%s.log",sLog.logPath,szFileName);
            }//logtoProxy(buf);
            if(sLog.logSock==-1)
			{
                logtofile(sLog.logPath,szFilepath,buf);
			}else{
				LogNet(buf);
			}

        }
    }
}

void setLog_Lev(LogLev logLev)
{
    sLog.localLev=logLev;
}
void setLog_display(LogLev dispLev)
{
    sLog.terminalDisp=dispLev;
}
void setLog_classify(int bclassify)
{
    sLog.bClassify=bclassify;
}

ERET setLog_udp(const char* ip,int port,LogLev lev,uint row)
{
    MUTEX_LOCK(&g_LogMutex);
    //sLog.netLev=lev;
    if(ip==NULL)
    {
        if(sLog.logSock!=-1)
		{
            close(sLog.logSock);
            sLog.logSock=-1;
		}
        MUTEX_UNLOCK(&g_LogMutex);
        return ERET_OK;
    }
    else
    {
        if(sLog.logSock!=-1)
        {
            close(sLog.logSock);
            sLog.logSock=-1;
        }
    }
    sLog.logAddr.sin_family=AF_INET;
    sLog.logAddr.sin_port=htons(port);
    sLog.logAddr.sin_addr.s_addr=inet_addr(ip);
    if((sLog.logSock=socket(AF_INET,SOCK_DGRAM,0))==-1)
	{
		printf("create log socket failed\n");
        MUTEX_UNLOCK(&g_LogMutex);
        return ERET_EXCP;
	}
    int begin=0;
    if(sendto(sLog.logSock,(char*)&begin,4,0,(struct sockaddr*)&sLog.logAddr,sizeof(sLog.logAddr))==4)
	{
        MUTEX_UNLOCK(&g_LogMutex);
        printf("send success!\n");
        return ERET_OK;
	}else{
        MUTEX_UNLOCK(&g_LogMutex);
        printf("send failed!\n");
        return ERET_OK;
	}
}


static char* FiltSpace( char* pKey )
{
    while( *pKey==' ' && *pKey!=0 )
    {
        pKey ++;
    }

    return pKey;
}



int readINIString(const char* section, const char* pKeyName, const char* pDefStr, char* pRetStr, int nSize, const char* sCfgName)
{
    char    FileName[128] = { 0 };
    char    strTxtLine[1024] = { 0 };				// 定义一行字符串
    int     nStep = 0;
    int    blExit = FALSE;
    char*   pKey;

    // 获取应用程序所在的目录
    if ( !getcwd( FileName, 100 ) )
    {
        Log0( "[%s] 获取配置文件<%s>所在目录失败！", __FUNCTION__, sCfgName );
    }

    // 设置文件的全路径名
    strcat( FileName, "/" );
    strcat( FileName, sCfgName );

    // 打开配置文件
    FILE *fp;
    fp = fopen( FileName, "rb" );
    if( fp == NULL)
    {
        blExit = TRUE;
    }

    sprintf( FileName, "[%s]", section );	// 应用名

    memset( pRetStr, 0, nSize);
    strncpy( pRetStr, pDefStr, nSize-1);							// 默认返回值

    while( !blExit )
    {
        memset( strTxtLine, 0, 1024 );
        if( fgets( strTxtLine, 1024-1, fp ) == NULL )
        {
            break;
        }

        pKey = strTxtLine;

        switch( nStep )
        {
            case 0: // 查找应用名
                if( strncmp( (char*)strTxtLine, (char*)FileName, strlen((char*)FileName) ) == 0 )
                {
                    nStep ++;
                }
                break;
            case 1: // 查找关键字
                if( strncmp((char*)strTxtLine, (char*)pKeyName, strlen((char*)pKeyName) )==0 )
                {
                    pKey = strTxtLine;
                }
                else
                {
                    pKey = NULL;
                }
                if( pKey == NULL )
                {   // 是否下一字段开始
                    if( strstr( strTxtLine, "[" ) )
                    {
                        blExit = TRUE;
                    }
                    break;
                }

                pKey += strlen(pKeyName);
                pKey = FiltSpace( pKey );
                if( pKey[0] != '=' )
                {
                    break;
                }

                pKey ++;
                pKey = FiltSpace( pKey );
                if( pKey[0] != 0 )
                {
                    strncpy( pRetStr, pKey, nSize-1 );
                }

                blExit = TRUE;

                break;
            default:
                break;
        }
    };

    if( fp )
        fclose( fp );

    // 过滤掉后面的空格和回车换行
    int nLen = strlen(pRetStr);
    int i = 0;
    int j = 0;
    for( i=0; i<nLen; i++ )
    {
        if( pRetStr[i]==' ' )
        {
            j ++;
        }
        else if( pRetStr[i]==0x0d || pRetStr[i]==0x0a )
        {
            pRetStr[i-j] = 0;
            break;
        }
        else
        {
            j = 0;
        }
    }

    return strlen(pRetStr);
}


int writeINIString(const char* section, const char* pKeyName, const char* pWriteStr, const char* sCfgName)
{
    char    FileName[128]    = { 0 };
    char    strTxtLine[1024] = { 0 };       // 定义一行字符串
    char    strKeyTxt[1024]  = { 0 };
    char    strAppTxt[1024]  = { 0 };
    char    szBuf[1024*10]   = { 0 };
    char*   pKey = NULL;

    int     iStep   = 0;
    int     iOffset = 0;
    int     iRet    = 0;
    int     iNewKeyLen = 0;

    int    blExit = FALSE;

    // 获取应用程序所在的目录
    if ( !getcwd( FileName, 100 ) )
    {
        return ERET_FAIL;
    }

    // 设置文件的全路径名
    strcat( FileName, "/" );
    strcat( FileName, sCfgName );

    // 打开配置文件
    FILE *fp;
    fp = fopen( FileName, "r+" );
    if( fp == NULL)
    {
        fp = fopen( FileName, "w+" );
        if(fp == NULL)
        {
            return ERET_EXCP;
        }
    }

    sprintf(strAppTxt, "[%s]\x0a",  section );             // 应用名
    sprintf(strKeyTxt, "%s=%s", pKeyName, pWriteStr);   // 关键字
    iNewKeyLen = strlen( strKeyTxt );

    while( !blExit )
    {
        memset(strTxtLine, 0, 1024 );
        if( fgets( strTxtLine, 1024-1, fp ) == NULL )
        {
            break;
        }

        pKey = strTxtLine;

        switch( iStep )
        {
            case 0: // 查找应用名
                if( strncmp( strTxtLine, strAppTxt, strlen(strAppTxt)-2 ) == 0 )
                {
                    iStep ++;
                }
                break;
            case 1: // 查找关键字
                if( strncmp((char*)strTxtLine, (char*)pKeyName, strlen((char*)pKeyName) ) != 0 )
                {
                    // 是否下一应用开始
                    if( strstr( strTxtLine, "[" ) )
                    {
                        blExit = TRUE;
                    }
                    break;
                }

                iOffset = strlen(strTxtLine);
                if( iOffset == iNewKeyLen+1 )
                {
                    fseek(fp, -iOffset, SEEK_CUR);
                    iRet = fputs(strKeyTxt, fp);
                }
                else if( iOffset > iNewKeyLen+1)
                {
                    memset(strKeyTxt+iNewKeyLen, ' ', iOffset-iNewKeyLen);
                    strKeyTxt[iOffset-1] = 0x0a;
                    strKeyTxt[iOffset-0] = 0x00;
                    fseek(fp, -iOffset, SEEK_CUR);
                    iRet = fputs(strKeyTxt, fp);
                }
                else
                {
                    iRet = fread(szBuf, 1, 1024*10, fp);
                    fseek(fp, -(iOffset+iRet), SEEK_CUR);
                    iOffset = iRet;
                    strcat(strKeyTxt, "\x0a");
                    iRet = fputs(strKeyTxt, fp);
                    fwrite(szBuf, 1, iOffset, fp);
                }

                fclose( fp );

                if(iRet > 0)
                {
                    return ERET_OK;
                }
                else
                {
                    return ERET_FAIL;
                }

                break;
            default:
                blExit = TRUE;
                break;
        }
    };

    if( fp )
    {
        fclose( fp );
    }

    return ERET_FAIL;
}




char* appAbsDirectory(char* buffer,int length)
{
#ifdef __linux__
     int nLen=readlink("/proc/self/exe",buffer,length);
     if(nLen<0||nLen>=MAX_LEN_FILENAME){
         return NULL;
     }
     while(buffer[nLen]!='/')
     {
         nLen--;
     }
     buffer[nLen+1]='\0';
#endif
     return buffer;
}



void* ding_malloc(size_t size,const char* file,int line,const char* func)
{
    printf("calls ding_malloc\n");
    SmemMgrNode* pre;
    SmemMgrNode* end;
    void* dest;
    SmemMgrNode* mem;
    MUTEX_LOCK(&g_cmemMutex);
    pre=&g_cmemMgr;
    end=g_cmemMgr.next;
    dest=malloc(size);
    mem=(SmemMgrNode*)malloc(sizeof(SmemMgrNode));

    memset(mem,0,sizeof(SmemMgrNode));
    mem->addr=(int)dest;
    mem->size=size;
    mem->line=line;
    strncpy(mem->file,file,127);
    strncpy(mem->func,func,31);
    while(end){pre=end;end=end->next;}
    pre->next=mem;
    g_cmemMgr.size++;
    MUTEX_UNLOCK(&g_cmemMutex);

    return dest;
}

void ding_free(void* dest)
{
    printf("calls ding_free\n");
    SmemMgrNode* pre;
    SmemMgrNode* cur;
    MUTEX_LOCK(&g_cmemMutex);
    pre=&g_cmemMgr;
    cur=g_cmemMgr.next;
    while(cur)
    {
        if(cur->addr==(int)dest)
        {
            pre->next=cur->next;
            free(cur);
            g_cmemMgr.size--;
            break;
        }
        pre=cur;
        cur=cur->next;
    }
    free(dest);
    MUTEX_UNLOCK(&g_cmemMutex);
    return;
}

void ding_malloc_trace()
{
    int i=1;
    SmemMgrNode* p;
    char tmp[16];
    time_t tNow=time(NULL);
    if(g_memLogName[0]==0)
    {
        strcpy(g_memLogName,"MemTrace");
        strftime(g_memLogName+8, 50, "%Y-%m-%d_%H_%M_%S.log", localtime(&tNow));
        //printf("%s\n",g_memLogName);
    }
    MUTEX_LOCK(&g_cmemMutex);
    p=g_cmemMgr.next;
    FILE* logFile=fopen(g_memLogName,"a");
    strftime(tmp, 16,"%H:%M:%S", localtime(&tNow));
    fprintf(logFile,"[ MemTrace (%s):  total %4d ]\n",tmp,g_cmemMgr.size);
    while(p)
    {
        fprintf(logFile,"%3d: addr[0x%08X],size[%4d],line[%4d],func[%s],file[%s]\n",
               i,p->addr,p->size,p->line,p->func,p->file);
        p=p->next;i++;
    }
    fclose(logFile);
    MUTEX_UNLOCK(&g_cmemMutex);
}

void ding_malloc_end()
{
    SmemMgrNode* tmp=NULL;
    MUTEX_LOCK(&g_cmemMutex);
    while(g_cmemMgr.next)
    {
        tmp=g_cmemMgr.next;
        g_cmemMgr.next=g_cmemMgr.next->next;
        free(tmp);
    }
    MUTEX_UNLOCK(&g_cmemMutex);
}