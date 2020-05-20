
/* Result Sets Interface */
#ifndef SQL_CRSR
#  define SQL_CRSR
struct sql_cursor
{
    unsigned int curocn;
    void *ptr1;
    void *ptr2;
    unsigned int magic;
};
typedef struct sql_cursor sql_cursor;
typedef struct sql_cursor SQL_CURSOR;
#endif /* SQL_CRSR */

/* Thread Safety */
typedef void * sql_context;
typedef void * SQL_CONTEXT;

/* Object support */
struct sqltvn
{
    unsigned char *tvnvsn;
    unsigned short tvnvsnl;
    unsigned char *tvnnm;
    unsigned short tvnnml;
    unsigned char *tvnsnm;
    unsigned short tvnsnml;
};
typedef struct sqltvn sqltvn;

struct sqladts
{
    unsigned int adtvsn;
    unsigned short adtmode;
    unsigned short adtnum;
    sqltvn adttvn[1];
};
typedef struct sqladts sqladts;

static struct sqladts sqladt = {
    1, 1, 0,
};

/* Binding to PL/SQL Records */
struct sqltdss
{
    unsigned int tdsvsn;
    unsigned short tdsnum;
    unsigned char *tdsval[1];
};
typedef struct sqltdss sqltdss;
static struct sqltdss sqltds =
{
    1,
    0,
};

/* File name & Package Name */
struct sqlcxp
{
    unsigned short fillen;
    char  filnam[9];
};
static struct sqlcxp sqlfpn =
{
    8,
    "test3.pc"
};


static unsigned int sqlctx = 19987;


static struct sqlexd {
    unsigned long  sqlvsn;
    unsigned int   arrsiz;
    unsigned int   iters;
    unsigned int   offset;
    unsigned short selerr;
    unsigned short sqlety;
    unsigned int   occurs;
    short *cud;
    unsigned char  *sqlest;
    char  *stmt;
    sqladts *sqladtp;
    sqltdss *sqltdsp;
    unsigned char  **sqphsv;
    unsigned long  *sqphsl;
    int   *sqphss;
    short **sqpind;
    int   *sqpins;
    unsigned long  *sqparm;
    unsigned long  **sqparc;
    unsigned short  *sqpadto;
    unsigned short  *sqptdso;
    unsigned int   sqlcmax;
    unsigned int   sqlcmin;
    unsigned int   sqlcincr;
    unsigned int   sqlctimeout;
    unsigned int   sqlcnowait;
    int   sqfoff;
    unsigned int   sqcmod;
    unsigned int   sqfmod;
    unsigned char  *sqhstv[4];
    unsigned long  sqhstl[4];
    int   sqhsts[4];
    short *sqindv[4];
    int   sqinds[4];
    unsigned long  sqharm[4];
    unsigned long  *sqharc[4];
    unsigned short  sqadto[4];
    unsigned short  sqtdso[4];
} sqlstm = {12, 4};

/* SQLLIB Prototypes */
extern "C" void sqlcxt (void **, unsigned int *, struct sqlexd *, struct sqlcxp *);
extern "C" void sqlcx2t(void **, unsigned int *, struct sqlexd *, struct sqlcxp *);
extern "C" void sqlbuft(void **, char * );
extern "C" void sqlgs2t(void **, char *);
extern "C" void sqlorat(void **, unsigned int *, void *);

/* Forms Interface */
static int IAPSUCC = 0;
static int IAPFAIL = 1403;
static int IAPFTL  = 535;
extern "C" void sqliem(unsigned char *, signed int *);

/*
static char *sq0003 =
    "select VEHICLE_PASS_ID ,IMAGE_URL_PATH  from passcars where rownum<=20 order\
 by VEHICLE_PASS_ID            ";
*/
typedef struct { unsigned short len; unsigned char arr[1]; } VARCHAR;
typedef struct { unsigned short len; unsigned char arr[1]; } varchar;

/* CUD (Compilation Unit Data) Array */
static short sqlcud0[] =
{12,4130,1,0,0,
5,0,0,1,0,0,32,22,0,0,0,0,0,1,0,
20,0,0,0,0,0,27,54,0,0,4,4,0,1,0,1,97,0,0,1,97,0,0,1,97,0,0,1,10,0,0,
51,0,0,3,200,0,9,83,0,0,0,0,0,1,0,
66,0,0,3,0,0,13,90,0,0,2,0,0,1,0,2,97,0,0,2,97,0,0,
89,0,0,3,0,0,15,97,0,0,0,0,0,1,0,
104,0,0,4,0,0,30,100,0,0,0,0,0,1,0,
};


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sqlca.h"
#include "fvdoracle.h"

extern "C" void sqlgls(char * , size_t *, size_t * );
extern "C" void sqlglmt(void *, char *, size_t *, size_t *);

//////
unsigned char connt_sta = 0; //connect: 0--failed 1--ok

void sqlerr02()
{
    char    stm[120];
    size_t  sqlfc, stmlen = 120;
    unsigned int ret = 0;

    //出错时,可以把错误SQL语言给打印出来
    /* EXEC SQL WHENEVER SQLERROR CONTINUE; */


    sqlgls(stm, &stmlen, &sqlfc);
    printf("出错的SQL:%.*s\n", stmlen, stm);
    printf("出错原因:%.*s\n", sqlca.sqlerrm.sqlerrml, sqlca.sqlerrm.sqlerrmc);
    //printf("出错原因:%.70s\n", sqlca.sqlerrm.sqlerrml, sqlca.sqlerrm.sqlerrmc);
    /* EXEC SQL ROLLBACK WORK RELEASE; */

    {
        struct sqlexd sqlstm;
        sqlstm.sqlvsn = 12;
        sqlstm.arrsiz = 0;
        sqlstm.sqladtp = &sqladt;
        sqlstm.sqltdsp = &sqltds;
        sqlstm.iters = (unsigned int  )1;
        sqlstm.offset = (unsigned int  )5;
        sqlstm.cud = sqlcud0;
        sqlstm.sqlest = (unsigned char  *)&sqlca;
        sqlstm.sqlety = (unsigned short)4352;
        sqlstm.occurs = (unsigned int  )0;
        sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
    }

    connt_sta = 0;
 }

void nodata()
{
    int ret = 0;
    printf("没有发现数据\n");
    if (sqlca.sqlcode != 0)
    {
        ret = sqlca.sqlcode;
        printf("sqlca.sqlcode: err:%d \n", sqlca.sqlcode);
        return ;
    }
}

/* EXEC SQL BEGIN DECLARE SECTION; */


char    *ora_usrname = "glch"; //scott
char    *ora_passwd = "glch"; //tiger
char    *serverid = "45.51.189.10:1521/orcl";

char pass_id[100] = {0}; //string 数据类型  //varchar类型 和 char 类型的区别. 与编译选项有关系
char url_path[500] = {0};
/* EXEC SQL END DECLARE SECTION; */


void connet()
{
    int ret = 0;
    //连接数据库
    /* EXEC SQL CONNECT:usrname IDENTIFIED BY:passwd USING:serverid ; */
    {
        struct sqlexd sqlstm;
        sqlstm.sqlvsn = 12;
        sqlstm.arrsiz = 4;
        sqlstm.sqladtp = &sqladt;
        sqlstm.sqltdsp = &sqltds;
        sqlstm.iters = (unsigned int  )10;
        sqlstm.offset = (unsigned int  )20;
        sqlstm.cud = sqlcud0;
        sqlstm.sqlest = (unsigned char  *)&sqlca;
        sqlstm.sqlety = (unsigned short)4352;
        sqlstm.occurs = (unsigned int  )0;
        sqlstm.sqhstv[0] = (unsigned char  *)ora_usrname;
        sqlstm.sqhstl[0] = (unsigned long )0;
        sqlstm.sqhsts[0] = (         int  )0;
        sqlstm.sqindv[0] = (         short *)0;
        sqlstm.sqinds[0] = (         int  )0;
        sqlstm.sqharm[0] = (unsigned long )0;
        sqlstm.sqadto[0] = (unsigned short )0;
        sqlstm.sqtdso[0] = (unsigned short )0;
        sqlstm.sqhstv[1] = (unsigned char  *)ora_passwd;
        sqlstm.sqhstl[1] = (unsigned long )0;
        sqlstm.sqhsts[1] = (         int  )0;
        sqlstm.sqindv[1] = (         short *)0;
        sqlstm.sqinds[1] = (         int  )0;
        sqlstm.sqharm[1] = (unsigned long )0;
        sqlstm.sqadto[1] = (unsigned short )0;
        sqlstm.sqtdso[1] = (unsigned short )0;
        sqlstm.sqhstv[2] = (unsigned char  *)serverid;
        sqlstm.sqhstl[2] = (unsigned long )0;
        sqlstm.sqhsts[2] = (         int  )0;
        sqlstm.sqindv[2] = (         short *)0;
        sqlstm.sqinds[2] = (         int  )0;
        sqlstm.sqharm[2] = (unsigned long )0;
        sqlstm.sqadto[2] = (unsigned short )0;
        sqlstm.sqtdso[2] = (unsigned short )0;
        sqlstm.sqphsv = sqlstm.sqhstv;
        sqlstm.sqphsl = sqlstm.sqhstl;
        sqlstm.sqphss = sqlstm.sqhsts;
        sqlstm.sqpind = sqlstm.sqindv;
        sqlstm.sqpins = sqlstm.sqinds;
        sqlstm.sqparm = sqlstm.sqharm;
        sqlstm.sqparc = sqlstm.sqharc;
        sqlstm.sqpadto = sqlstm.sqadto;
        sqlstm.sqptdso = sqlstm.sqtdso;
        sqlstm.sqlcmax = (unsigned int )100;
        sqlstm.sqlcmin = (unsigned int )2;
        sqlstm.sqlcincr = (unsigned int )1;
        sqlstm.sqlctimeout = (unsigned int )0;
        sqlstm.sqlcnowait = (unsigned int )0;
        sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
    }

    if (sqlca.sqlcode != 0)
    {
        connt_sta = 0;
        ret = sqlca.sqlcode;
        printf("sqlca.sqlcode: err:%d \n", sqlca.sqlcode);
        return ;
    }
    else
    {
        connt_sta = 1;
        //printf("oracle connect ok! \n");
    }

}


void my_oracle_init()
{
    connet();
}

int  get_recored_from_oracle(ora_record_t *rds, char *passid)
{
    int i = 0;
    int ret_cnt = 0;
    char sq0003[400] = {0};

    if (!connt_sta){
        my_oracle_init();
        return 0;
    }

    snprintf(sq0003, 400, "select VEHICLE_PASS_ID,IMAGE_URL_PATH from \"V_PASS\" where ( to_number(VEHICLE_PASS_ID) >%ld and rownum<=5) order by CREATE_TIME   ", atol(passid) );
    //snprintf(sq0003, 200, "select ENAME ,JOB  from EMP where rownum<=2 order by EMPNO");


    /* EXEC SQL WHENEVER SQLERROR DO sqlerr02(); */

    //1   定义游标 为某一次查询定义一个游标
    /* EXEC SQL DECLARE c CURSOR FOR
    select VEHICLE_PASS_ID,IMAGE_URL_PATH from passcars where rownum <=20 order by VEHICLE_PASS_ID; */


    //2.  打开游标
    /* EXEC SQL  OPEN c ; */
    {
        struct sqlexd sqlstm;
        sqlstm.sqlvsn = 12;
        sqlstm.arrsiz = 4;
        sqlstm.sqladtp = &sqladt;
        sqlstm.sqltdsp = &sqltds;
        sqlstm.stmt = sq0003;
        sqlstm.iters = (unsigned int  )1;
        sqlstm.offset = (unsigned int  )51;
        sqlstm.selerr = (unsigned short)1;
        sqlstm.cud = sqlcud0;
        sqlstm.sqlest = (unsigned char  *)&sqlca;
        sqlstm.sqlety = (unsigned short)4352;
        sqlstm.occurs = (unsigned int  )0;
        sqlstm.sqcmod = (unsigned int )0;
        sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
        if (sqlca.sqlcode < 0) sqlerr02();
    }

    //3.  提取数据 fetch into
    /* EXEC SQL WHENEVER NOT FOUND DO BREAK; */

    while (1)
    {
        /* EXEC SQL FETCH c INTO :pass_id, :url_path; */

        {
            struct sqlexd sqlstm;
            sqlstm.sqlvsn = 12;
            sqlstm.arrsiz = 4;
            sqlstm.sqladtp = &sqladt;
            sqlstm.sqltdsp = &sqltds;
            sqlstm.iters = (unsigned int  )1;
            sqlstm.offset = (unsigned int  )66;
            sqlstm.selerr = (unsigned short)1;
            sqlstm.cud = sqlcud0;
            sqlstm.sqlest = (unsigned char  *)&sqlca;
            sqlstm.sqlety = (unsigned short)4352;
            sqlstm.occurs = (unsigned int  )0;
            sqlstm.sqfoff = (         int )0;
            sqlstm.sqfmod = (unsigned int )2;
            sqlstm.sqhstv[0] = (unsigned char  *)pass_id;
            sqlstm.sqhstl[0] = (unsigned long )100;
            sqlstm.sqhsts[0] = (         int  )0;
            sqlstm.sqindv[0] = (         short *)0;
            sqlstm.sqinds[0] = (         int  )0;
            sqlstm.sqharm[0] = (unsigned long )0;
            sqlstm.sqadto[0] = (unsigned short )0;
            sqlstm.sqtdso[0] = (unsigned short )0;
            sqlstm.sqhstv[1] = (unsigned char  *)url_path;
            sqlstm.sqhstl[1] = (unsigned long )500;
            sqlstm.sqhsts[1] = (         int  )0;
            sqlstm.sqindv[1] = (         short *)0;
            sqlstm.sqinds[1] = (         int  )0;
            sqlstm.sqharm[1] = (unsigned long )0;
            sqlstm.sqadto[1] = (unsigned short )0;
            sqlstm.sqtdso[1] = (unsigned short )0;
            sqlstm.sqphsv = sqlstm.sqhstv;
            sqlstm.sqphsl = sqlstm.sqhstl;
            sqlstm.sqphss = sqlstm.sqhsts;
            sqlstm.sqpind = sqlstm.sqindv;
            sqlstm.sqpins = sqlstm.sqinds;
            sqlstm.sqparm = sqlstm.sqharm;
            sqlstm.sqparc = sqlstm.sqharc;
            sqlstm.sqpadto = sqlstm.sqadto;
            sqlstm.sqptdso = sqlstm.sqtdso;
            sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
            if (sqlca.sqlcode == 1403) break;
            if (sqlca.sqlcode < 0) sqlerr02();
        }

        strcpy(rds[ret_cnt].pass_id, pass_id);
        strcpy(rds[ret_cnt].pic_path, url_path);
        char *p_space = strchr(rds[ret_cnt].pass_id,' ');
        if (p_space)
            *p_space = 0;
        p_space = strchr(rds[ret_cnt].pic_path,' ');
        if (p_space)
            *p_space = 0;

        ret_cnt++;
        printf("get records: %s, %s \n", pass_id, url_path);

    }

    //4.  EXEC SQL CLOSE c;
    /* EXEC SQL CLOSE c; */

    {
        struct sqlexd sqlstm;
        sqlstm.sqlvsn = 12;
        sqlstm.arrsiz = 4;
        sqlstm.sqladtp = &sqladt;
        sqlstm.sqltdsp = &sqltds;
        sqlstm.iters = (unsigned int  )1;
        sqlstm.offset = (unsigned int  )89;
        sqlstm.cud = sqlcud0;
        sqlstm.sqlest = (unsigned char  *)&sqlca;
        sqlstm.sqlety = (unsigned short)4352;
        sqlstm.occurs = (unsigned int  )0;
        sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
        if (sqlca.sqlcode < 0) sqlerr02();
    }

    /* EXEC SQL COMMIT WORK RELEASE; */
    
    {
        struct sqlexd sqlstm;
        sqlstm.sqlvsn = 12;
        sqlstm.arrsiz = 4;
        sqlstm.sqladtp = &sqladt;
        sqlstm.sqltdsp = &sqltds;
        sqlstm.iters = (unsigned int  )1;
        sqlstm.offset = (unsigned int  )104;
        sqlstm.cud = sqlcud0;
        sqlstm.sqlest = (unsigned char  *)&sqlca;
        sqlstm.sqlety = (unsigned short)4352;
        sqlstm.occurs = (unsigned int  )0;
        sqlcxt((void **)0, &sqlctx, &sqlstm, &sqlfpn);
        if (sqlca.sqlcode < 0) sqlerr02();
    }
  
    connt_sta = 0;
    return ret_cnt ;
}