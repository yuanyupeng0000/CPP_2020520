#ifndef __COMMON__
#define __COMMON__
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <string.h>
#include <time.h>
#define BUFSIZE 200
//#define print(...)
//typedef   void (*THREAD_ENTITY)(void*);
extern unsigned char prt_log_leve;


enum PROTOCOL_TYPE {

	PROTO_PRIVATE = 0,
	PROTO_HAIXIN = 1,
	PROTO_YIHUALU = 2,
	PROTO_NANJING = 3,
	PROTO_HUAITONG = 4,
	PROTO_HUAITONG_PERSON = 5,
	PROTO_PRIVATE_PERSON = 6,
	PROTO_WS_RADAR = 7,
	PROTO_WS_VIDEO = 8,
	PROTO_WS_RADAR_VIDEO = 9,
	PROTO_JIERUI = 10,
	PROTO_NONE = 99,
};

typedef struct G_ALL_LOCK {
	pthread_mutex_t proto_lock;
} g_all_lock_t;


typedef void *(*THREAD_ENTITY)(void*);
typedef int (*p_func)(void*, void*);
typedef int (*p_func0)(void*);
typedef void *(*THREAD_ENTITY1)(int index, char *data, int size);
typedef struct timed_func {
	THREAD_ENTITY func;
	int time;
	void *data;
	pthread_t handle;
} m_timed_func_data;
inline void print_stacktrace();
inline void log_file(char *fname, char *log)
{
	char log_buf[BUFSIZE];
	char filename[BUFSIZE];
	char dirname[BUFSIZE] = { "log" };
	char cmd_buf[BUFSIZE] = { };
//	if(dname!=NULL){
//		strcpy(dirname,dname);
//	}
	FILE *fp = NULL;
	if ((fp = fopen(dirname, "r")) == NULL) {
		sprintf(cmd_buf, "mkdir %s", dirname);
		system(cmd_buf);
	}
	else
		fclose(fp);
	time_t timer;
	struct tm *tblock;
	/* gets time of day */
	timer = time(NULL);
	/* converts date/time to a structure */
	tblock = localtime(&timer);
	sprintf(filename, "%d_%d_%d", tblock->tm_year - 100 + 2000, tblock->tm_mon + 1,
	        tblock->tm_mday);
	if (fname != NULL) {
		strcpy(filename, fname);
	}

	strcat(dirname, "/");
	strcat(dirname, filename);
	fp = NULL;
	fp = fopen(dirname, "a");
	if (fp != NULL) {
		//	fwrite(log, 1, strlen(log)+1, fp);
		fwrite(log, 1, strlen(log), fp);
		fclose(fp);
	} else {
		perror("file  open fail");
		printf("file :%s\n", dirname);
	}
}

#define LOGFILE
#ifdef LOGFILE

//#define print_info(...) { \
//printf("[ info ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
//
//#define print_err(...) { \
//printf("[ err ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
//
//
//#define print_cmd(...) { \
//printf("[ cmd ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
////#define print_net(...)
//#define print_net(...) { \
//printf("[ net ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
//
//
//#define print_config(...) { \
//printf("[ config ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
////#define print_alg(...)
//#define print_alg(...) { \
//printf("[ alg ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
//
//#define print_sig(...) { \
//printf("[ sig ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
//
//#define print_change(...) { \
//printf("[ state change ]:");\
//printf(__VA_ARGS__);	\
//printf(".................%s(%d)\n", __FILE__,__LINE__); \
//fflush(NULL) ;}
//
//
//#define print_decoder(...) { \
//char t_string[100] ; \
//char label[10]="decoder" ;\
//printf(__VA_ARGS__);\
//sprintf(t_string,"[%s]..................%s(%d)\n", label,__FILE__,__LINE__); \
//printf("%s",t_string); \
//log_file(NULL,t_string);	\
//fflush(NULL) ; \
//}
//
#define print(label,...)
//#define print(label,...) { \
//char t_string1[100] ; \
//char t_string2[100] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"[%s].................%s(%d)\n", #label,__FILE__,__LINE__); \
//strcat(t_string1,t_string2);	\
//printf("%s\n",t_string1); \
//log_file(NULL,t_string1);	\
//fflush(NULL) ;\
//}

#define print_stack(...)
//#define print_stack(...) { \
//char label[BUFSIZE] ={"[stack]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//print_stacktrace(); \
//fflush(NULL) ; \
//}
#define print_info(...)
//#define print_info(...) { \
//char label[BUFSIZE] ={"[info]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}
#define print_error(...)
//#define print_error(...) { \
//char label[BUFSIZE] ={"[error]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}
#define print_decoder(...)
//#define print_decoder(...) { \
//char label[BUFSIZE] ={"[decoder]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}
#define print_alg(...)
//#define print_alg(...) { \
//char label[BUFSIZE] ={"[alg]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}
#define print_sig_state(...)

//#define print_sig_state(...) { \
//char label[BUFSIZE] ={"[sig_state]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}

#define print_cam_state(...)
//#define print_cam_state(...) { \
//char label[BUFSIZE] ={"[cam_state]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}


#define print_net(...)

//#define print_net(...) { \
//char label[BUFSIZE] ={"[net]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}
#define print_config(...)
//#define print_config(...) { \
//char label[BUFSIZE] ={"[config]"}; \
//char t_string1[BUFSIZE] ; \
//char t_string2[BUFSIZE] ; \
//sprintf(t_string1,__VA_ARGS__);	\
//sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__); \
//strcat(label,t_string1);\
//strcat(label,t_string2);\
//printf("%s",label); \
//log_file(NULL,label); \
//fflush(NULL) ; \
//}
#define LOG_NONE
#undef LOG_NONE
#define LOG_LEVEL1
#undef LOG_LEVEL1
#define LOG_LEVEL2
//#undef LOG_LEVEL2
#define LOG_LEVEL3
#undef LOG_LEVEL3


inline void add_title(char *label, char *str, int line, char *src_file)
{
	char tmp_str[BUFSIZE] = {};
	char time_label[BUFSIZE] = {};
	char title_label[BUFSIZE] = {};
	char line_label[BUFSIZE] = {};

	struct tm *p_tm;
	time_t timer;
	/* gets time of day */
	timer = time(NULL);
	/* converts date/time to a structure */
	p_tm = localtime(&timer);
	sprintf(time_label, "[%d:%d:%d]", p_tm->tm_hour, p_tm->tm_min, p_tm->tm_sec);
	sprintf(title_label, "[%s]", label);
	sprintf(line_label, "(%s:%d)====>", src_file, line);

#if 0
#if defined( LOG_NONE)
	if (strcmp(label, "info---1") == 0\
	        || strcmp(label, "err--1") == 0 \
	        || strcmp(label, "exit--1") == 0 \
	        || strcmp(label, "net---1") == 0 \
	        || strcmp(label, "in_loop---1") == 0 \
	        || strcmp(label, "config_change---1") == 0 \
	        || strcmp(label, "stack") == 0 \
	        || strcmp(label, "config---1") == 0 \
	        || strcmp(label, "cam_info---1") == 0 \
	        || strcmp(label, "check_setting--1") == 0 \
	        || strcmp(label, "check_client_cmd--1") == 0 \
	        || strcmp(label, "cam_state--1") == 0 \
	        || strcmp(label, "sig_state--1") == 0 \
	        || strcmp(label, "alg---1") == 0 \
	        || strcmp(label, "clients_msg---1") == 0 \
	        || strcmp(label, "camera_msg---1") == 0 \
	        || strcmp(label, "debug_sig----1") == 0 )
#elif defined( LOG_LEVEL1)
	if (strcmp(label, "info---1") == 0\
	        || strcmp(label, "err") == 0 \
	        || strcmp(label, "exit") == 0 \
	        || strcmp(label, "net---1") == 0 \
	        || strcmp(label, "in_loop---1") == 0 \
	        || strcmp(label, "config_change---1") == 0 \
	        || strcmp(label, "stack") == 0 \
	        || strcmp(label, "config---1") == 0 \
	        || strcmp(label, "cam_info---1") == 0 \
	        || strcmp(label, "check_setting--1") == 0 \
	        || strcmp(label, "check_client_cmd--1") == 0 \
	        || strcmp(label, "cam_state--1") == 0 \
	        || strcmp(label, "sig_state--1") == 0 \
	        || strcmp(label, "alg---1") == 0 \
	        || strcmp(label, "clients_msg") == 0 \
	        || strcmp(label, "camera_msg") == 0 \
	        || strcmp(label, "debug_sig----1") == 0 )
#elif defined( LOG_LEVEL2)
	if (strcmp(label, "info") == 0\
	        || strcmp(label, "err") == 0 \
	        || strcmp(label, "exit") == 0 \
	        || strcmp(label, "net") == 0 \
	        || strcmp(label, "in_loop---1") == 0 \
	        || strcmp(label, "config_change---1") == 0 \
	        || strcmp(label, "stack") == 0 \
	        || strcmp(label, "config---1") == 0 \
	        || strcmp(label, "cam_info---1") == 0 \
	        || strcmp(label, "check_setting--1") == 0 \
	        || strcmp(label, "check_client_cmd--1") == 0 \
	        || strcmp(label, "cam_state--1") == 0 \
	        || strcmp(label, "sig_state--1") == 0 \
	        || strcmp(label, "server") == 0 \
	        || strcmp(label, "alg---1") == 0 \
	        || strcmp(label, "clients_msg") == 0 \
	        || strcmp(label, "camera_msg") == 0 \
	        || strcmp(label, "debug_sig") == 0 )
#else defined( LOG_LEVEL3)
	if (strcmp(label, "info") == 0\
	        || strcmp(label, "err") == 0 \
	        || strcmp(label, "exit") == 0 \
	        || strcmp(label, "net---1") == 0 \
	        || strcmp(label, "in_loop---1") == 0 \
	        || strcmp(label, "config_change---1") == 0 \
	        || strcmp(label, "stack") == 0 \
	        || strcmp(label, "config---1") == 0 \
	        || strcmp(label, "cam_info---1") == 0 \
	        || strcmp(label, "check_setting--1") == 0 \
	        || strcmp(label, "check_client_cmd--1") == 0 \
	        || strcmp(label, "cam_state--1") == 0 \
	        || strcmp(label, "sig_state--1") == 0 \
	        || strcmp(label, "alg---1") == 0 \
	        || strcmp(label, "clients_msg") == 0 \
	        || strcmp(label, "camera_msg") == 0 \
	        || strcmp(label, "debug_sig----1") == 0 )
#endif

#endif
		if (strcmp(label, "info") == 0\
		        || strcmp(label, "err") == 0 \
		        || strcmp(label, "exit") == 0 \
		        || strcmp(label, "net---1") == 0 \
		        || strcmp(label, "in_loop---1") == 0 \
		        || strcmp(label, "config_change---1") == 0 \
		        || strcmp(label, "stack") == 0 \
		        || strcmp(label, "config---1") == 0 \
		        || strcmp(label, "cam_info---1") == 0 \
		        || strcmp(label, "check_setting--1") == 0 \
		        || strcmp(label, "check_client_cmd--1") == 0 \
		        || strcmp(label, "cam_state--1") == 0 \
		        || strcmp(label, "sig_state--1") == 0 \
		        || strcmp(label, "alg---1") == 0 \
		        || strcmp(label, "clients_msg") == 0 \
		        || strcmp(label, "camera_msg") == 0 \
		        || strcmp(label, "debug_sig----1") == 0 )
		{
			strcpy(tmp_str, str);
			strcpy(str, time_label);
			strcat(str, title_label);
			strcat(str, line_label);
			strcat(str, tmp_str);

			strcat(str, "\n");
			if (!strcmp(label, "stack")) {
				print_stacktrace();
			}
		} else {
			memset(str, 0, BUFSIZE);
//	sprintf(str, "ignoring unknown label [%s]\n", label);
		}

}
inline void print_str(char *str) {
	printf("%s", str);
	fflush(NULL);
}
#define prt(label,... ) {\
        unsigned char level = 0; \
        char str_title[20] = {0}; \
        sprintf(str_title, "%s", #label); \
        if( !strcmp(str_title, "debug")){ \
            level = 2; \
        } \
        if( !strcmp(str_title, "info")){ \
            level = 3; \
        } \
        if(!strcmp(str_title, "warning")){ \
            level = 4; \
        } \
        if(!strcmp(str_title, "err")){ \
            level = 9; \
        } \
        if (level >= prt_log_leve) {   \
    		char tmp_string[BUFSIZE] ;	\
    		sprintf(tmp_string,__VA_ARGS__);	\
    	    add_title(#label,tmp_string,__LINE__,__FILE__);  	\
    	    print_str(tmp_string);   \
    	    log_file(NULL,tmp_string);\
        } \
}

//void inline print_config(...)
//{
//	char label[BUFSIZE] ={"[config]"};
//	char t_string1[BUFSIZE] ;
//	char t_string2[BUFSIZE] ;
//	sprintf(t_string1,__VA_ARGS__);
//	sprintf(t_string2,"................%s(%d)\n", __FILE__,__LINE__);
//	strcat(label,t_string1);
//	strcat(label,t_string2);
//	printf("%s",label);
//	log_file(NULL,label);
//	fflush(NULL) ;
//}

#else

#define print_info(...) { \
printf("[ info ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}

#define print_err(...) { \
printf("[ err ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}


#define print_cmd(...) { \
printf("[ cmd ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}
//#define print_net(...)
#define print_net(...) { \
printf("[ net ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}


#define print_config(...) { \
printf("[ config ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}
//#define print_alg(...)
#define print_alg(...) { \
printf("[ alg ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}

#define print_sig(...) { \
printf("[ sig ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}

#define print_change(...) { \
printf("[ state change ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}
#define print_decoder(...) { \
printf("[ decoder ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;}


#define print(label,...) { \
char t_string[100] ; \
printf("[ %s ]:",#label);\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ;\
}

#define print_stack(...) { \
printf("[ stack ]:");\
printf(__VA_ARGS__);	\
printf(".................%s(%d)\n", __FILE__,__LINE__); \
fflush(NULL) ; print_stacktrace();\
}

#endif

typedef void *(*THREAD_ENTITY)(void*);
int create_detach_thread(THREAD_ENTITY callback, int level, void *data);
int create_joinable_thread(THREAD_ENTITY callback, int level, void *data);
void init_sig(void);

inline void print_stacktrace()
{
	int size = 16;
	void * array[16];
	char tmp[BUFSIZE * 2];
	int stack_num = backtrace(array, size);
	char ** stacktrace = backtrace_symbols(array, stack_num);
	if (stacktrace == NULL)
		return;
	log_file(NULL, "================dump stack begin===================\n");
	for (int i = 0; i < stack_num; ++i) {
		if (strlen( stacktrace[i]) > 100)
			continue;
		sprintf(tmp, "@@@@@@@@@@@%s@@@@@@@@@@@ \n", stacktrace[i]);
		log_file(NULL, tmp);
		printf("%s", tmp);
		fflush(NULL);
	}
	if (stacktrace != NULL)
		free(stacktrace);
	log_file(NULL, "================dump stack done================\n");

}
inline void exit_handle()
{
	prt(exit, "program call exited");
}
//inline void dump_cam_ip(char* ip)
//{
//	static unsigned char tmp[17];
//	memcpy(tmp, ip, 16);
//	prt(net,"ip %s", tmp);
//}

//inline char* cam_ip(char* ip)
//{
//	static  char tmp[17];
//	memcpy(tmp, ip, 16);
//	return tmp;
//}

//inline void dump_ip(unsigned char* ip)
//{
//	unsigned char tmp[17];
//	memcpy(tmp, ip, 16);
//	prt(info,"ip %s", tmp);
//}
//inline void exit_program()
//{
//	prt(exit, "program call exit");
//	print_stacktrace();
// 	exit(1);
//}

#define exit_program()\
{\
	prt(exit, "program call exit");\
	print_stacktrace();\
	exit(1);\
}


int regist_timed_callback(m_timed_func_data *p);
void unregist_timed_callback(pthread_t p);



m_timed_func_data *regist_timed_func(int time_us, void *ptr, void *data);
pthread_t start_timed_func(m_timed_func_data *p_ctx);
void stop_timed_func(m_timed_func_data *p_ctx);

//m_timed_func_data *regist_delayed_func(int time_us,void *ptr,void *data);
pthread_t start_delayed_func(void *func, void* data, int delay_ms);
//void stop_delayed_func(pthread_t p);

//m_timed_func_data *regist_detached_func(int time_us,void *ptr,void *data);
pthread_t start_detached_func(void *func, void* data);
//void stop_detached_func(pthread_t p);
typedef struct node {
	struct node *next;
	struct node *pre;
	void *data;
} m_node;
typedef struct list_info {
	m_node *head;
	m_node *tail;
	m_node *cur;//����seek
	int number;
	int data_size;
	p_func data_match_function;
	pthread_mutex_t list_lock;
} m_list_info;
////////////////////////////////////////////////////////////////////////
extern int setup_sig();
int  get_random_port();
m_list_info *new_list(int data_size, void *func);
void delete_list(m_list_info *p_info);
void list_node_alloc_tail(m_list_info *info_p);
static void list_node_free_tail(m_list_info *info_p);
int list_node_seek(m_list_info *info_p, void *data);
void * list_get_current_data(m_list_info *info_p);
void list_overwirte_current_data(m_list_info *info_p, void *data);
int list_operate_node_all(m_list_info *info_p, p_func func, void *arg);
int list_node_del_cur(m_list_info *info_p);
//
int  get_random_port();

void  init_variable();
void handle_log_file();
void delete_log_files(char *dirname, int left);
void set_serial(int sno, unsigned int buadrate, int databit, int stopbit, int checkbit);

bool change_sig_ip(char *ip, unsigned int port, char *ip2, unsigned int port2);

int set_ip_dns(char *ip, char *mask, char *dns1, char *dns2, char *gateway);
void kill_process();
void reboot_system();
//初始化全局�?
void init_global_lock();
void *start_ntpclient(void *arg);
int  exec_ssh_cmd(const char *ip, const char * user, const char * psswd, const char * cmd);
////
bool time_out();
long long get_ms();
void get_date_ms(char *str_date);
void get_uid(char *uid);
///////////////////////////////////////////////////////////////////////////
typedef struct ora_record {
	char pass_id[50];
	char pic_path[400];
} ora_record_t;
///
// int list_operate_node_cmd(m_list_info *p_info, p_func func, void *arg);

#endif
