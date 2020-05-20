/*
 * camera_service.h
 *
 *  Created on: 2016��9��9��
 *      Author: root
 */

#ifndef CAMERA_SERVICE_H_
#define CAMERA_SERVICE_H_
#include <pthread.h>
#include "g_define.h"
#include "cam_net.h"
#include "camera_rtsp.h"
#include "queue.h"
#include "cam_alg.h"

enum {
	CAM_RUNNING,
	CAM_STOPED
};
//现在状态
enum {
	CAM_RESET_NONE = 0,
	CAM_COMMAND,
	CAM_OPENING,
	CAM_OPENED_FAILED,
	CAM_FRAMING,
	CAM_FRAME_FAILED
};

enum {
	CAMERA_CONTROL_NULL,
	CAMERA_CONTROL_OPEN,
	CAMERA_CONTROL_CLOSE,
	CAMERA_CONTROL_RESET,
	CAMERA_CONTROL_SET_NTP,
	CAMERA_CONTROL_RESET_ALG,
	CAMERA_CONTROL_REPLAY
};

typedef struct tg_x_y{
    int id;
    int x;
    int y;
	long long stay_ms;
    double yaw_angle;
};

typedef struct camera_info {
	unsigned char exit_flag;
	int updated;
	int index;
	unsigned short gpu_index;
	unsigned char vtype;
//	int stop_run;//TODO: lock this
	pthread_mutex_t run_lock;
	int open_flg;
	unsigned char open_alg;
	unsigned char close_flag;
	int cam_running_state;
	int cam_pre_state;
	unsigned char cam_no_work_cnt;
	unsigned char fuzzy_flag;
	unsigned char visible_flag;
	
	pthread_mutex_t open_one_lock;
	int cmd;
	unsigned char ip[16];
	unsigned char name[16];
	unsigned char passwd[16];
	int port;
	pthread_mutex_t cmd_lock;
	m_cam_context * p_cam;
#ifdef USE_FILE
	m_h264_file_common h264_file_common;
#endif
#ifdef PLAY_BACK
	m_gl_common gl_common;
#endif
	m_timed_func_data *p_data;
	unsigned char *oubuf;
	unsigned char *oubufu;
	unsigned char *oubufv;
	int watchdog_value;
	int watchdog_frames;
	int watchdog_alg_frames;
	unsigned int run_time;
	unsigned int all_frames;
	unsigned int all_times;
	unsigned int no_frame_num;
	unsigned char zero_frame_cnt;

	long long per_second;
	/*
	unsigned char ybuf[640*480];
	unsigned char ubuf[640*480/4];
	unsigned char vbuf[640*480/4];
	*/
	/*
	unsigned char ybuf[FRAME_COLS*FRAME_ROWS];
	unsigned char ubuf[FRAME_COLS*FRAME_ROWS/4];
	unsigned char vbuf[FRAME_COLS*FRAME_ROWS/4];
	*/
	/*
	unsigned char alg_ybuf[FRAME_COLS*FRAME_ROWS];
	unsigned char alg_ubuf[FRAME_COLS*FRAME_ROWS/4];
	unsigned char alg_vbuf[FRAME_COLS*FRAME_ROWS/4];
	*/
	unsigned char *pybuf;
	unsigned char *pubuf;
	unsigned char *pvbuf;

	unsigned char *alg_ybuf;
	unsigned char *alg_ubuf;
	unsigned char *alg_vbuf;

	unsigned short frame_w;
	unsigned short frame_h;
	unsigned char  have_frame_size;

	pthread_mutex_t frame_lock;
	//pthread_mutex_t frame_lock_ex;
	pthread_t process_thread;
	pthread_t rtsp_thread;
	//file
	unsigned int     file_frame_num;
	unsigned int     file_frame_curr_no;
	unsigned char    file_start;
	unsigned char    file_replay;
	unsigned int     frame_no;
	pthread_mutex_t  file_lock;
	//event
	Queue *p_5s_frame_queue; //帧列表
	Queue *p_f_e_queue; //事件列表
	Queue *p_file_queue; //写视频列表
	//pthread_t watchdog_thread;
	//
	Queue *p_radar;
    tg_x_y vec_x_y[TARGET_MAX_NUM];
    //tg_x_y ps_x_y[200];
    int tg_num;
    bool is_full;
    //int ps_tg_num;
    //bool ps_is_full;
	//
	char pass_id[50];
	//
	unsigned char *pYuvBuf;
	//
	unsigned char curr_stat; //是否正在相机启动状态
	int cmd_fd;
	unsigned char rtsp_fag;//exit while flag
} m_camera_info;

#define DATA_SIZE 3
typedef struct yuv_list {
	/*
		unsigned char ybuf[DATA_SIZE][FRAME_COLS*FRAME_ROWS];
	    unsigned char ubuf[DATA_SIZE][FRAME_COLS*FRAME_ROWS/4];
	    unsigned char vbuf[DATA_SIZE][FRAME_COLS*FRAME_ROWS/4];
	*/
	unsigned char *pybuf[DATA_SIZE];
	unsigned char *pubuf[DATA_SIZE];
	unsigned char *pvbuf[DATA_SIZE];
	unsigned char valid[DATA_SIZE];
	int item_wloop[DATA_SIZE];
	unsigned short i_read;
	unsigned short i_write;
	int i_wloop;
	unsigned int frame_no[DATA_SIZE];
	unsigned short total_num;
	//unsigned int   frame_no;
	pthread_mutex_t data_lock;
} yuv_list_t;

typedef struct ffmpeg_object_type {
	DecodeContext    decode;
	AVCodecContext  *decoder_ctx;
	AVFormatContext *input_ctx;
	SwsContext      *img_convert_ctx;
	AVStream        *video_st;
	AVFrame  		*yuv_frame;
	AVFrame *frame;
	AVFrame *sw_frame;
	int video_index;
	//pthread_mutex_t data_lock;
} ffmpeg_object_type_t;

typedef struct radar_object_type { //雷达对象类型
	unsigned char obj_num;
	void *p_obj;
} radar_object_t;

//
//#include "cam_net.h"
//
//#include "cam_codec.h"
//#include "glplayer.h"
//#include "h264_stream_file.h"
///*
//
// * ����һ�������ȫ����Դ������Ϣ
// * */
//typedef struct camera_info {
//	int index;
//	int stop_run;//TODO: lock this
//	int open_flg;
//	int cam_state;
//	int cmd;
//	unsigned char ip[16];
//	unsigned char name[16];
//	unsigned char passwd[16];
//	int port;
//	pthread_mutex_t cmd_lock;
//	m_cam_context * p_cam;
//
//#ifdef USE_FILE
//	m_h264_file_common h264_file_common;
//#endif
//#ifdef PLAY_BACK
//	m_gl_common gl_common;
//#endif
////	m_codec_common codec_common;
//	m_timed_func_data *p_data;
//	unsigned char *oubuf;
//	unsigned char *oubufu;
//	unsigned char *oubufv;
//} m_camera_info;
//

void camera_service_init();
int camera_ctrl(int cmd, int index, int blocked, void *data);
int camera_open(int index, char ip[], int port, char username[], char passwd[], unsigned char vtype);
int camera_close(int index, bool is_call_cmd);
int get_cam_running_state(int index);
//
void get_camera_network(int index, void *cam_info);
bool set_camera_network(int index, char *ip, char *name, char *passwd, int port);
//
void init_watchdog();
void *watchdog_func(void *data);
//
void init_yuv_list();
void init_yuv_list_arrary(unsigned char index, unsigned short f_width, unsigned short f_height);

bool get_yuv_item(int index, unsigned char *p_ybuf, unsigned char *p_ubuf, unsigned char *p_vbuf, unsigned short f_width, unsigned f_height, unsigned int &frame_no);
void set_yuv_item(int index, unsigned char *p_ybuf, unsigned char *p_ubuf, unsigned char *p_vbuf, unsigned short f_width, unsigned f_height, unsigned int frame_no);
void init_cam_info_yuv(unsigned char index, unsigned short f_width, unsigned short f_height);
void deinit_cam_info_yuv(unsigned char index);
void clear_yuv_item(int index);
//
bool camera_set_open_one(int index);
void camera_reset_open_one(int index);
//
void *process_fun_for_picture(void* data);
//
void *process_client_picture(void* data);
//
void camera_set_curr_status(int index, int status);

#endif /* CAMERA_SERVICE_H_ */
