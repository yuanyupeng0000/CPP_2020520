#include <sys/socket.h>
#include <sys/time.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include "camera_service.h"
#include "cam_net.h"
#include "cam_alg.h"
#include "cam_net.h"
#include "cam_codec.h"
#include "glplayer.h"
#include "client_net.h"
#include "camera_service.h"
#include "ini/fvdconfig.h"
#include "h264_stream_file.h"
#include "common.h"
#include "g_fun.h"
#include "g_define.h"
#include "camera_rtsp.h"
#include "sig_service.h"
#include "url_downfile.h"
#include "fvdmysql/fvdoracle.h"
#include "ptserver/non_motor_server.h"

/*

 * ??????????????????????????
 * */
extern int  sig_fd;
extern int stoped;
extern IVDDevSets g_ivddevsets;
extern IVDCAMERANUM    g_cam_num;
extern mCamDetectParam g_camdetect[CAM_MAX];
extern m_camera_info cam_info[CAM_MAX];
extern mGlobalVAR g_member;
////////////////////
ora_record_t ora_rds[ORACLE_RECORD_MAX] = {0};
///////////////////////////

unsigned char  g_gpu_index[GPU_INDEX_MAX] = {0};
pthread_mutex_t gpu_lock;

#if (DECODE_TYPE == 2)
ffmpeg_object_type_t g_ffmpeg_obj[CAM_MAX];
#endif


enum {
	DOG_HUNGRY_STATE,
	DOG_FULL_STATE
};


yuv_list_t cam_yuv_list[CAM_MAX];

void init_yuv_list()
{
	memset(cam_yuv_list, 0, sizeof(yuv_list_t)*CAM_MAX);

	for (int i = 0; i < CAM_MAX; i++) {
		pthread_mutex_init(&cam_yuv_list[i].data_lock, NULL);
	}
}

void deinit_yuv_list(unsigned char index)
{
	for (int i = 0; i < DATA_SIZE; i++) {
		if ( cam_yuv_list[index].pybuf[i] ) {
			free(cam_yuv_list[index].pybuf[i]);
			cam_yuv_list[index].pybuf[i] = NULL;
		}

		if ( cam_yuv_list[index].pubuf[i] ) {
			free(cam_yuv_list[index].pubuf[i]);
			cam_yuv_list[index].pubuf[i] = NULL;
		}

		if ( cam_yuv_list[index].pvbuf[i] ) {
			free(cam_yuv_list[index].pvbuf[i]);
			cam_yuv_list[index].pvbuf[i] = NULL;
		}

		cam_yuv_list[index].i_read  = 0;
		cam_yuv_list[index].i_write = 0;
	}
}

void init_yuv_list_arrary(unsigned char index, unsigned short f_width, unsigned short f_height)
{
	pthread_mutex_lock(&cam_yuv_list[index].data_lock);
	for (int i = 0; i < DATA_SIZE; i++) {
		cam_yuv_list[index].pybuf[i] = (unsigned char*)malloc((int)f_width * f_height);
		cam_yuv_list[index].pubuf[i] = (unsigned char*)malloc((int)f_width * f_height / 4);
		cam_yuv_list[index].pvbuf[i] = (unsigned char*)malloc((int)f_width * f_height / 4);
	}
	pthread_mutex_unlock(&cam_yuv_list[index].data_lock);
}

void init_cam_info_yuv(unsigned char index, unsigned short f_width, unsigned short f_height)
{
	cam_info[index].pybuf = (unsigned char*)malloc((int)f_width * f_height);
	cam_info[index].pubuf = (unsigned char*)malloc((int)f_width * f_height / 4);
	cam_info[index].pvbuf = (unsigned char*)malloc((int)f_width * f_height / 4);
}

void deinit_cam_info_yuv(unsigned char index)
{
	if (cam_info[index].pybuf) {
		free(cam_info[index].pybuf);
		cam_info[index].pybuf = NULL;
	}

	if (cam_info[index].pubuf) {
		free(cam_info[index].pubuf);
		cam_info[index].pubuf = NULL;
	}

	if (cam_info[index].pvbuf) {
		free(cam_info[index].pvbuf);
		cam_info[index].pvbuf = NULL;
	}
}

bool get_yuv_item(int index, unsigned char *p_ybuf, unsigned char *p_ubuf, unsigned char *p_vbuf, unsigned short f_width, unsigned f_height, unsigned int &frame_no)
{
	bool ret = false;
	pthread_mutex_lock(&cam_yuv_list[index].data_lock);
	int iread = cam_yuv_list[index].i_read;
	//if (cam_yuv_list[index].total_num > 0 && cam_yuv_list[index].valid[iread] > 0) {
	if (cam_yuv_list[index].valid[iread] > 0 && (cam_yuv_list[index].item_wloop[iread] - cam_yuv_list[index].i_wloop) != 1 ) {
		memcpy(p_ybuf, cam_yuv_list[index].pybuf[iread], f_width * f_height);
		memcpy(p_ubuf, cam_yuv_list[index].pubuf[iread], f_width * f_height / 4);
		memcpy(p_vbuf, cam_yuv_list[index].pvbuf[iread], f_width * f_height / 4);
		frame_no = cam_yuv_list[index].frame_no[iread];
		cam_yuv_list[index].valid[iread] = 0;
		cam_yuv_list[index].i_read = (iread + 1) % DATA_SIZE;
		//cam_yuv_list[index].total_num--;
		ret = true;
	}

	pthread_mutex_unlock(&cam_yuv_list[index].data_lock);

	return ret;
}

void set_yuv_item(int index, unsigned char *p_ybuf, unsigned char *p_ubuf, unsigned char *p_vbuf, unsigned short f_width, unsigned f_height, unsigned int frame_no)
{
	pthread_mutex_lock(&cam_yuv_list[index].data_lock);
	//if (cam_yuv_list[index].total_num < DATA_SIZE) {
	int iwrite = cam_yuv_list[index].i_write;
	memcpy(cam_yuv_list[index].pybuf[iwrite], p_ybuf, f_width * f_height);
	memcpy(cam_yuv_list[index].pubuf[iwrite], p_ubuf, f_width * f_height / 4);
	memcpy(cam_yuv_list[index].pvbuf[iwrite], p_vbuf, f_width * f_height / 4);
	cam_yuv_list[index].frame_no[iwrite] = frame_no;
	cam_yuv_list[index].item_wloop[iwrite] = cam_yuv_list[index].i_wloop;
	cam_yuv_list[index].valid[iwrite] = 1;
	cam_yuv_list[index].i_write = (iwrite + 1) % DATA_SIZE;
	if (0 == cam_yuv_list[index].i_write)
		cam_yuv_list[index].i_wloop++;

	cam_yuv_list[index].i_wloop = (cam_yuv_list[index].i_wloop + 1) % 100; //100è½®ï¼Œè¿™ä¸ªä¸»è¦ç”¨æ¥æŽ§åˆ¶å½“è¯»å†™äº¤æ›¿æ—¶ï¼Œè¯»åˆ°æ–°å¸§çš„æ—¶å€™ï¼Œåˆè¯»åˆ°äº†æ—§å¸§

	/*
	cam_yuv_list[index].total_num++;
	if (cam_yuv_list[index].total_num > DATA_SIZE) {
		cam_yuv_list[index].total_num = DATA_SIZE;
	}
	*/
	//}
	pthread_mutex_unlock(&cam_yuv_list[index].data_lock);
}

void clear_yuv_item(int index)
{
	pthread_mutex_lock(&cam_yuv_list[index].data_lock);
	cam_yuv_list[index].total_num = 0;
	for (int i = 0; i < DATA_SIZE; i++) {
		cam_yuv_list[index].valid[i] = 0;
	}
	pthread_mutex_unlock(&cam_yuv_list[index].data_lock);

}

#if 0
void *move_pic(void *data)
{
	m_camera_info *p_info = (m_camera_info*)data;
	usleep(20);
	//my_mutex_lock(&p_info->frame_lock);
	pthread_cleanup_push(my_mutex_clean, &p_info->frame_lock);
	pthread_mutex_lock(&p_info->frame_lock);
	prt(info, "move_pic lock.........")
	//   prt(info,"--------------------------------------a-------------------------------->send out");

	run_alg(p_info->index, p_info->ybuf, p_info->ubuf, p_info->vbuf, 0);
	memcpy(p_info->ybuf, p_info->oubuf, FRAME_COLS * FRAME_ROWS);
	memcpy(p_info->ubuf, p_info->oubufu, FRAME_COLS * FRAME_ROWS / 4);
	memcpy(p_info->vbuf, p_info->oubufv, FRAME_COLS * FRAME_ROWS / 4);
	//  prt(info,"--------------------------------------b-------------------------------->send out");
	//my_mutex_unlock(&p_info->frame_lock);
	pthread_mutex_unlock(&p_info->frame_lock);
	pthread_cleanup_pop(0);
	prt(info, "move_pic unlock.........")
	usleep(20);
}

#endif
//#define CAMERA_NUMBER 4
#define CHECK_DURATION 500000//US
#define WATCHDOG_CHECK_DURATION 1000*1000 //1000*1000*10//10s
#define FRAME_DURATION 100000//US
m_camera_info cam_info[CAM_MAX];
/* Function Description
 * name:
 * return:
 * args:
 * comment:???????????????????§Ý??????????
 * todo:
 */

#if 0
void *process_fun(int index, char *data, int size)
{
	FRAME_HEAD	*pFrameHead = (FRAME_HEAD *)data;
	EXT_FRAME_HEAD *exfream = (EXT_FRAME_HEAD *)(data + sizeof(FRAME_HEAD));

	if (FRAME_FLAG_A == pFrameHead->streamFlag)
	{
		return NULL;
	}

	int ret = 0;

	m_camera_info *p_info = &cam_info[index];
	p_info->watchdog_value = DOG_FULL_STATE;
	data = data + sizeof(FRAME_HEAD) + sizeof(EXT_FRAME_HEAD);

	pthread_cleanup_push(my_mutex_clean, &p_info->frame_lock);
	if (0 == pthread_mutex_trylock(&p_info->frame_lock)) {
		h264_decode(index, data - 100, size, (unsigned char**)&p_info->alg_ybuf, (unsigned char**)&p_info->alg_ubuf, (unsigned char**)&p_info->alg_vbuf);
		p_info->updated = 1;
		pthread_mutex_unlock(&p_info->frame_lock);
	}
	p_info->watchdog_frames++;
	pthread_cleanup_pop(0);
	cam_info[index].frame_num = 0;

	usleep(30000);

}
#endif
void *process_fun_for_rtsp(void* data)
{
	Mat fm;
	int frame_rate = 25;
	unsigned char one_change = 0;
	int index = *(int*)data;
	VideoCapture cap;
	m_camera_info *p_info = &cam_info[index];
	char file_url[200] = {0};
	unsigned char *pYuvBuf = NULL;
	struct timeval s_tv, e_tv; //, tst_tv;
	memset(&s_tv, 0, sizeof(struct timeval));

	prt(info, "process_fun_for_rtsp");
	//#if (DECODE_TYPE == 2)
	//	memset(&g_ffmpeg_obj[index], 0, sizeof(ffmpeg_object_type_t) - sizeof(pthread_mutex_t));
	//#endif

	//-------test
#if 0
	time_t now = time(NULL);////»ñÈ¡1970.1.1ÖÁµ±Ç°ÃëÊýtime_t
	struct tm * timeinfo = localtime(&now); //´´½¨TimeDate,²¢×ª»¯Îªµ±µØÊ±¼ä
	char path[60];
	strftime(path, 60, "%Y_%m_%d_%H_%M_%S", timeinfo);
	char strPath[100];
	sprintf(strPath, "camera%d_%s.avi", index, path);//½«´´½¨ÎÄ¼þµÄÃüÁî´æÈëcmdcharÖÐ
	VideoWriter writer;
	writer.open(strPath, CV_FOURCC('X', 'V', 'I', 'D'), 25, Size(640, 480), true);//Size(640, 480)//Size(frame.rows, frame.cols)//"cam.avi"
#endif
	//--------end test

	cmd_output(index, SETALGPARAM);

	while (!p_info->close_flag) {
		//pthread_testcancel();
		if (!p_info->open_flg || p_info->file_replay || p_info->no_frame_num > 20 ) { //p_info->no_frame_num++
			char *p_url = NULL;
			if (FILE_TYPE == p_info->vtype) {
				p_url = strrchr(g_camdetect[index].other.filePath, '\\');
				if (p_url != NULL) {
					sprintf(file_url, "%s%s\0", FILE_BASE_DIR, p_url + 1);
					p_url = file_url;
				}
			} else {
				p_url = g_camdetect[index].other.rtsppath;
			}
			
			if (NULL == p_url || strlen(p_url) < 1) {
				
				// if (!one_change && CAM_RESETING == p_info->reset_status) {
				// 	one_change = 1;
				// 	camera_reset_open_one(index);
				// }
				camera_set_curr_status(index, CAM_OPENED_FAILED);
				sleep(1);
				continue;
			}

#if (DECODE_TYPE == 1)
			//if (!open_camera("rtsp://10.10.10.12:554/av0_1", cap)){
			if (!open_camera(p_url, cap)) {
				//cmd_output(index, SETALGPARAM);
				camera_set_curr_status(index, CAM_OPENED_FAILED); 
				prt(info, "camera[%d]: %s open faild. \n", index, p_url);
				sleep(1);
				// if (!one_change && CAM_RESETING == p_info->reset_status) {
				// 	one_change = 1;
				// 	camera_reset_open_one(index);
				// }
				continue;
			}

			if (FILE_TYPE == p_info->vtype) {
				p_info->file_frame_curr_no = 0;
				p_info->file_frame_num = cap.get(CV_CAP_PROP_FRAME_COUNT);
				pthread_mutex_lock(&p_info->file_lock);
				p_info->file_replay = 0;
				pthread_mutex_unlock(&p_info->file_lock);
				frame_rate = cap.get(CV_CAP_PROP_FPS);
				if (frame_rate < 1 || frame_rate > 50)
					frame_rate = 25;
			}
#elif (DECODE_TYPE == 2)
			//clear_frame(g_ffmpeg_obj[index].yuv_frame);
			g_ffmpeg_obj[index].video_st = NULL;
			clear_ffmpeg(&g_ffmpeg_obj[index].input_ctx, &g_ffmpeg_obj[index].decoder_ctx, &g_ffmpeg_obj[index].decode, &g_ffmpeg_obj[index].img_convert_ctx);
			if ( open_ffmpeg(p_url, &g_ffmpeg_obj[index].input_ctx, &g_ffmpeg_obj[index].decoder_ctx, g_ffmpeg_obj[index].decode, &g_ffmpeg_obj[index].video_st, g_ffmpeg_obj[index].video_index) < 0) {
				// if (!one_change && CAM_RESETING == p_info->reset_status) {
				// 	one_change = 1;
				// 	camera_reset_open_one(index);
				// }
				camera_set_curr_status(index, CAM_OPENED_FAILED); 
				printf("open_ffmpeg fail.\n");
				sleep(1);
				continue;
			}

#endif
			prt(info, "process_fun_for_rtsp2");
			p_info->open_flg = 1;
			p_info->no_frame_num = 0;
			//cmd_output(index, SETALGPARAM);
			camera_set_curr_status(index, CAM_FRAMING); 
			sleep(1);
		}

		

#if (DECODE_TYPE == 1)

		if (p_info->file_start && p_info->file_frame_curr_no >= p_info->file_frame_num) {
			pthread_mutex_lock(&cam_info[index].file_lock);
			cam_info[index].file_replay = 1;
			pthread_mutex_unlock(&cam_info[index].file_lock);
			continue;
		}

		if (FILE_TYPE == p_info->vtype && (!p_info->file_start) ) { // || p_info->file_frame_curr_no >= p_info->file_frame_num) ) {
			sleep(1);

			// if (!one_change && CAM_RESETING == p_info->reset_status) {
			// 	one_change = 1;
			// 	camera_reset_open_one(index);
			// }

			continue;
		}

		gettimeofday(&e_tv,NULL);
		if (FILE_TYPE == p_info->vtype){ //file play by frame_rate 
			if ((e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000  < 1000/frame_rate)
				continue;
			// prt(info, "..........camera[%d] frame time: %ld", index, ((e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000 ));
			gettimeofday(&s_tv,NULL);
		}
		
		// gettimeofday(&tst_tv,NULL);
		if (!get_frame(fm, cap) ) {
			camera_set_curr_status(index, CAM_FRAME_FAILED); 
			usleep(2000);
			p_info->no_frame_num++;
			continue;
		}

		// if (RTSP_TYPE == p_info->vtype) {
		// 	if ((e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000  < 1000/40) //40 frame/per
		// 		continue;
		// 	prt(debug, "..........camera[%d] frame time: %ld", index, ((e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000 ));
		// 	gettimeofday(&s_tv,NULL);
		// }
		// gettimeofday(&e_tv,NULL);
		p_info->no_frame_num = 0;

		// prt(info, "100000-camera[%d] frame time: %ld", index, (e_tv.tv_sec - tst_tv.tv_sec)*1000 + (e_tv.tv_usec - tst_tv.tv_usec)/1000 );
		if ((fm.cols == 0 || fm.rows == 0) ) {
			prt(debug, "frame is cols 0");
			continue;
		}

		p_info->file_frame_curr_no++;
		
		// int tm_cnt = (e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000;
		// int tm_stand = 1000/FRAME_RATE;
	
		// if ((e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000  < 1000/FRAME_RATE)
		// 	continue;
		// prt(info, "..........camera[%d] frame time: %ld", index, ((e_tv.tv_sec - s_tv.tv_sec)*1000 + (e_tv.tv_usec - s_tv.tv_usec)/1000 ));
		// gettimeofday(&s_tv,NULL);
	
		p_info->watchdog_value = DOG_FULL_STATE;

		Mat yuvImg;
		int fm_size = (int)fm.cols * fm.rows;
		// pthread_mutex_lock(&p_info->cmd_lock);
		if (!p_info->have_frame_size) {  //åˆå§‹åŒ–yuvæ•°ç»„
			p_info->frame_w =  fm.cols;
			p_info->frame_h = fm.rows;
			init_cam_info_yuv(index, fm.cols, fm.rows);
			init_yuv_list_arrary(index, fm.cols, fm.rows);
			p_info->have_frame_size = 1;
			pYuvBuf = (unsigned char *)malloc(fm_size / 2 * 3);
		}
		
		cvtColor(fm, yuvImg, CV_BGR2YUV_I420);
		memcpy(pYuvBuf, yuvImg.data, fm_size / 2 * 3);
		set_yuv_item(index, pYuvBuf, pYuvBuf + fm_size, pYuvBuf + fm_size + fm_size / 4, fm.cols, fm.rows, p_info->file_frame_curr_no);
		yuvImg.release();
		//set_yuv_item(index, pYuvBuf, pYuvBuf + frame_size, pYuvBuf + frame_size + frame_uv_size, p_info->file_frame_curr_no);

#endif

#if (DECODE_TYPE == 2)
		if (g_ffmpeg_obj[index].input_ctx == NULL || g_ffmpeg_obj[index].decoder_ctx == NULL) {
			sleep(1);
			continue;
		}

		if ( !get_frame_ffmpeg(g_ffmpeg_obj[index].input_ctx, g_ffmpeg_obj[index].decoder_ctx, \
		                       &g_ffmpeg_obj[index].decode, g_ffmpeg_obj[index].video_st, &g_ffmpeg_obj[index].img_convert_ctx, \
		                       &g_ffmpeg_obj[index].yuv_frame, &g_ffmpeg_obj[index].frame, &g_ffmpeg_obj[index].sw_frame, index, g_ffmpeg_obj[index].video_index) )
			continue;


		if (NULL != g_ffmpeg_obj[index].yuv_frame->data && NULL != g_ffmpeg_obj[index].yuv_frame->data[0]) {
			if (!p_info->have_frame_size) {
				p_info->frame_w =  g_ffmpeg_obj[index].yuv_frame->width;
				p_info->frame_h = g_ffmpeg_obj[index].yuv_frame->height;
				init_cam_info_yuv(index, g_ffmpeg_obj[index].yuv_frame->width, g_ffmpeg_obj[index].yuv_frame->height);
				init_yuv_list_arrary(index, g_ffmpeg_obj[index].yuv_frame->width, g_ffmpeg_obj[index].yuv_frame->height);
				p_info->have_frame_size = 1;
			}
			set_yuv_item(index, g_ffmpeg_obj[index].yuv_frame->data[0], g_ffmpeg_obj[index].yuv_frame->data[1], g_ffmpeg_obj[index].yuv_frame->data[2], g_ffmpeg_obj[index].yuv_frame->width, g_ffmpeg_obj[index].yuv_frame->height, 0);
			//av_frame_unref(g_ffmpeg_obj[index].yuv_frame);
		}

#endif



#if 0
		memcpy(p_info->ybuf, pYuvBuf, frame_size);
		memcpy(p_info->ubuf, pYuvBuf + frame_size, frame_uv_size);
		memcpy(p_info->vbuf, pYuvBuf + frame_size + frame_uv_size, frame_uv_size);
#endif

#if 0
		pthread_cleanup_push(my_mutex_clean, &p_info->frame_lock);

		// pthread_mutex_lock(&p_info->frame_lock);
		if (0 == pthread_mutex_trylock(&p_info->frame_lock)) {
			//prt(info,"cam[%d] for rtsp lock", index);
			//  pthread_mutex_lock(&p_info->frame_lock);
#if 0
			if (!p_info->updated) {
				pthread_mutex_lock(&p_info->frame_lock);
				//prt(info, "process_fun_for_rtsp lock.........")
				memcpy(p_info->alg_ybuf, p_info->ybuf, frame_size);
				memcpy(p_info->alg_ubuf, p_info->ubuf, frame_uv_size);
				memcpy(p_info->alg_vbuf, p_info->vbuf, frame_uv_size);
				p_info->updated = 1;
				//writer.write(fm);
				// imwrite("test.jpg", fm);
				pthread_mutex_unlock(&p_info->frame_lock);
			}
#endif

			memcpy(p_info->ybuf, pYuvBuf, frame_size);
			memcpy(p_info->ubuf, pYuvBuf + frame_size, frame_uv_size);
			memcpy(p_info->vbuf, pYuvBuf + frame_size + frame_uv_size, frame_uv_size);
			p_info->alg_ybuf = p_info->ybuf;
			p_info->alg_ubuf = p_info->ubuf;
			p_info->alg_vbuf = p_info->vbuf;
			p_info->updated = 1;
			pthread_mutex_unlock(&p_info->frame_lock);
			p_info->watchdog_frames++;

		}

		pthread_cleanup_pop(0);
#endif
		p_info->watchdog_value = DOG_FULL_STATE;
		p_info->watchdog_frames++;
		// p_info->no_frame_num = 0;

		// if (!one_change && CAM_RESETING == p_info->reset_status) {
		// 	one_change = 1;
		// 	camera_reset_open_one(index);
		// }

		// pthread_mutex_unlock(&p_info->cmd_lock);
		if (RTSP_TYPE == p_info->vtype) 
			usleep(30000);

	}

	p_info->rtsp_fag = 1;

	print("cam[%d]: exit while by close_flag \n", index);
}


int get_cmd(int index)
{
	return cam_info[index].cmd;
}

void set_cmd(int cmd, int index)
{
	cam_info[index].cmd = cmd;
}

#if 0
void *file_process_fun(int *index, char *data, int size)
{
	m_camera_info *p_info = (m_camera_info *) data;
	process_fun(*index, data, size);
//	start_detached_func((void *)real_process_fun,data);
}
#endif

long long start_ms[5] = {0};
long long end_ms[5] = {0};
long long end_ms2[5] = {0};

void *start_process(void *data)
{
	unsigned int frame_no = 0;
	m_camera_info *p_info = (m_camera_info*)data;
	//while(!p_info->exit_flag) {
	while (!p_info->close_flag) {
		//start_ms[p_info->index] = get_ms();

		if (!p_info->open_alg) {
			sleep(1);
			//prt(info, "start_process: %d not open_alg", p_info->index);
			continue;
		}

#if 0
		pthread_cleanup_push(my_mutex_clean, &p_info->frame_lock);
		pthread_mutex_lock(&p_info->frame_lock);

		if (p_info->updated == 1) {

			int bf_ms = get_ms();
			run_alg(p_info->index, p_info->alg_ybuf, p_info->alg_ubuf, p_info->alg_vbuf);
			int af_ms = get_ms();
			prt(info, "run_alg run time[%d]: %d", p_info->index, af_ms - bf_ms);
			p_info->run_time += (af_ms - bf_ms);
			p_info->all_times += (af_ms - bf_ms);
			p_info->all_frames++;

			p_info->watchdog_alg_frames++;
			p_info->updated = 0;
		}

		pthread_mutex_unlock(&p_info->frame_lock);
		pthread_cleanup_pop(0);
#endif
#if 0
		long long bf_ms = 0;
		long long af_ms = 0;

		long long bf_ms1 = get_ms();


		if (0 == p_info->per_second) {
			p_info->per_second = bf_ms1;
		}
#endif

		if (p_info->have_frame_size > 0 && get_yuv_item(p_info->index, p_info->pybuf, p_info->pubuf, p_info->pvbuf, p_info->frame_w, p_info->frame_h, p_info->frame_no) ) {

			pthread_cleanup_push(my_mutex_clean, &p_info->frame_lock);
			pthread_mutex_lock(&p_info->frame_lock);
			//bf_ms = get_ms();

			run_alg(p_info->index, p_info->pybuf, p_info->pubuf, p_info->pvbuf, p_info->frame_w, p_info->frame_h, p_info->frame_no);

			//af_ms = get_ms();
			//p_info->run_time += (af_ms-bf_ms);
			//p_info->all_frames++;
			p_info->watchdog_alg_frames++;
			pthread_mutex_unlock(&p_info->frame_lock);
			pthread_cleanup_pop(0);

			client_output(p_info->index);

			//end_ms2[p_info->index] = get_ms();

		} else {
			// if (FILE_TYPE == p_info->vtype) {
			// 	continue;
			// }
			usleep(10000);
			continue;
		}
#if 0
		int af_ms1 = get_ms();

		p_info->all_times += (af_ms1 - bf_ms1);
		if ( ( af_ms1 - bf_ms1 - (af_ms - bf_ms)) > 5)
			prt(info, "cam[%d] alg: %d process time: %d sa: %d *******************", p_info->index, af_ms - bf_ms,  af_ms1 - bf_ms1,  af_ms1 - bf_ms1 - (af_ms - bf_ms));

		if (p_info->per_second > 0 && (bf_ms1 - p_info->per_second) >= 1000 && p_info->watchdog_alg_frames > 0) {
			prt(info, "watchdog dog for cam %d , fp10s:%d / %d  alg ams: %d run time ams: %d list: %d",
			    p_info->index, p_info->watchdog_alg_frames, p_info->watchdog_frames,
			    p_info->all_times / p_info->watchdog_alg_frames,
			    p_info->all_times / p_info->watchdog_alg_frames, cam_yuv_list[p_info->index].total_num);

			p_info->per_second = 0;
			p_info->watchdog_alg_frames = 0;
			p_info->watchdog_frames = 0;
			p_info->run_time = 0;
			p_info->all_times = 0;
		}
#endif
		//prt(info, "cam[%d]: client_output start", p_info->index);
		//client_output(p_info->index);
		//prt(info, "cam[%d]: client_output end", p_info->index);
		//if (g_member.cmd_play && (g_ivddevsets.pro_type == PROTO_WS_RADAR || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) ) {
		//unsigned char send_data = 0xFF;
		//if (g_member.radar_sock > 0)
		//	send(g_member.radar_sock, &send_data, 1, 0); //å‘é€0xFFèŽ·å–é›·è¾¾æ•°æ®
		//}
		//usleep(5000);
		//usleep(10000);
		//end_ms[p_info->index] = get_ms();

		//prt(info, "run_time frame: %d end: %d", end_ms2[p_info->index] - start_ms[p_info->index], end_ms[p_info->index] - start_ms[p_info->index]);
	}
}

/* Function Description
 * name:
 * return:
 * args:
 * 		index?????????????????????????????????????????ctrl?? ??reset??
 * 		ip port username passwd
 * comment:
 * 		?????????????????????????????????????????????????????????
 * todo:
 */
int camera_open(int index, char ip[], int port, char username[], char passwd[], unsigned char vtype)
{
	int ret = -1;
	unsigned char open_flg = 0;

	prt(info, "camera_open");
	camera_set_curr_status(index, CAM_OPENING); 
	
	if (g_ivddevsets.pro_type == PROTO_WS_RADAR || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) {
		if (!cam_info[index].p_radar)
			cam_info[index].p_radar = InitQueue();// camera radar data queue
	}

	pthread_mutex_lock(&gpu_lock);
	for (int i = 0; i < GPU_INDEX_MAX; i++) {
		if (g_gpu_index[i] < GPU_ITEM_MAX) {
			g_gpu_index[i]++;
			cam_info[index].gpu_index = i;
			ret = i;
			break;
		}
	}
	pthread_mutex_unlock(&gpu_lock);

	if (ret < 0) {
		prt(info, "get gpu index failed.");
		return -1;
	}

	ret = -1;

	cam_info[index].index = index;
	memset(cam_info[index].ip, 0, 16);
	memset(cam_info[index].name, 0, 16);
	memset(cam_info[index].passwd, 0, 16);
	memcpy(cam_info[index].ip, ip, strlen(ip));
	memcpy(cam_info[index].name, username, strlen(username));
	memcpy(cam_info[index].passwd, passwd, strlen(passwd));
	cam_info[index].port = port;
	cam_info[index].updated = 0;
	cam_info[index].vtype = vtype;

	if (SDK_TYPE == vtype) {
		if ( (0 == open_flg ) && open_sdk()) {
			prt(info, "err in open sdk");
			return -1;
		} else {
			prt(info, "ok to open sdk");
			open_flg = 1;
		}
		camera_set_curr_status(index, CAM_FRAMING); 
		ret = 0;
	}
	else if (RTSP_TYPE == vtype || FILE_TYPE == vtype ) {
		pthread_create(&cam_info[index].rtsp_thread, NULL, process_fun_for_rtsp, (void *)&index);
		ret = 0;
	}
	else if (PICTURE_TYPE == vtype) {
		open_alg(index, cam_info[index].gpu_index);
#if ORACLE_VS == 1
		camera_set_curr_status(index, CAM_FRAMING); 
		pthread_create(&cam_info[index].rtsp_thread, NULL, process_fun_for_picture, (void *)&index);
#endif
#if RECEIVE_VS == 1
	camera_set_curr_status(index, CAM_FRAMING); 
	pthread_create(&cam_info[index].rtsp_thread, NULL, process_client_picture, (void *)&index);
#endif
		ret = 1;
	}

	if (0 == ret) {
		open_alg(index, cam_info[index].gpu_index);
		cam_info[index].open_alg = 1;
		cam_info[index].cam_running_state = CAM_RUNNING;
		prt(info, "cam_info[%d]: creating thread", index);
		pthread_create( &cam_info[index].process_thread, NULL, start_process, (void *)&cam_info[index]);

	} else {
		cam_info[index].cam_running_state = CAM_STOPED;
		cam_info[index].open_flg = 0;
		cam_info[index].open_alg = 0;
	}

	return 0;
}

int camera_close(int index, bool is_call_cmd)
{
	m_camera_info *p_info = (m_camera_info *) &cam_info[index];
	p_info->file_start = 0;
	// p_info->exit_flag = 1;
	int open_flg = p_info->open_flg;
	int open_alg = p_info->open_alg;
	p_info->close_flag = 1;

	p_info->open_alg = 0;
	p_info->open_flg = 0;
	p_info->updated = 0;

	pthread_mutex_lock(&gpu_lock);
	if (g_gpu_index[p_info->gpu_index] > 0)
		g_gpu_index[p_info->gpu_index]--;

	pthread_mutex_unlock(&gpu_lock);

	for ( int i = 0; i < 15 && !p_info->rtsp_fag; i++) {
		print( "camera_close go to sleeping \n");
		sleep(1); //µÈ´ýËøÊÍ·Å
	}
		

#if 0
	if (SDK_TYPE == p_info->vtype) {
		net_close_camera(index);
		//	close_camera(cam_info[index].p_cam);
		if (p_info->p_cam != NULL) {
			free(p_info->p_cam);
		}

		if (open_flg)
			close_h264_decoder(index);
	}
#endif

	printf("cam[%d] rtsp_thread: %u process_thread: %u \n", index, p_info->rtsp_thread
	    , p_info->process_thread);


	if ( p_info->rtsp_thread > 0) {
		pthread_cancel(p_info->rtsp_thread);
		pthread_join(p_info->rtsp_thread, NULL);
		memset(&p_info->rtsp_thread, 0, sizeof(p_info->rtsp_thread));
	}

	if (p_info->process_thread > 0) {
		pthread_cancel(p_info->process_thread);
		pthread_join(p_info->process_thread, NULL);
		memset(&p_info->process_thread, 0, sizeof(p_info->process_thread));
	}

	//unregist_timed_callback(cam_info[index].watchdog_thread);
	pthread_mutex_lock(&p_info->frame_lock);
	if (1 == open_alg) {
		release_alg(index);
		print("release_alg....... \n", index);
	}
	pthread_mutex_unlock(&p_info->frame_lock);
	//prt(info, "camera[%d] close unlock", index);
	usleep(200000);
	p_info->cam_running_state = CAM_STOPED;
	//prt(info, "camera[%d] close finish", index);
#if (DECODE_TYPE == 2)
	clear_frame(&g_ffmpeg_obj[index].yuv_frame, &g_ffmpeg_obj[index].frame, &g_ffmpeg_obj[index].sw_frame);
	clear_ffmpeg(&g_ffmpeg_obj[index].input_ctx, &g_ffmpeg_obj[index].decoder_ctx, &g_ffmpeg_obj[index].decode, &g_ffmpeg_obj[index].img_convert_ctx);
#endif
	deinit_cam_info_yuv(index);
	clear_yuv_item(index);
	deinit_yuv_list(index); //æ¸…é™¤yuv list

	p_info->have_frame_size = 0; //å¸§å¤§å°ä¸º0
	p_info->close_flag = 0;

	p_info->file_frame_curr_no = 0;
	p_info->file_frame_num = 0;

	if (p_info->p_radar) {
		ClearQueue(p_info->p_radar);
	}

	ClearQueue(p_info->p_f_e_queue);
	ClearQueue(p_info->p_5s_frame_queue);
	ClearQueue(p_info->p_file_queue);
//	DestroyQueue(cam_info[index].p_event_list);

	if (p_info->pYuvBuf) {
		free(p_info->pYuvBuf);
		p_info->pYuvBuf = NULL;
	}

	if (!is_call_cmd) { //
		camera_set_curr_status(index, CAM_RESET_NONE);
		cmd_output(index, DELCAMERANUMPARAM);
	}

	p_info->rtsp_fag = 0;
	printf("camera_close finished....................... \n");
	return 0;
}

int camera_ctrl(int cmd, int index, int blocked, void *data)
{
	pthread_mutex_lock(&cam_info[index].cmd_lock);
	set_cmd(cmd, index);
	pthread_mutex_unlock(&cam_info[index].cmd_lock);
	return 0;
}

void get_camera_network(int index, void *pinfo)
{
	m_camera_info *p_info = (m_camera_info *)pinfo;
	strncpy((char*)p_info->ip,  (char*)cam_info[index].ip, strlen((char*)cam_info[index].ip) + 1);
	strncpy((char*)p_info->name, (char*)cam_info[index].name, strlen((char*)cam_info[index].name) + 1);
	strncpy((char*)p_info->passwd, (char*)cam_info[index].passwd, strlen((char*)cam_info[index].passwd) + 1);
	p_info->port = cam_info[index].port;
	p_info->vtype = cam_info[index].vtype;
}

bool set_camera_network(int index, char *ip, char *name, char *passwd, int port)
{
	bool ret = false;

	pthread_mutex_lock(&cam_info[index].cmd_lock);

	if ( strcmp((char*)cam_info[index].ip, ip) != 0 ) {
		strncpy((char*)cam_info[index].ip, ip, strlen(ip) + 1);
		ret = true;
	}
	if ( strcmp((char*)cam_info[index].name, name) != 0 ) {
		strncpy((char*)cam_info[index].name, name, strlen(name) + 1);
		ret = true;
	}

	if ( strcmp((char*)cam_info[index].passwd, passwd) != 0 ) {
		strncpy((char*)cam_info[index].passwd, passwd, strlen(passwd) + 1);
		ret = true;
	}

	if ( cam_info[index].port != port ) {
		cam_info[index].port = port;
		ret = true;
	}

	pthread_mutex_unlock(&cam_info[index].cmd_lock);


	return ret;
}

void init_watchdog()
{
	for (int i = 0; i < CAM_MAX; i++) {
		cam_info[i].watchdog_value = DOG_HUNGRY_STATE;
	}
	m_timed_func_data *p_data_watchdog = regist_timed_func(WATCHDOG_CHECK_DURATION,
	                                     (void *) watchdog_func, (void *) NULL);
	start_timed_func(p_data_watchdog);

}

void camera_func(void *data)
{
	static unsigned char reboot_flag = 0;

	m_camera_info *info = (m_camera_info *) data;

	prt(info, "watchdog dog for cam %d , fp10s:%d / %d list: %d \n", info->index, info->watchdog_alg_frames, info->watchdog_frames, cam_yuv_list[info->index].total_num);
#if 0

	if (info->watchdog_alg_frames > 0)
		prt(info, "watchdog dog for cam %d , run time:%d  frame time: %d ", info->index, info->run_time, info->run_time / info->watchdog_alg_frames);
	if (info->all_frames > 0)
		prt(info, "watchdog dog for cam %d , all time:%d  frame time: %d ", info->index, info->all_times,  info->all_times / info->all_frames);
	info->run_time = 0;
#endif

	//if(info->watchdog_frames<2&&info->index==0&&stoped==0){

	//if(0){
	// if( (info->watchdog_frames < 2) && (1 == info->open_flg) ){
	//prt(info,"watchdog reboot ");

	//   release_alg(info->index);
	//	system("reboot");
	//}

	if (0 == info->watchdog_alg_frames) {
		info->zero_frame_cnt++;
		if (info->zero_frame_cnt > 120) { //2åˆ†é’ŸåŽé‡å?
			//system("reboot");
			info->zero_frame_cnt = 0;
		}
	} else {
		info->zero_frame_cnt = 0;
	}

	if (info->watchdog_value == DOG_HUNGRY_STATE) {

		//camera_ctrl(CAMERA_CONTROL_RESET,info->index,1,NULL); 2019.05.28 disable by roger
#if 0
		if (info->cam_no_work_cnt > 5) {
			if (FILE_TYPE != info->vtype )
				camera_ctrl(CAMERA_CONTROL_RESET, info->index, 0, NULL);
			info->cam_no_work_cnt = 0;
			//cam_info[info->index].open_flg = 0;
			prt(info, "watch dog for cam %d:camera loop stopped,resetting...", info->index);
		}

		info->cam_running_state = CAM_STOPED;
#endif
		info->cam_no_work_cnt++;
	} else {
		info->cam_running_state = CAM_RUNNING;
		//	prt(camera_msg,"watch dog for cam %d:camera running normally",info->index);
	}
	info->watchdog_value = DOG_HUNGRY_STATE;

	info->watchdog_frames = 0;
	info->watchdog_alg_frames = 0;

	if (g_ivddevsets.autoreset < 8) {

		struct tm *ptr;
		time_t lt;
		lt = time(NULL);
		ptr = localtime(&lt);

		if (g_ivddevsets.autoreset == ptr->tm_wday ) {

			if ( (ptr->tm_hour * 60 + ptr->tm_min)  ==  g_ivddevsets.timeset) {
				if (ptr->tm_sec < 30)
					reboot_flag = 1;
				else if (1 == reboot_flag) //ºó30ÃëÔÙÖØÆô£¬ÎªÁË±ÜÃâÍ¬Ò»·ÖÖÓÄÚ²»¶ÏÖØÆô
					reboot_cmd();
				//system("reboot");
			}
		}

	}

	if (PROTO_HUAITONG == g_ivddevsets.pro_type ) {

		if (info->cam_pre_state != info->cam_running_state) { // ×´Ì¬±äÊ±·¢ËÍ
			send_message(0x01, NULL);
		}

		info->cam_pre_state = info->cam_running_state;
	}

}

void *watchdog_func(void *data)
{
	static unsigned short cycle_seconds = 0;

	for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++) {
		if (1 == g_cam_num.exist[i])
			camera_func(&cam_info[i]);
	}

	//ÈÕÖ¾ÎÄ¼þÉ¾³ý
	if (cycle_seconds >= 1800 ) { // == g_ivddevsets.overWrite ) { //30 minute
		handle_log_file();
		cycle_seconds = 0;
	}

	cycle_seconds++;
}


int get_cam_running_state(int index)
{
	return cam_info[index].cam_running_state ;
}

/* Function Description
 * name:
 * return:
 * args:data????????????????????‰^
 * comment:?????????????????????????????????????þŸ
 * todo:
 */
void *camera_main(void *data)
{
	int ret;
	int cmd = CAMERA_CONTROL_NULL;
	//	for(unsigned short index = 0, j = 0; index < CAM_MAX && j < g_cam_num.cam_num; index++) {

	// if (1 == g_cam_num.exist[index]) {
	//         j++;
	//    }
	for (unsigned short index = 0; index < CAM_MAX; index++) { //æ³¨æ„ç›¸æœºåˆ é™¤æ—¶g_cam_num.exist[index]

		pthread_mutex_lock(&cam_info[index].cmd_lock);
		cmd = get_cmd(index);
		pthread_mutex_unlock(&cam_info[index].cmd_lock);

		if (cmd == CAMERA_CONTROL_NULL) {
			//index = (index + 1) % CAM_MAX;
			continue;
		}

		m_camera_info *p_info = (m_camera_info *) &cam_info[index];
		//int index = p_info->index;
		// pthread_mutex_lock(&p_info->cmd_lock);
		switch (cmd) {
		case CAMERA_CONTROL_NULL:
			break;
		case CAMERA_CONTROL_OPEN:
		{
			prt(info, "main_CAMERA_CONTROL_RESET--open");
			//prt(info, "open............ camera %d",index);
			m_camera_info camera_info;
			//get_camera_network(index, &camera_info);
			// if (!camera_set_open_one(index) )
			// 	break;
			camera_open(index, (char *)g_camdetect[index].other.camIp, (int)g_camdetect[index].other.camPort, (char *)g_camdetect[index].other.username, (char *)g_camdetect[index].other.passwd, g_camdetect[index].other.videotype); //´ò¿ªÏà»ú
			//camera_open(index, (char *)camera_info.ip, camera_info.port, (char *) camera_info.name, (char *) camera_info.passwd,camera_info.vtype);
			//prt(info, "open............ camera %d end.....",index);
		}
		break;
		case CAMERA_CONTROL_RESET:
		{
			prt(info, "main_CAMERA_CONTROL_RESET1");
			// if (!camera_reset_open_one(index) )
			// 	break;
			camera_close(index, true);

			prt(info, "main_CAMERA_CONTROL_RESET1--close");

			/*
			net_close_camera(index);

			if (1 == p_info->open_flg) {
			   pthread_mutex_lock(&p_info->frame_lock_ex);
			   reset_alg(p_info->index);
			   pthread_mutex_unlock(&p_info->frame_lock_ex);

			   p_info->exit_flag = 1; //ÍË³ö´¦ÀíÏß³Ì
			}

			p_info->open_flg = 0;
			prt(info, "  camera %d reset done",index);
			*/
			//m_camera_info camera_info;
			//get_camera_network(index, &camera_info);
			//camera_open(index, (char *)camera_info.ip, camera_info.port, (char *) camera_info.name, (char *) camera_info.passwd, camera_info.vtype);
			camera_open(index, (char *)g_camdetect[index].other.camIp, (int)g_camdetect[index].other.camPort, (char *)g_camdetect[index].other.username, (char *)g_camdetect[index].other.passwd, g_camdetect[index].other.videotype); //´ò¿ªÏà»ú
			prt(info, "main_CAMERA_CONTROL_RESET1--open--finish");
			/*
			ret=net_open_camera(index, (char *) camera_info.ip,
			        camera_info.port, (char *) camera_info.name,
			        (char *) camera_info.passwd, (void *) process_fun,
			        &cam_info[index]);
			prt(info, "  camera %d opened",index);

			if(ret<0){
			    prt(debug_long,"login %s (port %d) fail",camera_info.ip,camera_info.port);
			}
			*/
			// prt(info, "reseting............ camera %d end..",index);
		}
		break;
		case CAMERA_CONTROL_CLOSE:
		{
			// if ( cam_info[index].reset_status ==  CAM_RESETING) //é˜²æ­¢ç›¸æœºé‡å¯æ—¶å…³é—­ç›¸æœºæˆ–è€…å†æ¬¡æ‰“å¼€
			// 	continue;
			camera_close(index, false);
		}
		break;
		case CAMERA_CONTROL_SET_NTP:
			prt(info, "set cam");
			break;
		case CAMERA_CONTROL_RESET_ALG:
		{
			//prt(info, "reset alg begin");
			if (1 == p_info->open_alg) {
				p_info->open_alg = 0;
				usleep(50000);
				pthread_mutex_lock(&p_info->frame_lock);
				reset_alg(p_info->index, p_info->gpu_index);
				pthread_mutex_unlock(&p_info->frame_lock);
				p_info->open_alg = 1;
			}

			//prt(info, "reset alg finish");
		}
		break;
		case CAMERA_CONTROL_REPLAY:
		{

		}
		break;
		default:
			break;
		}

		pthread_mutex_lock(&p_info->cmd_lock);
		if (get_cmd(index) == cmd)
			set_cmd(CAMERA_CONTROL_NULL, index);
		pthread_mutex_unlock(&p_info->cmd_lock);

		// index = (index + 1) % CAM_MAX;
	}
}
/* Function Description
 * name:
 * return:
 * args:
 * comment:?????????????????????????????????????????????????????????????????????????cpu§¹???????§Õ????????î•
 * ?????????????????????????????????????
 * todo:
 */
void camera_service_init()
{

	int i, j;
#if SDK_TYPE == CAMERA_OPEN_TYPE
	if (open_sdk()) {
		prt(info, "err in open sdk");
	} else {
		prt(info, "ok to open net camera  sdk");
	}
#endif

	for (i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++) {

		if (!g_cam_num.exist[j])
			continue;
		else
			j++;

		init_alg(i);
		cam_info[i].open_flg = 0;
		cam_info[i].index = i;
		cam_info[i].cam_running_state = CAM_STOPED;

		set_cmd(CAMERA_CONTROL_NULL, i);

		/*
		        pthread_mutex_init(&cam_info[i].cmd_lock, NULL);
		        pthread_mutex_init(&cam_info[i].run_lock, NULL);
		        pthread_mutex_init(&cam_info[i].frame_lock, NULL);
		        pthread_mutex_init(&cam_info[i].frame_lock_ex, NULL);
		*/
		//prt(info,"init addr %p,i %d",&cam_info[i].cmd_lock,i);
#if 0
		m_timed_func_data *p_data = regist_timed_func(CHECK_DURATION,
		                            (void *) camera_main, (void *) &cam_info[i]);
		start_timed_func(p_data);

		cam_info[i].watchdog_value = DOG_HUNGRY_STATE;
		m_timed_func_data *p_data_watchdog = regist_timed_func(WATCHDOG_CHECK_DURATION,
		                                     (void *) watchdog_func, (void *) &cam_info[i]);
		start_timed_func(p_data_watchdog);
#endif
	}


	m_timed_func_data *p_data = regist_timed_func(CHECK_DURATION,
	                            (void *) camera_main, NULL);
	start_timed_func(p_data);
}

#if 0
bool camera_set_open_one(int index)
{
	bool ret = false;
	pthread_mutex_lock(&cam_info[index].open_one_lock);
	if ( cam_info[index].reset_status ==  CAM_RESET_NONE) {//é˜²æ­¢ç›¸æœºé‡å¯æ—¶å…³é—­ç›¸æœºæˆ–è€…å†æ¬¡æ‰“å¼€
		cam_info[index].reset_status = CAM_RESETING;
		ret = true;
	}
	pthread_mutex_unlock(&cam_info[index].open_one_lock);

	return ret;
}

void camera_reset_open_one(int index)
{
	pthread_mutex_lock(&cam_info[index].open_one_lock);
	if ( cam_info[index].reset_status ==  CAM_RESETING) {//é˜²æ­¢ç›¸æœºé‡å¯æ—¶å…³é—­ç›¸æœºæˆ–è€…å†æ¬¡æ‰“å¼€
		cam_info[index].reset_status = CAM_RESET_NONE;
	}
	pthread_mutex_unlock(&cam_info[index].open_one_lock);
}

#endif

void camera_set_curr_status(int index, int status)
{
	pthread_mutex_lock(&cam_info[index].open_one_lock);
	cam_info[index].curr_stat = status;
	pthread_mutex_unlock(&cam_info[index].open_one_lock);
}

#if ORACLE_VS == 1
void *process_fun_for_picture(void* data)
{
    //static int rd_count = 0;
	char *p_url = NULL;
	int index = *(int*)data;
	char file_url[50] = {0};


	m_camera_info *p_info = &cam_info[index];
	snprintf(file_url, 50, ORA_PASSID_FILE, index);
	FILE *fp = fopen( file_url , "w+" );//w+ clear file pass_id

   if (strlen(cam_info[index].pass_id) > 0) //write pass_id to file
        write_passid_to_file(fp, cam_info[index].pass_id, strlen(cam_info[index].pass_id) );

	while (!p_info->close_flag) {
		int rd_cnt = 0;
		memset(ora_rds, 0, ORACLE_RECORD_MAX * sizeof(ora_record_t));
		if (strlen(cam_info[index].pass_id) > 0) {
			rd_cnt = get_recored_from_oracle(ora_rds, cam_info[index].pass_id);
		}
		else {
			rd_cnt = get_recored_from_oracle(ora_rds, "0");
		}


		for (int i = 0; i < rd_cnt; i++) {
			if (strlen(ora_rds[i].pic_path) > 15 && NULL != strstr(ora_rds[i].pic_path, "http://")) {
				//sprintf(file_url, "%s%s\0", FILE_BASE_DIR, p_url + 1);
				if ( url_download_file(ora_rds[i].pic_path, URL_DOWNLOAD_DIR)) {
					handle_pic(index, URL_DOWNLOAD_DIR, FILE_FINISHED_DIR);
                    handle_last_pic(index, FILE_FINISHED_DIR);
                    if (atol(ora_rds[i].pass_id) > atol(cam_info[index].pass_id))
					    write_passid_to_file(fp, ora_rds[i].pass_id, strlen(ora_rds[i].pass_id) );
				}
			}
            if (atol(ora_rds[i].pass_id) > atol(cam_info[index].pass_id))
                strcpy(cam_info[index].pass_id, ora_rds[i].pass_id);
		}
#if 0
        if (!rd_cnt) {            
            rd_count++;
            if (rd_count > 500) {
                handle_last_pic(index, FILE_FINISHED_DIR);
                rd_count = 0;
            }
        }
#endif
		//p_url = strrchr(g_camdetect[index].other.filePath, '\\');

		p_info->reset_status = CAM_RESET_NONE;
        usleep(20000);
	}

}
#endif

void *process_client_picture(void* data)
{
	char wait_num = 0;
	char *p_url = NULL;
	int index = *(int*)data;
	char file_url[50] = {0};

	m_camera_info *p_info = &cam_info[index];

	while (!p_info->close_flag) {

		Queue *p_queue = (Queue*)get_non_motor_queue();
		PNode p_node = DeQueue(p_queue);

		//for test
		// p_node = (PNode)malloc(sizeof(QNode));
		// img_list_node_t *p_node11 = (img_list_node_t*)malloc(sizeof(img_list_node_t)); 
		// p_node->p_value = (char *)p_node11;
		
		// memcpy(&p_node11->img_id, "1576141021399205", 20);
		//memcpy(&p_node->cli_fd, &clt_len, sizeof(struct sockaddr_in));
		/////test end

		if (!p_node) {
			if (wait_num % 100 == 0) {
				handle_last_pic_cli(index, NULL);
				wait_num = 0;
			}
			usleep(30);
			wait_num++;
			continue;
		} 
			
		img_list_node_t *p_st_img = (img_list_node_t*)p_node->p_value;

		if (p_st_img && strlen(p_st_img->img_id) > 0) {
			memset(&file_url, 0, 50);
			sprintf(file_url,"%s/%s.jpg", NON_MOTOR_IMG_DIR, p_st_img->img_id);
			handle_non_motor_pic_cli(index, file_url, NULL, p_st_img->img_id, &p_st_img->cli_fd);
		}

		if (p_st_img)
			free(p_node->p_value);
		free(p_node);
		// p_info->reset_status = CAM_RESET_NONE;
		wait_num = 0;
        usleep(20);
	}

}