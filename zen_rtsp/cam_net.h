/*
 * cam_net.h
 *
 *  Created on: 2016��6��30��
 *      Author: root
 */
#ifndef INCLUDE_CAM_NET_H_
#define INCLUDE_CAM_NET_H_
#ifdef __cplusplus
extern "C"{
#endif
#include "Net.h"
#ifdef __cplusplus
}
#endif
#include <pthread.h>
#include "g_define.h"
#include "common.h"
#define NAME_LEN_MAX 20


typedef struct cam_context{
#ifdef HANHUI
	long id;
#else
	HANDLE server_hangle;
	HANDLE channel_handle;
#endif
//	void (*callback_fun)(void *data);
	THREAD_ENTITY1 callback_fun;
	void *pri;
	pthread_t timed_func_handle;
	int port;
	char ip[NAME_LEN_MAX];
	char passwd[NAME_LEN_MAX];
	char username[NAME_LEN_MAX];
	int cam_id;
	char *frame_data;
	int frame_size;
	int index;
}m_cam_context;
int open_sdk(void);
void close_sdk(void);
m_cam_context *prepare_camera(char ip[],int port,char name[],char passwd[],void *func ,void * pri);
//int open_camera(m_cam_context *);
int net_open_camera(int index,char ip[],int port,char name[],char passwd[],void *func,void *pri);
void net_close_camera(int index);
int cam_set_shutter(int index, int value);
#endif /* INCLUDE_CAM_NET_H_ */
