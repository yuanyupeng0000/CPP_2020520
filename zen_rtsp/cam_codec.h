/*
 * cam_codec.h
 *
 *  Created on: 2016��6��30��
 *      Author: root
 */

#ifndef INCLUDE_CAM_CODEC_H_
#define INCLUDE_CAM_CODEC_H_
#include <stdio.h>
#include <string.h>
#ifdef HW_DECODE
#include <va/va.h>
#include <va/va_drm.h>
#include "hw_vaapi.h"
#endif
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/frame.h>
#include <unistd.h>
#include <pthread.h>

#ifdef __cplusplus
}
#endif
//#define USE_FILE
#define PLAY_FRAMES
typedef struct{
	AVFrame * pFrame_;
	AVCodecContext * codec_;
	AVCodec * videoCodec;
	AVFormatContext *av_format_context;
	AVPacket h264_pkt;
	FILE * out_file;
#ifdef HW_DECODE
	struct va_info vainfo;
#endif
	unsigned char *y;
	unsigned char *u;
	unsigned char *v;
	int frames;
#ifdef USE_FILE
	int frms=0;
#endif
#ifdef SEND_TO_CLIENT
	unsigned int h264_len;
	unsigned char h264_buf[200000];

#endif
#ifdef PLAY_FRAMES

	int w;
 	unsigned char  buf_src[640*480*3/2];
 	unsigned char  buf_dst[640*480*3];
	int frame_ready;
	int first_frame;
	pthread_t p_handle;
	m_timed_func_data timer_data;
#endif
	char *fn=NULL;

}mDeCoder;
int init_h264_decoder(	AVFrame ** p_p_av_frame,
AVCodecContext ** p_p_av_codec_ctx,
AVCodec ** p_p_av_codec);
typedef struct codec_common {
	AVFrame * p_av_frame;
	AVCodecContext * p_av_codec_ctx;
	AVCodec * p_av_codec;
	AVPacket av_pkt;
	unsigned char *oubuf;
	unsigned char *oubufu;
	unsigned char *oubufv;
	int frames;
#ifdef HW_DECODE
	struct va_info vainfo;
#endif
} m_codec_common;
int open_h264_decoder(int index);
int close_h264_decoder(int index);
//int h264_decode(int index,char *data,int size,unsigned char **y, unsigned char **u,unsigned char **v);
int H264DeocderReset(mDeCoder *de);
int H264DeocderInit(mDeCoder *de, int width,int height);
int H264DeocderRelease(mDeCoder *de);
int H264Decode(mDeCoder *de,   AVPacket *p_pkt, unsigned char **oubuf, unsigned char **oubufu, unsigned char **oubufv);
int H264Decode_file(mDeCoder *de, unsigned char **oubuf, unsigned char **oubufu, unsigned char **oubufv);
int decode_h264_frame(mDeCoder *de,unsigned char **oubuf,
		unsigned char **oubufu,unsigned char **oubufv);
int alloc_decoder(mDeCoder *de);
void release_decoder(mDeCoder *de);
int get_file_264buf(AVFormatContext *av_format_context,AVPacket *h264_pkt);

int reset_codec(mDeCoder *de, int width,int height);
#endif /* INCLUDE_CAM_CODEC_H_ */
