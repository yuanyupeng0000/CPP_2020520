/*
 * h264_stream_file.h
 *
 *  Created on: 2016��9��7��
 *      Author: root
 */

#ifndef H264_STREAM_FILE_H_
#define H264_STREAM_FILE_H_
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#ifdef __cplusplus
}
#endif
#include "common.h"
#define FILE_NAME_MAX 50
typedef struct h264_file_common {
	char file_name[FILE_NAME_MAX];
	AVFormatContext *av_format_context;
	AVPacket h264_pkt;
} m_h264_file_common;

//int H264Decode_file(mDeCoder *de, unsigned char **oubuf, unsigned char **oubufu, unsigned char **oubufv);
//int get_file_264buf(mDeCoder *de);
int open_h264_file(m_h264_file_common *common)
;

int get_file_264buf1(m_h264_file_common *common)
;
int close_h264_file(m_h264_file_common *common);
#endif /* H264_STREAM_FILE_H_ */
