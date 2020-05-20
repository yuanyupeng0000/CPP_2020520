#ifndef DECODE_H264_H
#define DECODE_H264_H
#ifdef __cplusplus
extern "C" {
#endif
#include "vaapi.h"
#include "DSPARMProto.h"
#define DST_PITCH FULL_COLS
#define SRC_PITCH 640
int open_decoder(AVCodecContext **m_AVCodeContext, AVFrame **m_AVFrame,
		struct va_info *info);
int run_decode(AVPacket *pkt, AVCodecContext *m_AVCodeContext,
		AVFrame *m_AVFrame, struct va_info *vainfo);
int close_decoder(AVCodecContext *m_AVCodeContext, AVFrame *m_AVFrame);
void open_va(struct va_info *info);
int ffmpeg_init_context(AVCodecContext **avctx);
int vaapi_decode_to_image(struct va_info *info);
#ifdef __cplusplus
}
#endif

#endif

