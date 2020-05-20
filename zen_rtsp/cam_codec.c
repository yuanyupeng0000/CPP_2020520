#include "cam_codec.h"
#include "common.h"
#include "glplayer.h"
//#define USE_FILE
//#define PLAY_FRAMES
#include "g_define.h"

m_codec_common codec_info[CAM_MAX];

int H264DeocderRelease(mDeCoder *de)
{
	if (de->pFrame_)
		av_free(de->pFrame_);
	if (de->codec_)
		avcodec_close(de->codec_);
	de->pFrame_ = NULL;
	de->codec_ = NULL;
	de->videoCodec = NULL;
	return 1;
}

//int H264DeocderInit(mDeCoder *de, int width,int height)
//{
//	char *fn = de->fn;
//	de->frames=0;
//#ifdef PLAY_FRAMES
////	create_detach_thread(play_thread, 1, de);
////	start_gl_window(&de->w);
//#endif
////	av_register_all();
////	if (fn != NULL) {
////		const char *fileName = fn;
////		de->av_format_context = avformat_alloc_context();
////		if (avformat_open_input(&de->av_format_context, fileName, NULL, 0) != 0) {
////			prt(err, "file %s not found", fileName);
////			exit_program()
////			;
////		} else if (avformat_find_stream_info(de->av_format_context, NULL) < 0) {
////			exit_program()
////			;
////		}
////	}
//	de->videoCodec = avcodec_find_decoder(CODEC_ID_H264);
//	de->codec_ = avcodec_alloc_context3(de->videoCodec);
//	if (avcodec_open2(de->codec_, de->videoCodec, NULL) >= 0)
//		de->pFrame_ = avcodec_alloc_frame();
//	else {
//		H264DeocderRelease(de);
//		return -1;
//	}
//#ifdef HW_DECODE
//	de->vainfo.vaapi_context = NULL;
//	de->vainfo.vaapi_context_ffmpeg = NULL;
//	de->vainfo.av_frame = de->pFrame_;
//#ifdef HW_DECODE
//	open_va(&(de->vainfo));
//	ffmpeg_init_context(&(de->codec_));
//#endif
//	de->codec_->opaque = &de->vainfo;
//#endif
//	return 0;
//}

//int init_h264_decoder(	AVFrame ** p_p_av_frame,
//AVCodecContext ** p_p_av_codec_ctx,
//AVCodec ** p_p_av_codec)
//{
////	AVFrame * p_av_frame=*p_p_av_frame;
////	AVCodecContext * p_av_codec_ctx=*p_p_av_codec_ctx;
////	AVCodec * p_av_codec=*p_p_av_codec;
//	//char *fn = de->fn;
//	//de->frames=0;
//#ifdef PLAY_FRAMES
////	create_detach_thread(play_thread, 1, de);
////	start_gl_window(&de->w);
//#endif
////	av_register_all();
////	if (fn != NULL) {
////		const char *fileName = fn;
////		de->av_format_context = avformat_alloc_context();
////		if (avformat_open_input(&de->av_format_context, fileName, NULL, 0) != 0) {
////			prt(err, "file %s not found", fileName);
////			exit_program()
////			;
////		} else if (avformat_find_stream_info(de->av_format_context, NULL) < 0) {
////			exit_program()
////			;
////		}
////	}
//	*p_p_av_codec = avcodec_find_decoder(CODEC_ID_H264);
//	*p_p_av_codec_ctx = avcodec_alloc_context3(*p_p_av_codec);
//	if (avcodec_open2(*p_p_av_codec_ctx, *p_p_av_codec, NULL) >= 0)
//		*p_p_av_frame= avcodec_alloc_frame();
//	else {
//	//	H264DeocderRelease(de);
//		return -1;
//	}
//#ifdef HW_DECODE
//	de->vainfo.vaapi_context = NULL;
//	de->vainfo.vaapi_context_ffmpeg = NULL;
//	de->vainfo.av_frame = de->pFrame_;
//#ifdef HW_DECODE
//	open_va(&(de->vainfo));
//	ffmpeg_init_context(&(de->codec_));
//#endif
//	de->codec_->opaque = &de->vainfo;
//#endif
//	return 0;
//}

#if 0 // disable by roger 2019.08.16

int open_h264_decoder(int index)
{
	m_codec_common *codec_common=&codec_info[index];

	AVFrame ** p_p_av_frame=&codec_common->p_av_frame;
	AVCodecContext ** p_p_av_codec_ctx=&codec_common->p_av_codec_ctx;
	AVCodec ** p_p_av_codec=&codec_common->p_av_codec;

//	AVFrame * p_av_frame=*p_p_av_frame;
//	AVCodecContext * p_av_codec_ctx=*p_p_av_codec_ctx;
//	AVCodec * p_av_codec=*p_p_av_codec;
	//char *fn = de->fn;
	//de->frames=0;
#ifdef PLAY_FRAMES
//	create_detach_thread(play_thread, 1, de);
//	start_gl_window(&de->w);
#endif
	av_register_all();
//	if (fn != NULL) {
//		const char *fileName = fn;
//		de->av_format_context = avformat_alloc_context();
//		if (avformat_open_input(&de->av_format_context, fileName, NULL, 0) != 0) {
//			prt(err, "file %s not found", fileName);
//			exit_program()
//			;
//		} else if (avformat_find_stream_info(de->av_format_context, NULL) < 0) {
//			exit_program()
//			;
//		}
//	}
	*p_p_av_codec = avcodec_find_decoder(CODEC_ID_H264);
	*p_p_av_codec_ctx = avcodec_alloc_context3(*p_p_av_codec);
	if (avcodec_open2(*p_p_av_codec_ctx, *p_p_av_codec, NULL) >= 0)
		*p_p_av_frame= avcodec_alloc_frame(); //2019.04.18 by roger
	else {
	//	H264DeocderRelease(de);
		return -1;
	}
#ifdef HW_DECODE
	codec_common->vainfo.vaapi_context = NULL;
	codec_common->vainfo.vaapi_context_ffmpeg = NULL;
	codec_common->vainfo.av_frame = codec_common->p_av_frame;
//#ifdef HW_DECODE
	open_va(&(codec_common->vainfo));
	ffmpeg_init_context(&(codec_common->p_av_codec_ctx));
//#endif
	codec_common->p_av_codec_ctx->opaque = &codec_common->vainfo;
#endif
	return 0;
}
int close_h264_decoder(int index)
{
	m_codec_common *common=&codec_info[index];

	if (common->p_av_frame) {
		av_free(common->p_av_frame);
		common->p_av_frame=NULL;
	}
	if (common->p_av_codec_ctx) {
		avcodec_close(common->p_av_codec_ctx);
		common->p_av_codec_ctx=NULL;
	}

	return 1;
}


int h264filereset(mDeCoder *de, int width,int height)
{
	const char *fileName = de->fn;

	avcodec_close(de->codec_);
	avformat_close_input(&de->av_format_context);
	de->av_format_context = avformat_alloc_context();
	if (avformat_open_input(&de->av_format_context, fileName, NULL, 0) != 0) {
		exit_program()
		;
	} else if (avformat_find_stream_info(de->av_format_context, NULL) < 0) {
		exit_program()
		;
	}
	de->videoCodec = avcodec_find_decoder(CODEC_ID_H264);
	de->codec_ = avcodec_alloc_context3(de->videoCodec);
	if (avcodec_open2(de->codec_, de->videoCodec, NULL) >= 0)
		de->pFrame_ = avcodec_alloc_frame();
	else {
		H264DeocderRelease(de);
		return -1;
	}
	return 0;
}



//int ffmeg_decode_video(AVPacket *pkt,
//		AVCodecContext *m_AVCodeContext, AVFrame *m_AVFrame,
//		mDeCoder *de)
//{
//#ifdef HW_DECODE
//	struct va_info *vainfo = &de->vainfo;
//#endif
//	int got_picture = 0;
//	int len = 0;
//	av_init_packet(pkt);
//
//#ifdef SEND_TO_CLIENT
//	memcpy(de->h264_buf, pkt->data, pkt->size);
//
//	de->h264_len = pkt->size;
//#endif
//	while (pkt->size > 0) {
//		len = avcodec_decode_video2(m_AVCodeContext, m_AVFrame, &got_picture,
//				pkt);
//		if (len < 0) {
//			return -1;
//		}
//		if (got_picture) {
//#ifdef HW_DECODE
//			vaapi_decode_to_image(vainfo);		//get rst buffer from hw surface
//#endif
//
//			return 0;
//			break;
//		}
//		pkt->size -= len;
//		pkt->data += len;
//	}
//	return 1;
//}
int ffmeg_decode_video1(AVPacket *pkt,
		AVCodecContext *m_AVCodeContext, AVFrame *m_AVFrame,int index)
{
#ifdef HW_DECODE
	//struct va_info *vainfo = &de->vainfo;
#endif
	int got_picture = 0;
	int len = 0;
	av_init_packet(pkt);

#ifdef SEND_TO_CLIENT
	memcpy(de->h264_buf, pkt->data, pkt->size);

	de->h264_len = pkt->size;
#endif
	while (pkt->size > 0) {
		len = avcodec_decode_video2(m_AVCodeContext, m_AVFrame, &got_picture,
				pkt);
		if (len < 0) {
			return -1;
		}
		if (got_picture) {
#ifdef HW_DECODE
			vaapi_decode_to_image(&codec_info[index].vainfo);		//get rst buffer from hw surface
#endif

			return 0;
			break;
		}
		pkt->size -= len;
		pkt->data += len;
	}
	return 1;
}
//int H264Decode(mDeCoder *de,   AVPacket *p_pkt,unsigned char **oubuf, unsigned char **oubufu, unsigned char **oubufv)
// {
//	if (!ffmeg_decode_video(&de->h264_pkt, de->codec_, de->pFrame_, de)) {
//		*oubuf = (unsigned char *) de->pFrame_->data[0];
//		*oubufu = (unsigned char *) de->pFrame_->data[1];
//		*oubufv = (unsigned char *) de->pFrame_->data[2];
//	}
//	return 0;
//}
int h264_decode(int index,char *data,int size,unsigned char **y, unsigned char **u,unsigned char **v)
{
	m_codec_common *de=&codec_info[index];
	de->av_pkt.data=(unsigned char *)data;
	de->av_pkt.size=size;

	if (!ffmeg_decode_video1(&de->av_pkt, de->p_av_codec_ctx, de->p_av_frame,index)) {
	//	prt(info,"decode ok");
		(de->oubuf) = (unsigned char *) de->p_av_frame->data[0];
		(de->oubufu) = (unsigned char *) de->p_av_frame->data[1];
		(de->oubufv) = (unsigned char *) de->p_av_frame->data[2];

		*y = (unsigned char *) de->p_av_frame->data[0];
		*u = (unsigned char *) de->p_av_frame->data[1];
		*v= (unsigned char *) de->p_av_frame->data[2];
		//printf("###################frame-- width: %d height: %d \n", de->p_av_frame->width, de->p_av_frame->height);
	}else{
		printf("decode err ");
		}
	return 0;
}
#ifdef USE_FILE
#define MAXFILEFRMS 100
//int H264Decode_file(mDeCoder *de, unsigned char **oubuf, unsigned char **oubufu, unsigned char **oubufv)
//{
//	AVPacket packet;
//	av_init_packet(&packet);
//
//	if (de->frms++ == MAXFILEFRMS) {
//		h264filereset(de, 640, 480);
//		de->frms = 0;
//	}
//	if ((av_read_frame(de->av_format_context, &packet) >= 0)) {
//	 	usleep(40000);
//		if (!ffmeg_decode_video(&packet, de->codec_, de->pFrame_, de)) {
//			*oubuf = de->pFrame_->data[0];
//			*oubufu = de->pFrame_->data[1];
//			*oubufv = de->pFrame_->data[2];
//		}
//	}
//	return 0;
//}
int get_file_264buf(AVFormatContext *av_format_context,AVPacket *h264_pkt)
{
	av_init_packet(h264_pkt);
//	if (de->frms++ == MAXFILEFRMS) {
//		h264filereset(de, 640, 480);
//		de->frms = 0;
//	}
	if ((av_read_frame(av_format_context, h264_pkt) >= 0)) {
		prt(info,"read frm");
	}
	return 0;
}
#endif
//int decode_h264_frame(mDeCoder *de, unsigned char **oubuf,
//		unsigned char **oubufu, unsigned char **oubufv)
//{
//	de->frames++;
//	if (!ffmeg_decode_video(&de->h264_pkt, de->codec_, de->pFrame_, de)) {
//		*oubuf = de->pFrame_->data[0];
//		*oubufu = de->pFrame_->data[1];
//		*oubufv = de->pFrame_->data[2];
//		return 0;
//	}
//	else{
//		return 1;
//	}
//}

int alloc_decoder(mDeCoder *de)
{
#ifdef PLAY_FRAMES
	de->first_frame=0;
	de->w=-1;
#endif
	av_register_all();
#ifdef USE_FILE
#endif
	de->videoCodec = avcodec_find_decoder(CODEC_ID_H264);
	de->codec_ = avcodec_alloc_context3(de->videoCodec);

	if (avcodec_open2(de->codec_, de->videoCodec, NULL) >= 0)
		de->pFrame_ = avcodec_alloc_frame();
	else {
		H264DeocderRelease(de);
		return -1;
	}
//#ifdef HW_DECODE
//	de->vainfo.vaapi_context = NULL;
//	de->vainfo.vaapi_context_ffmpeg = NULL;
//	de->vainfo.av_frame = de->pFrame_ ;
//
//	open_va(&(de->vainfo));
//	ffmpeg_init_context(&(de->codec_));
//
//	de->codec_->opaque=&de->vainfo;
//#endif
	return 0;
}
void release_decoder(mDeCoder *de)
{
#ifdef PLAY_FRAMES


//	pthread_cancel(de->p_handle);
//	stop_gl_window(&de->w);
#endif
	if(de->pFrame_)
		av_free(de->pFrame_);
	if(de->codec_)
		avcodec_close(de->codec_);
	de->pFrame_ = NULL;
	de->codec_ = NULL;
	de->videoCodec = NULL;
}
int H264DeocderReset(mDeCoder *de)
{
	release_decoder(de);
	alloc_decoder(de);
	return 0;
}
int reset_codec(mDeCoder *de, int width,int height)
{
	const char *fileName = de->fn;

	avcodec_close(de->codec_);
	avformat_close_input(&de->av_format_context);
	de->av_format_context = avformat_alloc_context();
	if (avformat_open_input(&de->av_format_context, fileName, NULL, 0) != 0) {
		exit_program()
		;
	} else if (avformat_find_stream_info(de->av_format_context, NULL) < 0) {
		exit_program()
		;
	}
	de->videoCodec = avcodec_find_decoder(CODEC_ID_H264);
	de->codec_ = avcodec_alloc_context3(de->videoCodec);
	if (avcodec_open2(de->codec_, de->videoCodec, NULL) >= 0)
		de->pFrame_ = avcodec_alloc_frame();
	else {
		H264DeocderRelease(de);
		return -1;
	}
	return 0;
}

#endif
