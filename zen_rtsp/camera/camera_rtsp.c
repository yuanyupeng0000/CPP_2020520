#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "../common.h"
#include "camera_rtsp.h"
#include "../g_define.h"

#if DECODE_TYPE == 2

extern "C"
{
#include <libavcodec/avcodec.h>
//#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
#include "libavutil/hwcontext_qsv.h"
#include "libavutil/mem.h"
}

AVFrame *nv12_to_yuv420p(AVFrame *nv12_frame)
{
	int x, y, ret;
	AVFrame *frame = av_frame_alloc();
	if (!frame) {
		fprintf(stderr, "Could not allocate video frame\n");
		return NULL;
	}

	frame->format = AV_PIX_FMT_YUV420P;
	frame->width = nv12_frame->width;
	frame->height = nv12_frame->height;

	ret = av_frame_get_buffer(frame, 32); //4
	if (ret < 0) {
		fprintf(stderr, "Could not allocate frame data.\n");
		return NULL;
	}

	ret = av_frame_make_writable(frame);

	if (ret < 0)
		return NULL;

	if (nv12_frame->linesize[0] == nv12_frame->width) {
		memcpy(frame->data[0], nv12_frame->data[0], nv12_frame->height * nv12_frame->linesize[0]);
	} else {
		for (y = 0; y < frame->height / 2; y++) {
			for (x = 0; x < frame->width / 2; x++) {
				frame->data[1][y * frame->linesize[1] + x] = nv12_frame->data[1][y * nv12_frame->linesize[1] + 2 * x];
				frame->data[2][y * frame->linesize[2] + x] = nv12_frame->data[1][y * nv12_frame->linesize[1] + 2 * x + 1];
			}
		}
	}

	return frame;
}

AVFrame *alloc_picture(enum AVPixelFormat pix_fmt, int width, int height)
{
	AVFrame *picture;
	int ret;

	picture = av_frame_alloc();
	if (!picture)
		return NULL;
	picture->format = pix_fmt;
	picture->width = width;
	picture->height = height;

	/* allocate the buffers for the frame data */
	ret = av_frame_get_buffer(picture, 4); //4
	if (ret < 0) {
		fprintf(stderr, "Could not allocate frame data.\n");
		return NULL;
	}

	ret = av_frame_make_writable(picture);

	if (ret < 0)
		return NULL;

	return picture;
}


static AVPixelFormat get_format(AVCodecContext *avctx, const enum AVPixelFormat *pix_fmts)
{
	while (*pix_fmts != AV_PIX_FMT_NONE) {
		if (*pix_fmts == AV_PIX_FMT_QSV) {
			DecodeContext *decode = (DecodeContext*)avctx->opaque;
			AVHWFramesContext  *frames_ctx;
			AVQSVFramesContext *frames_hwctx;
			int ret;

			/* create a pool of surfaces to be used by the decoder */
			avctx->hw_frames_ctx = av_hwframe_ctx_alloc(decode->hw_device_ref);
			if (!avctx->hw_frames_ctx)
				return AV_PIX_FMT_NONE;
			frames_ctx   = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
			frames_hwctx = (AVQSVFramesContext *)frames_ctx->hwctx;

			frames_ctx->format            = AV_PIX_FMT_QSV;
			frames_ctx->sw_format         = avctx->sw_pix_fmt;
			frames_ctx->width             = FFALIGN(avctx->coded_width,  32);
			frames_ctx->height            = FFALIGN(avctx->coded_height, 32);
			frames_ctx->initial_pool_size = 32;

			frames_hwctx->frame_type = MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET;

			ret = av_hwframe_ctx_init(avctx->hw_frames_ctx);
			if (ret < 0)
				return AV_PIX_FMT_NONE;

			return AV_PIX_FMT_QSV;
		}

		pix_fmts++;
	}

	fprintf(stderr, "The QSV pixel format not offered in get_format()\n");

	return AV_PIX_FMT_NONE;
}

static int decode_packet(DecodeContext *decode, AVCodecContext *decoder_ctx,
                         AVFrame *frame, AVFrame *sw_frame, AVFrame **yuv_frame, AVPacket *pkt, AVIOContext *output_ctx, SwsContext **img_convert_ctx, int index)
{
	int ret = 0;
	int decode_flag;

#if (1 == FFMPEG_DECODE_TYPE)
	ret = avcodec_send_packet(decoder_ctx, pkt);
	if (ret < 0) {
		fprintf(stderr, "Error during decoding\n");
		return ret;
	}

	//while (ret >= 0) {
	if (ret >= 0) {
		int i, j;

		ret = avcodec_receive_frame(decoder_ctx, frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			ret = -1;
			goto fail;
		}
		else if (ret < 0) {
			fprintf(stderr, "Error during decoding\n");
			goto fail;
		}

		/* A real program would do something useful with the decoded frame here.
		 * We just retrieve the raw data and write it to a file, which is rather
		 * useless but pedagogic. */
		ret = av_hwframe_transfer_data(sw_frame, frame, 0);
		if (ret < 0) {
			fprintf(stderr, "Error transferring the data to system memory\n");
			//goto fail;
			goto fail;
		}


		if (yuv_frame && !(*yuv_frame) )
			*yuv_frame = alloc_picture(AV_PIX_FMT_YUV420P, sw_frame->width, sw_frame->height);

		//av_frame_unref(*yuv_frame);

		if (sw_frame->linesize[0] == sw_frame->width) {
			memcpy((*yuv_frame)->data[0], sw_frame->data[0], sw_frame->height * sw_frame->linesize[0]);
		} else {
			for (int y = 0; y < (*yuv_frame)->height; y++) {
				for (int x = 0; i < (*yuv_frame)->width; x++) {
					(*yuv_frame)->data[0][y * (*yuv_frame)->linesize[0] + x] = sw_frame->data[0][y * sw_frame->linesize[0] + x];
				}
			}

		}

		for (int y = 0; y < (*yuv_frame)->height / 2; y++) {
			for (int x = 0; x < (*yuv_frame)->width / 2; x++) {
				(*yuv_frame)->data[1][y * (*yuv_frame)->linesize[1] + x] = sw_frame->data[1][y * sw_frame->linesize[1] + 2 * x];
				(*yuv_frame)->data[2][y * (*yuv_frame)->linesize[2] + x] = sw_frame->data[1][y * sw_frame->linesize[1] + 2 * x + 1];
			}
		}


#if 0
		if (yuv_frame && !(*yuv_frame) )
			*yuv_frame = alloc_picture(AV_PIX_FMT_YUV420P, frame->width, frame->height);

		if (!(*img_convert_ctx)) {
			*img_convert_ctx = sws_getContext(frame->width, frame->height, \
			                                  AV_PIX_FMT_NV12, frame->width, frame->height, \
			                                  AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);
		}

		sws_scale(*img_convert_ctx, (const unsigned char* const*)sw_frame->data, sw_frame->linesize, 0, sw_frame->height, (*yuv_frame)->data, (*yuv_frame)->linesize);
#endif
#if 0 //yuv420p
		for (i = 0; i < FF_ARRAY_ELEMS( (*yuv_frame)->data) && (*yuv_frame)->data[i]; i++ ) {
			for (j = 0; j < (*yuv_frame)->height; j++) {
				if (0 == i)
					avio_write(output_ctx, (*yuv_frame)->data[i] + j * (*yuv_frame)->linesize[i], (*yuv_frame)->width);
				else {

					if ( j == (*yuv_frame)->height / 2)
						break;

					avio_write(output_ctx, (*yuv_frame)->data[i] + j * (*yuv_frame)->linesize[i], (*yuv_frame)->width / 2);
				}
			}
		}

#endif

#if 0 //nv12
		//if (1 == index) {
		for (i = 0; i < FF_ARRAY_ELEMS(sw_frame->data) && sw_frame->data[i]; i++)
			for (j = 0; j < (sw_frame->height >> (i > 0)); j++)
				avio_write(output_ctx, sw_frame->data[i] + j * sw_frame->linesize[i], sw_frame->width);
		//}
#endif

//fail:

		//av_frame_unref(sw_frame);
		//av_frame_unref(frame);

//		if (ret < 0)
//			return ret;
	}
#endif

#if (2 == FFMPEG_DECODE_TYPE)
	if (!(*yuv_frame)) {
		*yuv_frame    = av_frame_alloc();
	}

	ret = avcodec_decode_video2(decoder_ctx, *yuv_frame, &decode_flag, pkt);
	/*
	if (decode_flag) {
		if (yuv_frame && !(*yuv_frame) )
			*yuv_frame = alloc_picture(AV_PIX_FMT_YUV420P, sw_frame->width, sw_frame->height);

		if (!(*img_convert_ctx)) {
			*img_convert_ctx = sws_getContext(sw_frame->width, sw_frame->height, \
			                                  AV_PIX_FMT_NV12, sw_frame->width, sw_frame->height, \
			                                  AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);
		}

		sws_scale(*img_convert_ctx, (const unsigned char* const*)sw_frame->data, sw_frame->linesize, 0, sw_frame->height, (*yuv_frame)->data, (*yuv_frame)->linesize);
	}
	*/

#endif

fail:
	av_frame_unref(sw_frame);
	av_frame_unref(frame);
	if (ret < 0)
		return ret;
	else
		return 0;
}

int open_ffmpeg(char *url, AVFormatContext **input_ctx, AVCodecContext **decoder_ctx, struct DecodeContext &decode, AVStream **video_s,  int &video_index)
{
	const AVCodec *decoder;
	int ret, i, videoindex;
	AVDictionary* opts = NULL;
	/* open the input file */
	//av_dict_set(&opts, "rtsp_transport", "tcp", 0); //设置tcp
	//av_dict_set(&opts, "stimeout", "1000000", 0);//设置超时2秒
	//av_dict_set(&opts, "buffer_size", "102400000", 0);
	//av_dict_set(&opts, "max_delay", "500000", 0);
#if (2 == FFMPEG_DECODE_TYPE)
	//av_register_all();
	//avformat_network_init();
	*input_ctx = avformat_alloc_context();
#endif

	ret = avformat_open_input(input_ctx, url, NULL, &opts);

	if (ret < 0) {
		fprintf(stderr, "Cannot open input file '%s': ", url);
		goto finish;
	}


#if (2==FFMPEG_DECODE_TYPE) //soft decode

	if (avformat_find_stream_info(*input_ctx, NULL) < 0) {
		printf("Couldn't find stream information.\n");
		return -1;
	}
	video_index = -1;
	for (i = 0; i < (*input_ctx)->nb_streams; i++)
		if ((*input_ctx)->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			video_index = i;
			break;
		}
	if (video_index == -1) {
		printf("Didn't find a video stream.\n");
		return -1;
	}

	*decoder_ctx = (*input_ctx)->streams[video_index]->codec;
	decoder = avcodec_find_decoder((*decoder_ctx)->codec_id);
	if (decoder == NULL) {
		printf("Codec not found.\n");
		return -1;
	}

	if (avcodec_open2((*decoder_ctx), decoder, NULL) < 0) {
		printf("Could not open codec.\n");
		return -1;
	}

#endif

#if (1==FFMPEG_DECODE_TYPE) // hard decode
	/* find the first H.264 video stream */
	for (i = 0; i < (*input_ctx)->nb_streams; i++) {
		AVStream *st = (*input_ctx)->streams[i];

		if (st->codecpar->codec_id == AV_CODEC_ID_H264 && !(*video_s) )
			(*video_s) = st;
		else
			st->discard = AVDISCARD_ALL;
	}

	if (! (*video_s) ) {
		fprintf(stderr, "No H.264 video stream in the input file\n");
		goto finish;
	}

	/* open the hardware device */
	ret = av_hwdevice_ctx_create(&decode.hw_device_ref, AV_HWDEVICE_TYPE_QSV, "auto", NULL, 0);
	if (ret < 0) {
		fprintf(stderr, "Cannot open the hardware device\n");
		goto finish;
	}

	/* initialize the decoder */
	decoder = avcodec_find_decoder_by_name("h264_qsv");
	//decoder = avcodec_find_decoder_by_name("h264_vaapi");
	if (!decoder) {
		fprintf(stderr, "The QSV decoder is not present in libavcodec\n");
		goto finish;
	}

	(*decoder_ctx) = avcodec_alloc_context3(decoder);
	if (! (*decoder_ctx)) {
		ret = AVERROR(ENOMEM);
		goto finish;
	}
	(*decoder_ctx)->codec_id = AV_CODEC_ID_H264;
	if ( (*video_s)->codecpar->extradata_size) {
		(*decoder_ctx)->extradata = (uint8_t*)av_mallocz( (*video_s)->codecpar->extradata_size +
		                            AV_INPUT_BUFFER_PADDING_SIZE);
		if (!(*decoder_ctx)->extradata) {
			ret = AVERROR(ENOMEM);
			goto finish;
		}
		memcpy((*decoder_ctx)->extradata, (*video_s)->codecpar->extradata,
		       (*video_s)->codecpar->extradata_size);
		(*decoder_ctx)->extradata_size = (*video_s)->codecpar->extradata_size;
	}

	(*decoder_ctx)->opaque      = &decode;
	(*decoder_ctx)->get_format  = get_format;

	ret = avcodec_open2((*decoder_ctx), NULL, NULL);
	if (ret < 0) {
		fprintf(stderr, "Error opening the decoder: ");
		goto finish;
	}

#endif


#if 0
	*img_convert_ctx = sws_getContext((*input_ctx)->streams[(*video_s)->index]->codecpar->width, (*input_ctx)->streams[(*video_s)->index]->codecpar->height, \
	                                  AV_PIX_FMT_NV12, (*input_ctx)->streams[(*video_s)->index]->codecpar->width, (*input_ctx)->streams[(*video_s)->index]->codecpar->height, \
	                                  AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);

	*img_convert_ctx = sws_getContext(640, 480, \
	                                  AV_PIX_FMT_NV12, 640, 480, \
	                                  AV_PIX_FMT_YUV420P, 0, NULL, NULL, NULL);
#endif
finish:
	if (ret < 0) {
		char buf[1024];
		av_strerror(ret, buf, sizeof(buf));
		fprintf(stderr, "%s\n", buf);
	}

//    avformat_close_input(input_ctx);

	//  av_frame_free(&frame);
	//  av_frame_free(&sw_frame);

	//avcodec_free_context(decoder_ctx);
	//av_buffer_unref(&decode.hw_device_ref);
	//avio_close(output_ctx);

	return ret;
}

//AVIOContext *output_ctx = NULL;

bool get_frame_ffmpeg(AVFormatContext *input_ctx , AVCodecContext *decoder_ctx, struct DecodeContext *decode, AVStream *video_st, SwsContext **img_convert_ctx, AVFrame **yuv_frame, AVFrame **frame, AVFrame **sw_frame, int index, int video_index)
{
	int ret = -1;
	bool bret = true;
	AVPacket packet;

	if (!(*frame)) {
		*frame    = av_frame_alloc();
	}

	if (!(*sw_frame)) {
		*sw_frame = av_frame_alloc();
	}

	if (!frame || !sw_frame) {
		ret = AVERROR(ENOMEM);
		return false;
	}

	AVIOContext *output_ctx = NULL;
#if 0
	//if (index == 1 && !output_ctx)
	if (!output_ctx)
		ret = avio_open(&output_ctx, "test_file.yuv", AVIO_FLAG_WRITE);
#endif
	/* actual decoding and dump the raw data */
	//while (1) {
	if ((ret = av_read_frame(input_ctx, &packet)) < 0) {
		bret = false;

		printf("fail...............................input_ctx: %x packet: %x \n", input_ctx, packet);
	} else {
		printf("ok...............................input_ctx: %x packet: %x \n", input_ctx, packet);
	}

#if (1==FFMPEG_DECODE_TYPE)
	if (packet.stream_index == video_st->index)
#endif
#if (2==FFMPEG_DECODE_TYPE)
		if (packet.stream_index == video_index)
#endif
		{
			if ( !decode_packet(decode, decoder_ctx, *frame, *sw_frame, yuv_frame, &packet, output_ctx, img_convert_ctx, index) ) {
				bret = true;
			}
		}

	av_packet_unref(&packet);
	//}

	return bret;

}

//释放frame
void clear_frame(AVFrame **yuvframe, AVFrame **frame, AVFrame **sw_frame)
{
	if (!(*yuvframe)) {
		av_frame_unref(*yuvframe);
		av_frame_free(yuvframe);
		*yuvframe = NULL;
	}

	if (!(*frame)) {
		av_frame_unref(*frame);
		av_frame_free(frame);
		*frame = NULL;
	}
	if (!(*sw_frame)) {
		av_frame_unref(*sw_frame);
		av_frame_free(sw_frame);
		*sw_frame = NULL;
	}
}


void clear_ffmpeg(AVFormatContext **input_ctx , AVCodecContext **decoder_ctx, DecodeContext * decode, SwsContext **img_convert_ctx)
{

#if (1==FFMPEG_DECODE_TYPE)
	if (*decoder_ctx) {
		avcodec_free_context(decoder_ctx);
		*decoder_ctx = NULL;
	}

	if (decode) {
		av_buffer_unref(&decode->hw_device_ref);
	}
#endif

#if (2==FFMPEG_DECODE_TYPE)
	avcodec_close(*decoder_ctx);
#endif

	if (*img_convert_ctx) {
		sws_freeContext(*img_convert_ctx);
		*img_convert_ctx = NULL;
	}

	if (*input_ctx) {
		avformat_close_input(input_ctx);
		*input_ctx = NULL;
	}
}

#endif

//////////////////////////////////////////////////////////////////////////////////////
//VideoCapture cap;
bool open_camera(char *path, VideoCapture &cap)
{
	bool ret = false;
	if (cap.isOpened())
		cap.release();

	cap = VideoCapture(path);//CAP_INTEL_MFX
	// cap.set(CV_CAP_PROP_FPS, 25);
	//cap.get(CV_CAP_PROP_FRAME_COUNT);

	if (cap.isOpened())
		ret = true;

	return ret;
}

bool  get_frame(Mat & fm, VideoCapture cap)
{
	bool ret = false;
	if (cap.isOpened()) {
		fm.release();
		ret = cap.read(fm);
		//if (fm.empty()){
		//     ret = false;
		// }
	}
	return ret;
}

