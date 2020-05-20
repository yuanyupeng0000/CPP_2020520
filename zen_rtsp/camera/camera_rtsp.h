#ifndef __CAMERA_RTSP_H__
#define __CAMERA_RTSP_H__


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/videoio.hpp>
extern "C" {
#include <libavformat/avformat.h>
#include "libswscale/swscale.h"

}
struct DecodeContext {
	AVBufferRef *hw_device_ref;
};


using namespace cv;
bool open_camera(char *path, VideoCapture &cap);
bool get_frame(Mat &fm, VideoCapture cap);


AVFrame *alloc_picture(enum AVPixelFormat pix_fmt, int width, int height);
bool get_frame_ffmpeg(AVFormatContext *input_ctx , AVCodecContext *decoder_ctx, struct DecodeContext *decode, AVStream *video_st, SwsContext **img_convert_ctx, AVFrame **yuv_frame, AVFrame **frame, AVFrame **sw_frame, int index, int video_index);
int  open_ffmpeg(char *url, AVFormatContext **input_ctx, AVCodecContext **decoder_ctx, struct DecodeContext &decode, AVStream **video_st, int &video_index);
void clear_ffmpeg(AVFormatContext **input_ctx , AVCodecContext **decoder_ctx, DecodeContext *decode, SwsContext **img_convert_ctx);
void clear_frame(AVFrame **yuvframe, AVFrame **frame, AVFrame **sw_frame);

#endif



