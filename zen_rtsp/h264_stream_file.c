#include "h264_stream_file.h"

//int H264Decode_file(mDeCoder *de, unsigned char **oubuf, unsigned char **oubufu, unsigned char **oubufv)
//{
//	AVPacket packet;
//	av_init_packet(&packet);
//
//	if (de->frms++ == MAXFILEFRMS) {
//		h264filereset(de, 640, 480);
//		de->frms = 0;
//	}
//	if ((av_read_frame(de->ic, &packet) >= 0)) {
//	 	usleep(40000);
//		if (!ffmeg_decode_video(&packet, de->codec_, de->pFrame_, de)) {
//			*oubuf = de->pFrame_->data[0];
//			*oubufu = de->pFrame_->data[1];
//			*oubufv = de->pFrame_->data[2];
//		}
//	}
//	return 0;
//}
//int get_file_264buf(mDeCoder *de)
//{
//
//	av_init_packet(&de->h264_pkt);
//
//	if (de->frms++ == MAXFILEFRMS) {
//		h264filereset(de, 640, 480);
//		de->frms = 0;
//	}
//	if ((av_read_frame(de->ic, &de->h264_pkt) >= 0)) {
//		prt(info,"read frm");
//	}
//	return 0;
//}
int open_h264_file(m_h264_file_common *common)
{

	AVFormatContext **av_format_context=&common->av_format_context;
	char * fn=common->file_name;
//	av_register_all();
	if (fn != NULL) {
		const char *fileName = fn;
		 *av_format_context = avformat_alloc_context();
		if (avformat_open_input(av_format_context, fileName, NULL, 0) != 0) {
			prt(err, "file %s not found", fileName);
			exit_program()
			;
		} else if (avformat_find_stream_info(*av_format_context, NULL) < 0) {
			exit_program()
			;
		}
	}
	//   prt(info,"%x",av_format_context);
}

int get_file_264buf1(m_h264_file_common *common)
{
	AVFormatContext *av_format_context=common->av_format_context;
	AVPacket *h264_pkt=&common->h264_pkt;
	//prt(info,"%p",h264_pkt);
	av_init_packet(h264_pkt);
//	if (de->frms++ == MAXFILEFRMS) {
//		h264filereset(de, 640, 480);
//		de->frms = 0;
//	}
	if ((av_read_frame(av_format_context, h264_pkt) >= 0)) {
	//	prt(info,"read frm size %d",h264_pkt->size);
	}
	return 0;
}
int close_h264_file(m_h264_file_common *common)
{
	avformat_close_input(&common->av_format_context);

}
