#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include "cam_event.h"
#include "g_define.h"
#include "cvxtext.h"
#include "sig_service.h"
#include "camera_service.h"

extern IVDCAMERANUM  g_cam_num;
extern m_camera_info cam_info[CAM_MAX];
extern mEventInfo    events_info[CAM_MAX];
extern EX_mRealStaticInfo ex_static_info;
extern IVDNetInfo       g_netInfo;


///////显示中文///////////////
int ToWchar(char* &src, wchar_t* &dest, const char *locale = "zh_CN.utf8")
{
	if (src == NULL) {
		dest = NULL;
		return 0;
	}

	// 根据环境变量设置locale
	setlocale(LC_CTYPE, locale);

	// 得到转化为需要的宽字符大�?
	int w_size = mbstowcs(NULL, src, 0) + 1;

	// w_size = 0 说明mbstowcs返回值为-1。即在运行过程中遇到了非法字�?很有可能使locale
	// 没有设置正确)
	if (w_size == 0) {
		dest = NULL;
		return -1;
	}

	//wcout << "w_size" << w_size << endl;
	dest = new wchar_t[w_size];
	if (!dest) {
		return -1;
	}

	int ret = mbstowcs(dest, src, strlen(src) + 1);
	if (ret <= 0) {
		return -1;
	}
	return 0;
}

static void CopyYUVToImage(uchar *dst, uint8_t *pY, uint8_t *pU, uint8_t *pV, int width, int height)
{
	uint32 size = width * height;
	memcpy(dst, pY, size);
	memcpy(dst + size, pU, size / 4);
	memcpy(dst + size + size / 4, pV, size / 4);
}

void get_names(char *pic_name, char *video_name, int cam_index, int type)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	sprintf(pic_name, "%s%d_%d_%lu.jpg", "/ftphome/pic/", cam_index, type, tv.tv_sec * 1000000 + tv.tv_usec);
	sprintf(video_name, "%s%d_%d_%lu.mp4", "/ftphome/video/", cam_index, type, tv.tv_sec * 1000000 + tv.tv_usec, type);
}

#if 0
int save_video(int cam_index, char *f_name, Mat fst)
{
	frame_save_lock[cam_index].lock();
	if (buffer_num[cam_index] < 1) {
		frame_save_lock[cam_index].unlock();
		return -1;
	}

	Mat fst = buffer_frames[cam_index].front();
	cv::VideoWriter recVid(f_name, cv::VideoWriter::fourcc('H', '2', '6', '4'), 15,  cv::Size(fst.cols, fst.rows));
	for (int i = 0; i < buffer_num[cam_index]; i++) {
		recVid.write(buffer_frame[i]);
		buffer_frame[i].release();
	}

	frame_save_lock[cam_index].unlock();
	return 1;
}
#endif

void insert_picture(Mat *pframe, CPoint *outline, int type, string pic_path)
{

	string str;

	switch (type) {
	case OVER_SPEED:
		// str.append("OVER_SPEED");
		str.append("超速");
		break;
	case REVERSE_DRIVE:
		//str.append("REVERSE_DRIVE");
		str.append("逆道行驶");
		break;
	case STOP_INVALID:
		//str.append("STOP_INVALID");
		str.append("违法停车");
		break;
	case NO_PEDESTRIANTION:
		//str.append("NO_PEDESTRIANTION");
		str.append("行人");
		break;
	case DRIVE_AWAY:
		//str.append("DRIVE_AWAY");
		str.append("驶离");
		break;
	case CONGESTION:
		//str.append("CONGESTION");
		str.append("拥堵");
		break;
	case DROP:
		//str.append("AbANDON_OBJECT");
		str.append("抛洒物");
		break;
	case NONMOTOR:
		//str.append("NON_MOTOR");
		str.append("禁行非机动车");
		break;
	case ACCIDENTTRAFFIC:
		str.append("事故");
		break;
	default: break;

	}

	CvxText text("./simhei.ttf"); //指定字体
	cv::Scalar size1{ 20, 0.5, 0.1, 0 }; // (字体大小, 无效�? 字符间距, 无效�?}

	text.setFont(nullptr, &size1, nullptr, 0);
	wchar_t *w_str;
	char *pstr = (char *)str.data();
	ToWchar(pstr, w_str);
	Mat frame = pframe->clone();

	if (outline != NULL) {
		for (int i = 0; i < 4; i++) {
			line(frame, Point(outline[i].x, outline[i].y), Point( outline[(i + 1) % 4].x, outline[(i + 1) % 4].y), Scalar(255, 255 , 0), 3, 8, 0 );
		}

		text.putText(frame, w_str, Point(outline[0].x, outline[0].y - 5), Scalar(0, 255, 255));
	}

	imwrite(pic_path, frame);
	frame.release();

}

#if 0
//保存行人属性图�?
void insert_man_picture(Mat frame, VdRect &region, string pic_path)
{
	if ( (region.x + region.w) > frame.cols || (region.y + region.h) > frame.rows ) {
		prt(info, "region error!");
		prt(info, "rows: %d cols: %d x: %d y: %d w: %d H: %d", frame.rows, frame.cols, region.x, region.y, region.w, region.h);
		return;
	}

	Rect rect(region.x, region.y, region.w, region.h);
	Mat man_image = frame(rect);
	imwrite(pic_path, man_image);
}
#endif

void *pic_video_handle(void *data)
{
	do {
		for (int i = 0; i < CAM_MAX; i++) {

			if (!g_cam_num.exist[i] || cam_info[i].close_flag)
				continue;
			//prt(info, "start camera[%d]", i);
			//prt(info, "get before ****************************cam_info[%d]: %d", i, cam_info[i].p_f_e_queue->size);
			PNode p_node = DeQueue(cam_info[i].p_f_e_queue);
			//prt(info, "get after ****************************cam_info[%d]: %d", i, cam_info[i].p_f_e_queue->size);


			if (!p_node)
				continue;

			struct timeval tv;
			gettimeofday(&tv, NULL);
			unsigned long ts = (unsigned long)tv.tv_sec;
			frame_event_node_t *p_value = (frame_event_node_t *)p_node->p_value;
			//prt(info, "get before ............................p_5s_frame_queue: %d", cam_info[i].p_5s_frame_queue->size);
			frame_in_5s_handle(cam_info[i].p_5s_frame_queue, p_value->p_frame, ts);
			//prt(info, "get after ****************************p_5s_frame_queue: %d",  cam_info[i].p_5s_frame_queue->size);

			file_list_handle(cam_info[i].p_file_queue, p_value->p_frame, ts);
			if (p_value->event_num > 0) {
				//prt(info, "start event_num ");
#if 1
				for (int j = 0; j < p_value->event_num; j++) {
					insert_picture(p_value->p_frame, p_value->event_data[j].eventBox, p_value->event_data[j].type, p_value->event_data[j].pic_name); //插入图片
					event_handle(cam_info[i].p_5s_frame_queue, cam_info[i].p_file_queue, p_value->event_data[j].video_name, ts); //插入前5s帧数及加文件node
				}
#endif
				//prt(info, "start tis_data_insert_mysql ");
				tis_data_insert_mysql(i, (void *)p_value);
				//prt(info, "end tis_data_insert_mysql ");
				if (g_netInfo.UpServer)
					send_event_to_third_server(i, (void *)p_value);
				//prt(info, "end event_num ");
			}

			//在5s frame list delete
			free(p_node->p_value);
			free(p_node);
			//prt(info, "finish camera[%d]", i);
		}
		usleep(30000);
	} while (1);
}

void delete_event_frame_cb(void *node)
{
	frame_event_node_t *p_obj = (frame_event_node_t *)node;
	if (p_obj->p_frame) {
		p_obj->p_frame->release();
		delete p_obj->p_frame;
	}
}


void add_event_list(int index, unsigned char *inybuf, unsigned char *inubuf, unsigned char *invbuf,  EVENTOUTBUF *p_event_data, int iheight, int iwidth)
{
	unsigned char new_cnt = 0;
	New_Event_t new_event[EVENT_MAX];
	struct timeval tv;
	gettimeofday(&tv, NULL);
	unsigned int ts = (unsigned int)tv.tv_sec;
	ex_static_info.real_event_info[index].camId = htonl(index);
	ex_static_info.real_event_info[index].deviceId = 0;

	ex_static_info.real_event_info[index].time = htonl(ts);
	ex_static_info.real_event_info[index].newEventFlag = p_event_data->uNewEventFlag;
	ex_static_info.real_event_info[index].eventNum = p_event_data->uEventNum;

	//if (p_event_data->uNewEventFlag > 0) {
	for (int i = 0; i < p_event_data->uEventNum; i++) {
		memset(ex_static_info.real_event_info[index].eventData[i].picPath, 0, FILE_PATH_MAX);
		memset(ex_static_info.real_event_info[index].eventData[i].videoPath, 0, FILE_PATH_MAX);
		ex_static_info.real_event_info[index].eventData[i].eventId = htonl(p_event_data->EventBox[i].uEventID);
		ex_static_info.real_event_info[index].eventData[i].eventType = p_event_data->EventBox[i].uEventType;
		ex_static_info.real_event_info[index].eventData[i].eventRect[0].x = htons(p_event_data->EventBox[i].EventBox[0].x);
		ex_static_info.real_event_info[index].eventData[i].eventRect[1].x = htons(p_event_data->EventBox[i].EventBox[1].x);
		ex_static_info.real_event_info[index].eventData[i].eventRect[2].x = htons(p_event_data->EventBox[i].EventBox[2].x);
		ex_static_info.real_event_info[index].eventData[i].eventRect[3].x = htons(p_event_data->EventBox[i].EventBox[3].x);
		ex_static_info.real_event_info[index].eventData[i].eventRect[0].y = htons(p_event_data->EventBox[i].EventBox[0].y);
		ex_static_info.real_event_info[index].eventData[i].eventRect[1].y = htons(p_event_data->EventBox[i].EventBox[1].y);
		ex_static_info.real_event_info[index].eventData[i].eventRect[2].y = htons(p_event_data->EventBox[i].EventBox[2].y);
		ex_static_info.real_event_info[index].eventData[i].eventRect[3].y = htons(p_event_data->EventBox[i].EventBox[3].y);
		/*
		for (int j = 0; j < events_info[index].eventAreaNum;  j++) { //左下角与右上角
			if ( (events_info[index].eventArea[j].eventType.type & (1 << ((int)p_event_data->EventBox[i].uEventType - 1) ) ) > 0 && events_info[index].eventArea[j].realcoordinate[1].x <= p_event_data->EventBox[i].EventBox[0].x &&
			        events_info[index].eventArea[j].realcoordinate[1].y <= p_event_data->EventBox[i].EventBox[0].y &&
			        events_info[index].eventArea[j].realcoordinate[3].x >= p_event_data->EventBox[i].EventBox[2].x &&
			        events_info[index].eventArea[j].realcoordinate[3].y >= p_event_data->EventBox[i].EventBox[2].y) {
				ex_static_info.real_event_info[index].eventData[i].ereaId = events_info[index].eventArea[j].areaNum;
				new_event[new_cnt].region_id = events_info[index].eventArea[j].areaNum;
				break;
			}
		}*/
		ex_static_info.real_event_info[index].eventData[i].ereaId = p_event_data->EventBox[i].uRegionID;

		//memcpy(ex_static_info.real_event_info[index].eventData[i].eventRect,p_event_data->EventBox[i].EventBox,4*sizeof(CPoint) );
		if (p_event_data->EventBox[i].uEventID > 0) {
			get_names(new_event[new_cnt].pic_name, new_event[new_cnt].video_name, index, p_event_data->EventBox[i].uEventType);

			strcpy((char *)ex_static_info.real_event_info[index].eventData[i].picPath, new_event[new_cnt].pic_name);
			strcpy((char *)ex_static_info.real_event_info[index].eventData[i].videoPath, new_event[new_cnt].video_name);
			new_event[new_cnt].type = (int)p_event_data->EventBox[i].uEventType;
			memcpy(new_event[new_cnt].eventBox, p_event_data->EventBox[i].EventBox, 4 * sizeof(CPoint));
			new_event[new_cnt].region_id = p_event_data->EventBox[i].uRegionID;
#if 0
			prt(info, "event: %d", new_event[new_cnt].region_id);
			for (int j = 0; j < events_info[index].eventAreaNum;  j++) {
				for (int ii = 0; ii < 4; ii++) {
					prt(info, "event_info [%d] x:%d y:%d", ii, events_info[index].eventArea[j].realcoordinate[ii].x, events_info[index].eventArea[j].realcoordinate[ii].y);
				}
			}

			for (int m = 0; m < 4; m++)
				prt(info, "event [%d] x:%d y:%d", new_cnt, p_event_data->EventBox[i].EventBox[(m + 3) % 4].x, p_event_data->EventBox[i].EventBox[(m + 3) % 4].y);
#endif
			new_cnt++;
		}
	}
	//}

	//frame list
	if (cam_info[index].p_f_e_queue->size < EVENT_QUEUE_MAX) {
		Mat *rgbImag = new Mat();
		Mat yuvImg;
		yuvImg.create(iheight * 3 / 2, iwidth,	CV_8UC1);
		CopyYUVToImage(yuvImg.data, inybuf, inubuf, invbuf, iwidth, iheight);
		//event list

		frame_event_node_t *node = (frame_event_node_t *)malloc(sizeof(frame_event_node_t));
		node->event_num = new_cnt;
		if (new_cnt > 0) {
			memcpy(&node->event_data, new_event, new_cnt * sizeof(New_Event_t));
		}
		cv::cvtColor(yuvImg, *rgbImag, CV_YUV2BGR_I420);
		node->p_frame = rgbImag;
		//cv::cvtColor(yuvImg, node->frame, CV_YUV2BGR_I420);
		//yuvImg.copyTo(node->frame);
		EnQueue(cam_info[index].p_f_e_queue, (char*)node, sizeof(frame_event_node_t), delete_event_frame_cb);
	}
	//prt(info, "add.................cam_info[%d]: %d", index, cam_info[index].p_f_e_queue->size);

}

void delete_5s_frame_cb(void *node)
{
	frame_node_t *p_obj = (frame_node_t *)node;
	if (p_obj->p_frame) {
		p_obj->p_frame->release();
		delete p_obj->p_frame;
	}
}

void delete_fhandle_cb(void *node)
{
	file_node_t *f_node = (file_node_t *)node;
	if (f_node) {
		f_node->p_write->release();
		delete f_node->p_write;
	}
}


//前5s帧
void frame_in_5s_handle(Queue *p_queue, Mat *pframe, unsigned long ts) //前5秒帧数
{
	ConditionDeleteQueue(p_queue, ts);
	frame_node_t *p_node = (frame_node_t *)malloc(sizeof(frame_node_t));
	p_node->p_frame = pframe;
	p_node->ts = ts;

	EnQueue(p_queue, (char *)p_node, sizeof(frame_node_t), delete_5s_frame_cb);
}

void event_handle(Queue *p_queue, Queue * p_f_queue, char *video_name , unsigned long ts)
{
	if (IsEmpty(p_queue))
		return;

	PNode front = GetFront(p_queue);
	//PNode  rear = GetRear(p_queue);

	Mat *p_frame = ((frame_node_t *)front->p_value)->p_frame;
	cv::VideoWriter *p_write = new  VideoWriter(video_name, cv::VideoWriter::fourcc('H', '2', '6', '4'), 10,  cv::Size(p_frame->cols, p_frame->rows));

	while (front) {
		p_frame = ((frame_node_t *)front->p_value)->p_frame;
		p_write->write(*p_frame);
		front = front->next;
		usleep(20);
	}

	//p_write->release();
	//delete p_write;


	int len = sizeof(file_node_t);
	file_node_t *p_f_node = (file_node_t *)malloc(len);
	memset(p_f_node, 0, len);
	p_f_node->ts = ts;
	p_f_node->p_write = p_write;

	EnQueue(p_f_queue, (char *)p_f_node, len, delete_fhandle_cb);

}


//文件列表插入当前帧
void file_list_handle(Queue *p_queue, Mat *frame, unsigned long ts) //插入后5s的帧
{
	if (IsEmpty(p_queue))
		return;

	PNode front = GetFront(p_queue);
	//PNode  rear = GetRear(p_queue);

	while (front) {
		file_node_t *f_node = (file_node_t *)front->p_value;
		//cv::VideoWriter wt(f_node->file_name, cv::VideoWriter::fourcc('H', '2', '6', '4'), 15,  cv::Size(frame->cols, frame->rows));
		f_node->p_write->write(*frame);
		front = front->next;

		if (f_node->ts < ts - 5 || f_node->ts > ts) {
			PNode d_node = DeQueue(p_queue);
			//f_node->p_write->release();
			//delete f_node->p_write;
			if (d_node->p_del_fun)
				d_node->p_del_fun(d_node->p_value);
			free(f_node);
			free(d_node);
		}
	}
}








