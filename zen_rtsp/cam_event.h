#ifndef INCLUDE_CAM_EVENT_H_
#define INCLUDE_CAM_EVENT_H_
#include <stdlib.h>
#include <pthread.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/videoio.hpp"

#include "../alg-mutilthread/DSPARMProto.h"
#include "queue.h"


using namespace cv;
using namespace std;

#define EVENT_MAX 20

typedef struct{
	unsigned char region_id;
	unsigned int type;
	CPoint eventBox[4];
	char pic_name[FILENAME_SIZE];
	char video_name[FILENAME_SIZE];
}New_Event_t;

typedef struct frame_node {
	//cv::Mat frame;
	Mat *p_frame;
	unsigned long ts;
}frame_node_t;
	
typedef struct file_node { 
	//char file_name[126];
	VideoWriter *p_write;
	long ts; 
}file_node_t;

typedef struct frame_event_node {
	//cv::Mat frame;
	Mat *p_frame;
	unsigned char event_num;
	New_Event_t event_data[EVENT_MAX];
}frame_event_node_t;

void *pic_video_handle(void *data);
void get_names(char *pic_name,char *video_name, int type);
void frame_in_5s_handle(Queue *p_queue, Mat *frame, unsigned long ts);
void event_handle(Queue *p_queue, Queue * p_f_queue, char *video_name ,unsigned long ts);
void file_list_handle(Queue *p_queue, Mat *frame, unsigned long ts) ;
void add_event_list(int index,unsigned char *inybuf,unsigned char *inubuf,unsigned char *invbuf,  EVENTOUTBUF *p_event_data, int iheight, int iwidth);
#endif


