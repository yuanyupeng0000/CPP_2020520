#ifdef DETECT_PERSON_ATTRIBUTE
#include "attribute_detect.h"
#define StandardHeight 200
int attri_init_flag = 0;//多线程只执行一次,初始化全局变量
///////////////////////////////////////////////////////////////////////////行人属性初始化
bool HumanAttributeInit(ALGCFGS *pCfgs)
{
	pCfgs->uPersonNum = 0;//行人数
	memset(pCfgs->PersonAttributeBox, 0, MAX_PERSON_NUM * sizeof(HumanAttributeBox));//行人属性
	return TRUE;
}
///////////////////////////////////////////////////////////////////////////单车属性初始化
bool BicycleAttributeInit(ALGCFGS *pCfgs)
{
	pCfgs->uBicycleNum = 0;//单车数
	memset(pCfgs->BikeAttributeBox, 0, MAX_BICYCLE_NUM * sizeof(BicycleAttributeBox));//单车属性
	return TRUE;
}
#ifndef USE_PYTHON
#include <pthread.h>
#define  NET_NUM  8
#define MAX_GPU_NUM 8
#define DARKNET_NET_NUM  8 //多少个darknet net 加载一个行人属性网络
int darknet_num = 0;//加载的darknet数量
int human_attri_init[MAX_GPU_NUM];//给每个gpu初始化一个行人属性网络
pthread_t attri_detect_thread[NET_NUM];
int init_attri_lock[NET_NUM];
pthread_mutex_t attri_lock[NET_NUM];
typedef struct attri_detect_args{
	int thread_idx;//线程ID
	IplImage* imgROI;//图像数据
	int net_idx;//检测网络id
	int* result;//输出结果
} attri_detect_args;
//////////////////////////////////////////////////////////////////////////////caffe c++进行检测
int attri_net_num = 0;
int detect_net_idx = 0;
const char* deploy_file = "deploy_mcnn_Attri.prototxt";
const char* trained_file = "mcnnsolver_iter_15000.caffemodel";
//const char* deploy_file = "deploy.prototxt";
//const char* trained_file = "solver_iter_40000.caffemodel";

void attri_detect_in_thread(void* ptr)
{
	attri_detect_args args  = *(attri_detect_args *)ptr;
	int thread_id = args.thread_idx;
	unsigned char* imgdata = (unsigned char* )(args.imgROI->imageData);
	int w = args.imgROI->width;
	int h = args.imgROI->height;
	int net_idx = args.net_idx;
	int* result = args.result;
	if(init_attri_lock[thread_id] == 0)
	{
		pthread_mutex_init(&(attri_lock[thread_id]), NULL);
		init_attri_lock[thread_id] = 1;
	}	
	pthread_mutex_lock(&(attri_lock[thread_id]));
	AttriDetect(imgdata, w, h, net_idx, result);
	pthread_mutex_unlock(&(attri_lock[thread_id]));

}
void LoadAttriNet(int gpu_idx)
{
	//加载属性识别网络
	if(attri_net_num >= NET_NUM)//当达到网络个数限制时，不加载网络
		return;
	if(human_attri_init[gpu_idx] == 0)//此gpu未加载行人属性网络
	{
		LoadAttriNet(deploy_file, trained_file, gpu_idx, attri_net_num);//加载行人属性网络
		human_attri_init[gpu_idx] = 1;//设置此gpu有加载网络
	}
	attri_net_num++;//行人属性识别网络
}
///////////////////////////////////////////////////////////////////////////行人属性识别
/*HumanAttribute HumanAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs)
{
	int result[10] = { 0 };
	struct attri_detect_args attri_arg;
	attri_arg.thread_idx = detect_net_idx;
	attri_arg.imgROI = imgROI;
	attri_arg.net_idx = detect_net_idx;
	attri_arg.result = result;
	if(pthread_create(&attri_detect_thread[detect_net_idx], 0, attri_detect_in_thread, &attri_arg)) 
		error("Thread creation failed");
	pthread_join(attri_detect_thread[detect_net_idx], 0);

	detect_net_idx++;
	detect_net_idx = (detect_net_idx >= attri_net_num)? 0 : detect_net_idx; //采用哪个网络进行识别
	HumanAttribute val;
	val.age = result[0];//年龄
	val.sex = result[1];//性别
	val.uppercolor = result[2];//上衣颜色
	val.lowercolor = result[3];//下衣颜色
	val.shape = result[4];//体型
	val.head = result[5];//头顶
	val.glasses = result[6];//眼镜
	val.upstyle = result[7];//上衣类型
	val.lowerstyle = result[8];//下衣类型
	val.face = result[9];//面向
	if(imgROI)
	{
		cvReleaseImage(&imgROI);
		imgROI = NULL;
	}
	return val;
}*/
HumanAttribute HumanAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs)
{
	int result[10] = { 0 };
	if(init_attri_lock[detect_net_idx] == 0)
	{
		pthread_mutex_init(&(attri_lock[detect_net_idx]), NULL);
		init_attri_lock[detect_net_idx] = 1;
	}	
	pthread_mutex_lock(&(attri_lock[detect_net_idx]));
	AttriDetect((unsigned char*)imgROI->imageData, imgROI->width, imgROI->height, detect_net_idx, result);
	pthread_mutex_unlock(&(attri_lock[detect_net_idx]));
	detect_net_idx++;
	detect_net_idx = (detect_net_idx >= attri_net_num)? 0 : detect_net_idx; //采用哪个网络进行识别
	HumanAttribute val;
	val.age = result[0];//年龄
	val.sex = result[1];//性别
	val.uppercolor = result[2];//上衣颜色
	val.lowercolor = result[3];//下衣颜色
	val.shape = result[4];//体型
	val.head = result[5];//头顶
	val.glasses = result[6];//眼镜
	val.upstyle = result[7];//上衣类型
	val.lowerstyle = result[8];//下衣类型
	val.face = result[9];//面向
	if(imgROI)
	{
		cvReleaseImage(&imgROI);
		imgROI = NULL;
	}
	return val;
}
///////////////////////////////////////////////////////////////////////////单车属性识别
BicycleAttribute BicycleAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs)
{

	/*int result[10] = { 0 };
	AttriDetect((unsigned char*)imgROI, imgROI->width, imgROI->height, detect_net_idx, result);
	detect_net_idx++;
	detect_net_idx = (detect_net_idx >= attri_net_num)? 0 : detect_net_idx; //采用哪个网络进行识别*/
	BicycleAttribute val;
	return val;
}
#else
#include "pthread.h"
#include "Python.h"
#include <numpy/arrayobject.h>
//////////////////////////////////////////////////////////////////////////////加载python
PyObject *pModule;
PyGILState_STATE gstate;
int AttributeDetectWidth = 0;//属性识别宽度
int AttributeDetectHeight = 0;//属性识别高度
int init_numpy()
{
	import_array();
	return 1;
}
void py_attri_init()
{
	Py_Initialize();
	if ( !Py_IsInitialized() ) {
		printf("init err\n");
	}else{
		printf("init ok\n");
	}
	printf("finding ...\n");
	init_numpy();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	if(pModule)
		Py_DECREF(pModule);
	pModule = PyImport_ImportModule("tttest");
	if ( !pModule ) {
		printf("can't find .py");
	}else{
		printf("py found\n");
	}
	//加载行人属性网络
	PyObject* result;
	PyObject* pFunc;
	pFunc = PyObject_GetAttrString(pModule, "init");
	result = PyObject_CallObject(pFunc, NULL);//初始化，加载行人属性网络，返回行人属性输入图像尺寸
	PyObject* ret_objs;
	PyArg_Parse(result, "O!", &PyList_Type, &ret_objs);
	AttributeDetectWidth = PyLong_AsLong(PyList_GetItem(ret_objs,0));
	AttributeDetectHeight = PyLong_AsLong(PyList_GetItem(ret_objs,1));
	Py_DECREF(result);
	Py_DECREF(pFunc);
	PyEval_InitThreads(); 
	PyEval_ReleaseThread(PyThreadState_Get()); 

}
///////////////////////////////////////////////////////////////////////////行人属性识别
HumanAttribute HumanAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs)
{
	int width = AttributeDetectWidth;//行人属性识别宽度
	int height = AttributeDetectHeight;//行人属性识别高度
	HumanAttribute val;
	IplImage* imgROIResize = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	cvResize(imgROI, imgROIResize, CV_INTER_LINEAR);
	//cvSaveImage("roi.jpg", imgROIResize, 0);
	//传数据给python,进行检测
	unsigned char* imagedata = (unsigned char *)malloc(width * height * 3);
	memcpy(imagedata, imgROIResize->imageData, width * height * 3);
	npy_intp Dims[3]= { height, width, 3}; //给定维度信息
	gstate = PyGILState_Ensure();   //如果没有GIL，则申请获取GIL
	//Py_BEGIN_ALLOW_THREADS;
	//Py_BLOCK_THREADS;
	PyObject* PyListRGB = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, imagedata);
	PyObject* ArgList = PyTuple_New(1);
	PyTuple_SetItem(ArgList, 0, PyListRGB);//将PyList对象放入PyTuple对象中
	PyObject* pFunc = PyObject_GetAttrString(pModule, "classify");
	PyObject* Pyresult = PyObject_CallObject(pFunc, ArgList);//调用函数，完成传递
	PyObject* ret_objs;
	PyArg_Parse(Pyresult, "O!", &PyList_Type, &ret_objs);
	val.age = PyLong_AsLong(PyList_GetItem(ret_objs, 0));//年龄
	val.sex = PyLong_AsLong(PyList_GetItem(ret_objs, 1));//性别
	val.uppercolor = PyLong_AsLong(PyList_GetItem(ret_objs, 2));//上衣颜色
	val.lowercolor = PyLong_AsLong(PyList_GetItem(ret_objs, 3));//下衣颜色
	val.shape = PyLong_AsLong(PyList_GetItem(ret_objs, 4));//体型
	val.head = PyLong_AsLong(PyList_GetItem(ret_objs, 5));//头顶
	val.glasses = PyLong_AsLong(PyList_GetItem(ret_objs, 6));//眼镜
	val.upstyle = PyLong_AsLong(PyList_GetItem(ret_objs, 7));//上衣类型
	val.lowerstyle = PyLong_AsLong(PyList_GetItem(ret_objs, 8));//下衣类型
	val.face = PyLong_AsLong(PyList_GetItem(ret_objs, 9));//面向
	//printf("age = %d, sex = %d, uppercolor = %d, lowercolor = %d,shape = %d, head = %d, glasses = %d, upstyle = %d, lowerstyle =%d, face = %d\n", val.age, val.sex, val.uppercolor, val.lowercolor, val.shape, val.head, val.glasses, val.upstyle, val.lowerstyle, val.face);
	if(Pyresult)
		Py_DECREF(Pyresult);
	if(ArgList)
		Py_DECREF(ArgList);
	if(pFunc)
		Py_DECREF(pFunc);
	//Py_UNBLOCK_THREADS;  
	//Py_END_ALLOW_THREADS; 
	PyGILState_Release(gstate);    //释放当前线程的GIL
	if(imagedata)
	{
		free(imagedata);
		imagedata = NULL;
	}
	if(imgROIResize)
	{
		cvReleaseImage(&imgROIResize);
		imgROIResize = NULL;
	}
	if(imgROI)
	{
		cvReleaseImage(&imgROI);
		imgROI = NULL;
	}
	return val;
}
///////////////////////////////////////////////////////////////////////////单车属性识别
BicycleAttribute BicycleAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs)
{

	int width = AttributeDetectWidth;//单车属性识别宽度
	int height = AttributeDetectHeight;//单车属性识别高度
	BicycleAttribute val;
	/*IplImage* imgROIResize = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	cvResize(imgROI, imgROIResize, CV_INTER_LINEAR);
	//cvSaveImage("roi.jpg", imgROIResize, 0);
	//传数据给python,进行检测
	unsigned char* imagedata = (unsigned char *)malloc(width * height * 3);
	memcpy(imagedata, imgROIResize->imageData, width * height * 3);
	npy_intp Dims[3]= { height, width, 3}; //给定维度信息
	gstate = PyGILState_Ensure();   //如果没有GIL，则申请获取GIL
	Py_BEGIN_ALLOW_THREADS;
	Py_BLOCK_THREADS;
	PyObject* PyListRGB = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, imagedata);
	PyObject* ArgList = PyTuple_New(1);
	PyTuple_SetItem(ArgList, 0, PyListRGB);//将PyList对象放入PyTuple对象中
	PyObject* pFunc = PyObject_GetAttrString(pModule, "classify");
	PyObject* Pyresult = PyObject_CallObject(pFunc, ArgList);//调用函数，完成传递
	PyObject* ret_objs;
	PyArg_Parse(Pyresult, "O!", &PyList_Type, &ret_objs);
	if(Pyresult)
		Py_DECREF(Pyresult);
	if(ArgList)
		Py_DECREF(ArgList);
	if(pFunc)
		Py_DECREF(pFunc);
	Py_UNBLOCK_THREADS;  
	Py_END_ALLOW_THREADS; 
	PyGILState_Release(gstate);    //释放当前线程的GIL
	if(imagedata)
	{
		free(imagedata);
		imagedata = NULL;
	}
	if(imgROIResize)
	{
		cvReleaseImage(&imgROIResize);
		imgROIResize = NULL;
	}*/
	return val;
}
#endif
//void HumanAttributeDetect(ALGCFGS *pCfgs, IplImage* img)//行人属性检测分析
//{
//	int i = 0, j = 0;
//	int val1 = 0, val2 = 0;
//	//设置行人属性框为未检测
//	for(i = 0; i < pCfgs->uPersonNum; i++)
//	{
//		pCfgs->PersonAttributeBox[i].detected = FALSE;
//	}
//	for(i = 0; i < pCfgs->event_targets_size; i++)
//	{
//		if(strcmp(pCfgs->event_targets[i].names, "person") != 0)
//			continue;
//		if(pCfgs->event_targets[i].continue_num < 10)//开始一段时间不进行行人属性识别
//			continue;
//		//进行行人属性识别
//		if(pCfgs->event_targets[i].attribute_detected == FALSE)//此目标没有进行行人属性识别
//		{
//			//printf("detect person %d,%d,%d\n", pCfgs->event_targets[i].detected, (pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height), img->height / 2);
//			//val1 = pCfgs->k * (pCfgs->event_targets[i].box.x) + pCfgs->b - pCfgs->event_targets[i].box.y;
//			//val2 = pCfgs->k * (pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width) + pCfgs->b - pCfgs->event_targets[i].box.y - pCfgs->event_targets[i].box.height;
//			//if(pCfgs->event_targets[i].detected && (val1 * val2 < 0))//检测到，并且检测框与检测线相交
//			int bottom = pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height;
//			//if(pCfgs->event_targets[i].detected && bottom > img->height / 2 && bottom < (img->height - 10))//检测线到图像下端,并且不在图像边界
//			{
//				CvRect roi = cvRect(pCfgs->event_targets[i].box.x, pCfgs->event_targets[i].box.y, pCfgs->event_targets[i].box.width, pCfgs->event_targets[i].box.height);
//				IplImage* imgROI = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 3);
//				//设置ROI区域  
//				/*cvSetImageROI(img, roi);   
//				printf("copy start\n");
//				cvCopy(img, imgROI, NULL);  
//				printf("copy end\n");
//				cvResetImageROI(img);*/
//				for(int ii = 0; ii < roi.height; ii++)
//				{
//					memcpy(imgROI->imageData + ii * imgROI->widthStep, img->imageData + (ii + roi.y) * img->widthStep + roi.x * 3, roi.width * 3);
//				}
//				HumanAttribute val = HumanAttributeRecognition(imgROI, pCfgs);//行人属性识别
//				if(imgROI)
//				{
//					cvReleaseImage(&imgROI);
//					imgROI = NULL;
//				}
//				pCfgs->event_targets[i].attribute_detected = TRUE;
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.age = val.age;//年龄
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.sex = val.sex;//性别
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.uppercolor = val.uppercolor;//上衣颜色
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.lowercolor = val.lowercolor;//下衣颜色
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.shape = val.shape;//体型
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.head = val.head;//头顶
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.glasses = val.glasses;//眼镜
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.upstyle = val.upstyle;//上衣类型
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.lowerstyle = val.lowerstyle;//下衣类型
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.face = val.face;//面向
//				//根据标准高度求行人身高
//				int height = (float)pCfgs->event_targets[i].box.height / (float)StandardHeight * 170;
//				height = (height > 180)? 180 : height;
//				height = (height < 150)? 150 : height;
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.height = height;//身高
//				//计算速度
//				float speed = 0;
//				int pos1 = pCfgs->event_targets[i].trajectory[0].y;
//				int pos2 = pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height / 2;
//				float len = (pos1 - pos2) * 1.7 / (pCfgs->event_targets[i].box.height);//假设人的身高为1.7m,比例求出人的运动实际距离
//				speed = len * 3.6 / (pCfgs->currTime - pCfgs->event_targets[i].start_time);
//				speed = (speed < 0)? -speed : speed;
//				speed = speed + rand() % 5;//速度
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.speed = (speed > 15)? 15 : speed;
//				printf("speed = %d\n", pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.speed);
//				//计算行驶方向,下行为0，上行为1
//				int direction = (pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num - 1].y < pCfgs->event_targets[i].trajectory[0].y)? 1 : 0;//方向
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.direction = direction;
//				//判断是否骑行
//				bool isCycling = FALSE;
//				for( j = 0; j < pCfgs->event_targets_size; j++)
//				{
//					if(strcmp(pCfgs->event_targets[j].names, "bicycle") != 0 && strcmp(pCfgs->event_targets[j].names, "motorbike") != 0)
//						continue;
//					CRect rct;
//					rct.x = pCfgs->event_targets[j].box.x + 10;
//					rct.y = pCfgs->event_targets[j].box.y;
//					rct.width = pCfgs->event_targets[j].box.width - 20;
//					rct.height = pCfgs->event_targets[j].box.height;
//					int overlapratio = overlapRatio(rct, pCfgs->event_targets[i].box);
//					if(overlapratio > 5)
//					{
//						isCycling = TRUE;//骑行
//						break;
//					}
//				}
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.isCycling = isCycling;//是否骑行
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.box = pCfgs->event_targets[i].box;
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.flag = 1;//新检测属性
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].detected = TRUE;
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].uBoxID = pCfgs->event_targets[i].target_id;
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].ReportTime = pCfgs->currTime;
//				pCfgs->PersonAttributeBox[pCfgs->uPersonNum].lost_detected = pCfgs->event_targets[i].lost_detected;
//				pCfgs->uPersonNum++;
//			}
//		}
//		else//此目标已经进行了行人属性识别
//		{
//			for(j = 0; j < pCfgs->uPersonNum; j++)
//			{
//				if(pCfgs->PersonAttributeBox[j].uBoxID == pCfgs->event_targets[i].target_id)//同一目标
//				{
//					pCfgs->PersonAttributeBox[j].AttributeInfo.box = pCfgs->event_targets[i].box;
//					pCfgs->PersonAttributeBox[j].lost_detected = pCfgs->event_targets[i].lost_detected;
//					pCfgs->PersonAttributeBox[j].AttributeInfo.flag = 0;//跟踪属性
//					pCfgs->PersonAttributeBox[j].detected = TRUE;
//					break;
//				}
//			}
//
//		}
//	}
//	//去除没有检测到的行人属性框
//	for(i = 0; i < pCfgs->uPersonNum; i++)
//	{
//		if(pCfgs->PersonAttributeBox[i].detected == FALSE)
//		{
//			for( j = i + 1; j < pCfgs->uPersonNum; j++)
//			{
//				pCfgs->PersonAttributeBox[j - 1] = pCfgs->PersonAttributeBox[j];
//			}
//			i--;
//			pCfgs->uPersonNum--;
//		}
//		else
//		{
//			if(pCfgs->currTime - pCfgs->PersonAttributeBox[i].ReportTime > (5 * 60))//上报间隔为5分钟
//			{
//				pCfgs->PersonAttributeBox[i].AttributeInfo.flag = 1;//上报
//				pCfgs->PersonAttributeBox[i].ReportTime = pCfgs->currTime;
//			}
//		}
//	}
//
//}
void HumanAttributeDetect(ALGCFGS *pCfgs, IplImage* img)//行人属性检测分析
{
	int i = 0, j = 0, k = 0;
	int val1 = 0, val2 = 0;
	//设置行人属性框为未检测
	for(i = 0; i < pCfgs->uPersonNum; i++)
	{
		pCfgs->PersonAttributeBox[i].detected = FALSE;
	}
	for(i = 0; i < pCfgs->objPerson_size; i++)
	{
		if(strcmp(pCfgs->objPerson[i].names, "person") != 0)
			continue;
		if(pCfgs->objPerson[i].cal_flow == FALSE)//没有进行行人统计，不进行行人属性识别
			continue;
		//进行行人属性识别
		if(pCfgs->objPerson[i].attribute_detected == FALSE)//此目标没有进行行人属性识别
		{
			CvRect roi = cvRect(pCfgs->objPerson[i].box.x, pCfgs->objPerson[i].box.y, pCfgs->objPerson[i].box.width, pCfgs->objPerson[i].box.height);
			IplImage* imgROI = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 3);
			//设置ROI区域  
			/*cvSetImageROI(img, roi);   
			printf("copy start\n");
			cvCopy(img, imgROI, NULL);  
			printf("copy end\n");
			cvResetImageROI(img);*/
			for(int ii = 0; ii < roi.height; ii++)
			{
				memcpy(imgROI->imageData + ii * imgROI->widthStep, img->imageData + (ii + roi.y) * img->widthStep + roi.x * 3, roi.width * 3);
			}
			HumanAttribute val = HumanAttributeRecognition(imgROI, pCfgs);//行人属性识别
			pCfgs->objPerson[i].attribute_detected = TRUE;
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.age = val.age;//年龄
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.sex = val.sex;//性别
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.uppercolor = val.uppercolor;//上衣颜色
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.lowercolor = val.lowercolor;//下衣颜色
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.shape = val.shape;//体型
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.head = val.head;//头顶
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.glasses = val.glasses;//眼镜
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.upstyle = val.upstyle;//上衣类型
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.lowerstyle = val.lowerstyle;//下衣类型
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.face = val.face;//面向
			//根据标准高度求行人身高
			int height = (float)pCfgs->objPerson[i].box.height / (float)StandardHeight * 170;
			height = (height > 180)? 180 : height;
			height = (height < 150)? 150 : height;
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.height = height;//身高
			//计算速度
			float speed = 0;
			int pos1 = pCfgs->objPerson[i].trajectory[0].y;
			int pos2 = pCfgs->objPerson[i].box.y + pCfgs->objPerson[i].box.height / 2;
			float len = (pos1 - pos2) * 1.7 / (pCfgs->objPerson[i].box.height);//假设人的身高为1.7m,比例求出人的运动实际距离
			speed = len * 3.6 / (pCfgs->currTime - pCfgs->objPerson[i].start_time);
			speed = (speed < 0)? -speed : speed;
			speed = speed + rand() % 5;//速度
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.speed = (speed > 15)? 15 : speed;
			printf("speed = %d\n", pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.speed);
			//计算行驶方向,下行为0，上行为1
			int direction = (pCfgs->objPerson[i].trajectory[pCfgs->objPerson[i].trajectory_num - 1].y < pCfgs->objPerson[i].trajectory[0].y)? 1 : 0;//方向
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.direction = direction;
			//判断是否骑行
			bool isCycling = FALSE;
			for( j = 0; j < pCfgs->classes; j++)
			{
				if(strcmp(pCfgs->detClasses[j].names, "bicycle") != 0 && strcmp(pCfgs->detClasses[j].names, "motorbike") != 0)
					continue;
				for(k = 0; k < pCfgs->detClasses[j].classes_num; k++)
				{
					CRect rct;
					rct.x = pCfgs->detClasses[j].box[k].x + 10;
					rct.y = pCfgs->detClasses[j].box[k].y;
					rct.width = pCfgs->detClasses[j].box[k].width - 20;
					rct.height = pCfgs->detClasses[j].box[k].height;
					int overlapratio = overlapRatio(rct, pCfgs->objPerson[i].box);
					if(overlapratio > 5)
					{
						isCycling = TRUE;//骑行
						break;
					}
				}
				if(isCycling)
					break;
			}
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.isCycling = isCycling;//是否骑行
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.box = pCfgs->objPerson[i].box;
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].AttributeInfo.flag = 1;//新检测属性
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].detected = TRUE;
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].uBoxID = pCfgs->objPerson[i].target_id;
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].ReportTime = pCfgs->currTime;
			pCfgs->PersonAttributeBox[pCfgs->uPersonNum].lost_detected = pCfgs->objPerson[i].lost_detected;
			pCfgs->uPersonNum++;
		}
		else//此目标已经进行了行人属性识别
		{
			for(j = 0; j < pCfgs->uPersonNum; j++)
			{
				if(pCfgs->PersonAttributeBox[j].uBoxID == pCfgs->objPerson[i].target_id)//同一目标
				{
					pCfgs->PersonAttributeBox[j].AttributeInfo.box = pCfgs->objPerson[i].box;
					pCfgs->PersonAttributeBox[j].lost_detected = pCfgs->objPerson[i].lost_detected;
					pCfgs->PersonAttributeBox[j].AttributeInfo.flag = 0;//跟踪属性
					pCfgs->PersonAttributeBox[j].detected = TRUE;
					break;
				}
			}

		}
	}
	//去除没有检测到的行人属性框
	for(i = 0; i < pCfgs->uPersonNum; i++)
	{
		if(pCfgs->PersonAttributeBox[i].detected == FALSE)
		{
			for( j = i + 1; j < pCfgs->uPersonNum; j++)
			{
				pCfgs->PersonAttributeBox[j - 1] = pCfgs->PersonAttributeBox[j];
			}
			i--;
			pCfgs->uPersonNum--;
		}
		else
		{
			if(pCfgs->currTime - pCfgs->PersonAttributeBox[i].ReportTime > (5 * 60))//上报间隔为5分钟
			{
				pCfgs->PersonAttributeBox[i].AttributeInfo.flag = 1;//上报
				pCfgs->PersonAttributeBox[i].ReportTime = pCfgs->currTime;
			}
		}
	}

}
void BicycleAttributeDetect(ALGCFGS *pCfgs, IplImage* img)//单车属性检测分析
{
	int i = 0, j = 0;
	int val1 = 0, val2 = 0;
	//设置单车属性框为未检测
	for(i = 0; i < pCfgs->uBicycleNum; i++)
	{
		pCfgs->BikeAttributeBox[i].detected = FALSE;
	}
	for(i = 0; i < pCfgs->event_targets_size; i++)
	{
		if(strcmp(pCfgs->event_targets[i].names, "bicycle") != 0)
			continue;
		//进行单车属性识别
		if(pCfgs->event_targets[i].attribute_detected == FALSE)//此目标没有进行单车属性识别
		{
			//val1 = pCfgs->k * (pCfgs->event_targets[i].box.x) + pCfgs->b - pCfgs->event_targets[i].box.y;
			//val2 = pCfgs->k * (pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width) + pCfgs->b - pCfgs->event_targets[i].box.y - pCfgs->event_targets[i].box.height;
			//if(pCfgs->event_targets[i].detected && (val1 * val2 < 0))//检测到，并且检测框与检测线相交
			if(pCfgs->event_targets[i].detected && (pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height) > img->height / 2)//检测线到图像下端
			{
				CvRect roi = cvRect(pCfgs->event_targets[i].box.x, pCfgs->event_targets[i].box.y, pCfgs->event_targets[i].box.width, pCfgs->event_targets[i].box.height);
				IplImage* imgROI = cvCreateImage(cvSize(roi.width, roi.height), IPL_DEPTH_8U, 3);
				//设置ROI区域  
				/*cvSetImageROI(img, roi);   
				printf("copy start\n");
				cvCopy(img, imgROI, NULL);  
				printf("copy end\n");
				cvResetImageROI(img);*/
				for(int ii = 0; ii < roi.height; ii++)
				{
					memcpy(imgROI->imageData + ii * imgROI->widthStep, img->imageData + (ii + roi.y) * img->widthStep + roi.x * 3, roi.width * 3);
				}
				BicycleAttribute val = BicycleAttributeRecognition(imgROI, pCfgs);//单车属性识别
				if(imgROI)
				{
					cvReleaseImage(&imgROI);
					imgROI = NULL;
				}
				pCfgs->event_targets[i].attribute_detected = TRUE;
				//pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].AttributeInfo.brand = val.brand;//品牌
				pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].AttributeInfo.colour = val.colour;//颜色
				//根据单车和人是否相交进行判断
				/*bool isCycling = FALSE;
				for( j = 0; j < pCfgs->event_targets_size; j++)
				{
					if(strcmp(pCfgs->event_targets[j].names, "person") != 0)
						continue;
					CRect rct;
					rct.x = pCfgs->event_targets[i].box.x + 10;
					rct.y = pCfgs->event_targets[i].box.y;
					rct.width = pCfgs->event_targets[i].box.width - 20;
					rct.height = pCfgs->event_targets[i].box.height;
					int overlapratio = overlapRatio(rct, pCfgs->event_targets[j].box);
					if(overlapratio > 5)
					{
						isCycling = TRUE;//骑行
						break;
					}
				}
				pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].AttributeInfo.isCycling = isCycling;//是否骑行*/
				pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].AttributeInfo.box = pCfgs->event_targets[i].box;
				pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].AttributeInfo.flag = 1;//新检测属性
				pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].detected = TRUE;
				pCfgs->BikeAttributeBox[pCfgs->uBicycleNum].uBoxID = pCfgs->event_targets[i].target_id;

				pCfgs->uBicycleNum++;
			}
		}
		else//此目标已经进行了单车属性识别
		{
			for(j = 0; j < pCfgs->uBicycleNum; j++)
			{
				if(pCfgs->BikeAttributeBox[j].uBoxID == pCfgs->event_targets[i].target_id)//同一目标
				{
					pCfgs->BikeAttributeBox[j].AttributeInfo.box = pCfgs->event_targets[i].box;
					pCfgs->BikeAttributeBox[j].AttributeInfo.flag = 0;//跟踪属性
					pCfgs->BikeAttributeBox[j].detected = TRUE;
					break;
				}
			}

		}
	}
	//去除没有检测到的单车属性框
	for(i = 0; i < pCfgs->uBicycleNum; i++)
	{
		if(pCfgs->BikeAttributeBox[i].detected == FALSE)
		{
			for( j = i + 1; j < pCfgs->uBicycleNum; j++)
			{
				pCfgs->BikeAttributeBox[j - 1] = pCfgs->BikeAttributeBox[j];
			}
			i--;
			pCfgs->uBicycleNum--;
		}
	}

}
///////////////////////////////////////////////////////////////////////////
void attri_init()
{
	if(attri_init_flag == 0)//初始化python
	{
#ifdef USE_PYTHON
		py_attri_init();
#else
		memset(human_attri_init, 0, MAX_GPU_NUM * sizeof(int));//将gpu设置为未加载行人属性网络
#endif
	}
	attri_init_flag++;
}
#endif