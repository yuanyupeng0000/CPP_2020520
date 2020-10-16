#include "pthread.h"
#ifdef DETECT_GPU
#include "yolo_detector.h"
#else
#include "NCS_detector.h"
#include "intel_dldt.h"
#endif
#ifndef DETECT_GPU
PyObject *pModule;
PyGILState_STATE gstate;
#define MAX_NCS_NUM  50//最大计算棒数量
int NCS_INDEX[MAX_NCS_NUM] = {0};//计算棒是否在运行
int NCS_NUM = 0;//可用的计算棒数目
int initflag = 0;//加载Python，所有相机直执行一次
//#define USE_PYTHON
int init_numpy()
{
	import_array();
	return 1;
}
void py_init()
{
#ifdef USE_PYTHON
	Py_Initialize();
	if ( !Py_IsInitialized() ) {
		printf("init err\n");
	}else{
		printf("init ok\n");
	}
	printf("finding ...\n");
	init_numpy();
	/*pName = PyString_FromString("Test111");


	if(!pName){
	printf("finding err \n");
	}else{
	printf("finding ok \n");
	}*/
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	if(pModule)
		Py_DECREF(pModule);
	pModule = PyImport_ImportModule("mvnc-yolo-tiny");
	if ( !pModule ) {
		printf("can't find .py");
	}else{
		printf("py found\n");
	}
	while(NCS_NUM <= 0)
	{
		PyObject* result;
		PyObject* pFunc;
		//pFunc = PyObject_GetAttrString(pModule, "release");//释放计算棒
		pFunc = PyObject_GetAttrString(pModule, "init");
		result = PyObject_CallObject(pFunc, NULL);////初始化计算棒,得到读取计算棒的个数
		PyArg_Parse(result, "i", &NCS_NUM);
		if(NCS_NUM <= 0)
		{
			printf("no device open,continuing load NCS device\n");
		}
		if(result)
			Py_DECREF(result);
		if(pFunc)
			Py_DECREF(pFunc);
	}
	printf("Load NCS devices ok！ NCS num =%d\n",NCS_NUM);
	PyEval_InitThreads(); 
	PyEval_ReleaseThread(PyThreadState_Get()); 
#else
#ifdef USE_OPENVINO
	printf("[ INFO ] Intel dldt init \n");
	NCS_NUM = intel_dldt_init("FP16/vpu_config.ini");
	printf("[ INFO ] Intel dldt init end. NCS_NUM = %d\n", NCS_NUM);
#endif
#ifdef USE_LIBTORCH
	printf("[ INFO ] Libtorch init \n");
	NCS_NUM = intel_dldt_init("torch/torch_config.ini");
	printf("[ INFO ] Libtorch init end. MAX_NUM = %d\n", NCS_NUM);
#endif
#endif
	memset(NCS_INDEX, 0, MAX_NCS_NUM * sizeof(int));
}
void py_free()
{
	printf("py free\n");
	/*if(pName)
		Py_DECREF(pName);
	if(pModule)
		Py_DECREF(pModule);
	if(pFunc)
		Py_DECREF(pFunc);*/
	// 关闭Python
	//Py_Finalize();
}
int get_ncs_id()//给相机分配NCS ID
{
	int i = 0;
	int NCS_ID = -1;
	if(initflag == 0)
		py_init();
	initflag++;
	//给相机分配计算棒
	NCS_ID = -1;
	for(i = 0; i < NCS_NUM; i++)
	{
		if(NCS_INDEX[i] == 0)
		{
			NCS_ID = i;
			NCS_INDEX[i] = i + 1;
			break;
		}
	}
	if(i == NCS_NUM && NCS_ID < 0)
	{
		printf("no device usable\n");
	}

	printf("ncs_id =%d\n",NCS_ID);
	return NCS_ID;
}
void free_ncs_id(int NCS_ID)//释放相机NCS ID
{
	NCS_INDEX[NCS_ID] = 0;//设置此计算棒不能运行
}
//采用计算棒进行检测
int NCSArithDetect(Mat BGRImage, ALGCFGS* pCfgs, int* rst)
{
	if(pCfgs->NCS_ID  < 0)
	{
		printf("no device\n");
		return 0;
	}
	int nboxes = 0;
	int width = BGRImage.cols;
	int height =BGRImage.rows;
	int size = width * height;
#ifdef  SAVE_VIDEO
	BGRImage.copyTo(img);
#endif //  SAVE_VIDEO
#ifdef USE_PYTHON
	//传数据给python,进行检测
	unsigned char* imagedata = (unsigned char *)malloc(width * height * 3);
	memcpy(imagedata, BGRImage.data, width * height * 3);
	npy_intp Dims[3]= { height, width, 3}; //给定维度信息
	gstate = PyGILState_Ensure();   //如果没有GIL，则申请获取GIL
	Py_BEGIN_ALLOW_THREADS;
	Py_BLOCK_THREADS;
	PyObject* PyListRGB = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, imagedata);
	PyObject* ArgList = PyTuple_New(4);
	PyTuple_SetItem(ArgList, 0, PyListRGB);//将PyList对象放入PyTuple对象中
	PyTuple_SetItem(ArgList, 1, Py_BuildValue("i", width));
	PyTuple_SetItem(ArgList, 2, Py_BuildValue("i", height));
	PyTuple_SetItem(ArgList, 3, Py_BuildValue("i", pCfgs->NCS_ID));
	/*int resize_width = 300;
	int resize_height = 300;
	Mat resizeImage;
	resize(BGRImage, resizeImage, Size(300,300));
	//传数据给python,进行检测
	unsigned char* imagedata = (unsigned char *)malloc(resize_width * resize_height * 3);
	memcpy(imagedata, resizeImage.data, resize_width * resize_height * 3);
	npy_intp Dims[2]= { resize_height, resize_width * 3}; //给定维度信息
	PyObject* PyListRGB = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, imagedata);

	PyObject* ArgList = PyTuple_New(3);
	PyTuple_SetItem(ArgList, 0, PyListRGB);//将PyList对象放入PyTuple对象中
	PyTuple_SetItem(ArgList, 1, Py_BuildValue("i", resize_width));
	PyTuple_SetItem(ArgList, 2, Py_BuildValue("i", resize_height));*/

	//struct timeval start_time, end_time;
	//gettimeofday( &start_time, NULL );
	PyObject* pFunc;
	pFunc = PyObject_GetAttrString(pModule, "process1");
	PyObject* Pyresult = PyObject_CallObject(pFunc, ArgList);//调用函数，完成传递
	//gettimeofday( &end_time, NULL );
	//printf("detect time =%f\n",(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec)/1000000.0);
    PyObject* ret_objs;
	PyArg_Parse(Pyresult, "O!", &PyList_Type, &ret_objs);
	size = PyList_Size(ret_objs);
	int i, j;
	//得到检测框数据
	for(i = 0; i < size/6; i++)
	{

		int class_id, confidence, x, y, w, h;
		class_id = PyLong_AsLong(PyList_GetItem(ret_objs,i*6+0));
		confidence = PyLong_AsLong(PyList_GetItem(ret_objs,i*6+1));
		x = PyLong_AsLong(PyList_GetItem(ret_objs,i*6+2));
		y = PyLong_AsLong(PyList_GetItem(ret_objs,i*6+3));
		w = PyLong_AsLong(PyList_GetItem(ret_objs,i*6+4));
		h = PyLong_AsLong(PyList_GetItem(ret_objs,i*6+5));
		/*if(class_id == 0)//背景
			continue;//mobile net 0 为背景*/
		rst[i * 6 + 0] = class_id;
		rst[i * 6 + 1] = confidence;
		rst[i * 6 + 2] = x;
		rst[i * 6 + 3] = y;
		rst[i * 6 + 4] = w;
		rst[i * 6 + 5] = h;
		nboxes++;
	}
	printf("nboxes = %d\n",nboxes);
	//PyObject_CallObject(pFunc, NULL);
	if(imagedata)
	{
		free(imagedata);
		imagedata = NULL;
	}
	if(Pyresult)
	   Py_DECREF(Pyresult);
	if(ArgList)
		Py_DECREF(ArgList);
	if(pFunc)
		Py_DECREF(pFunc);
	Py_UNBLOCK_THREADS;  
	Py_END_ALLOW_THREADS; 
	PyGILState_Release(gstate);    //释放当前线程的GIL
#else
	std::vector<DetectionObject> objs;
	//printf("%s", "[ INFO ] Call intel_dldt_detect\n");
	intel_dldt_detect(BGRImage, pCfgs->NCS_ID, objs);
	nboxes = objs.size();
	for(int i = 0; i < nboxes; i++)
	{
	    rst[i * 6 + 0] = objs[i].class_id;
	    rst[i * 6 + 1] = int(objs[i].confidence*100);
	    rst[i * 6 + 2] = objs[i].xmin;
	    rst[i * 6 + 3] = objs[i].ymin;
	    rst[i * 6 + 4] = objs[i].xmax - objs[i].xmin;
	    rst[i * 6 + 5] = objs[i].ymax - objs[i].ymin;
	    //printf("class=%d; confidence=%f; xmin=%d; ymin=%d; w=%d; h=%d;\n",objs[i].class_id, objs[i].confidence, objs[i].xmin, objs[i].ymin, objs[i].xmax - objs[i].xmin, objs[i].ymax - objs[i].ymin);
	}
	printf("nboxes = %d\n", nboxes);
#endif
    return nboxes;

}
#else//GPU检测


#endif




