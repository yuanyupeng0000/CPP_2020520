
#include <stdio.h>
#include "include/PlateRecognize.h"
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <dirent.h>
using namespace cv;
#define MAX_PLATE_NUM 100
int flag = -1;
int main(int argc, char** argv)
{
	flag = LoadPlateNet();
#ifdef DETECT_IMAGE
	std::string file = argv[1];
	 cv::Mat image = cv::imread(file);
	 struct timeval start_time, end_time;
	 gettimeofday( &start_time, NULL );
	 PlateInfo plateInfo[MAX_PLATE_NUM];
	 int platenum = PlateDetectandRecognize(image, 0, plateInfo, flag);
	 for(int i = 0; i < platenum; i++)
	 {
		 printf("num = %d, name = %s,confidence = %f\n", i, plateInfo[i].plateName, plateInfo[i].confidence);
		 /*cv::Rect rct = Rect(plateInfo[i].plateRect.x, plateInfo[i].plateRect.y, plateInfo[i].plateRect.width, plateInfo[i].plateRect.height);
		 Mat plateROI;
		 image(rct).copyTo(plateROI);
		 imwrite("plate0.jpg", plateROI);
		 PlateInfo plateInfo1 = PlateRecognizeOnly(plateROI, 0, flag);
		 printf("num = %d, name = %s,confidence = %f\n", i, plateInfo1.plateName, plateInfo1.confidence);*/
	 }
	 
	 //PlateInfo plateInfo = PlateRecognizeOnly(images, 0, flag);
	 gettimeofday(&end_time, NULL);
	 printf("image size = [%d,%d],time = %f\n",image.cols,image.rows,(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec)/1000000.0);
	 if(argc > 2)
	 {
		 cv::imshow("image",image);
		 cv::waitKey(0);
	 }
#else
	char *basePath = (argc > 1) ? argv[1]: 0;
	char *savePath = (argc > 2) ? argv[2]: 0;
	DIR *dir;
	struct dirent *ptr;
	char base[1000];
	if((dir=opendir(basePath)) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}
	while ((ptr=readdir(dir)) != NULL)
	{
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
			continue;
		else if(ptr->d_type == 8)    ///file
		{
			printf("d_name:%s/%s\n",basePath,ptr->d_name);
			float * result = (float *)calloc( 100 * 6, sizeof(float));
			char filename[50];
			strcpy(filename, basePath);
			strcat(filename, "/");
			strcat(filename, ptr->d_name);
			cv::Mat image = cv::imread(filename);
			struct timeval start_time, end_time;
			gettimeofday( &start_time, NULL );
			PlateInfo plateInfo[MAX_PLATE_NUM];
			int platenum = PlateDetectandRecognize(image, 0, plateInfo, flag);
			for(int i = 0; i < platenum; i++)
			{
				printf("num = %d, name = %s,confidence = %f\n", i, plateInfo[i].plateName, plateInfo[i].confidence);
				cv::Rect rct = Rect(plateInfo[i].plateRect.x, plateInfo[i].plateRect.y, plateInfo[i].plateRect.width, plateInfo[i].plateRect.height);
				cv::rectangle(image,rct,Scalar(0,0,255),3,8,0);
				/*Mat plateROI;
				image(rct).copyTo(plateROI);
				PlateInfo plateInfo1 = PlateRecognizeOnly(plateROI, 0, flag);
				printf("num = %d, name = %s,confidence = %f\n", i, plateInfo1.plateName, plateInfo1.confidence);*/
			}

			//PlateInfo plateInfo = PlateRecognizeOnly(image, 0, flag);
			gettimeofday(&end_time, NULL);
			printf("image size = [%d,%d],time = %f\n",image.cols,image.rows,(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec)/1000000.0);
			char savefile[200];
			strcpy(savefile, savePath);
			strcat(savefile, "/");
			strcat(savefile, ptr->d_name);
			imwrite(savefile, image);
			if(argc > 3)
			{
				cv::imshow("image",image);
				cv::waitKey(0);
			}

		}
		else if(ptr->d_type == 10)    ///link file
			printf("d_name:%s/%s\n",basePath,ptr->d_name);
		else if(ptr->d_type == 4)    ///dir
		{
			memset(base,'\0',sizeof(base));
			strcpy(base,basePath);
			strcat(base,"/");
			strcat(base,ptr->d_name);
			//readFileList(base);
		}
	}
	closedir(dir);
#endif
	FreePlateNet(flag);
	 return 0;
}
