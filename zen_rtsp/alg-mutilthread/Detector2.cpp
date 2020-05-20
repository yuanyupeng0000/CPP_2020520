/* runtime include files */
#include <stdio.h>
#include <stdlib.h>

#include "Detector2.h"
#include "cascade_day_xml.h"
#include "cascade_dusk_xml.h"
#include "cascade_night_xml.h"
CvLBPClassifierCascade* cascade_LBP=0;
//bool initial_param_ok = false;
int InitHaarParam(int flag)
{

	int test_stages=17;
	if(flag==4)
	{
		cascade_LBP=(CvLBPClassifierCascade*)&(cascade_day_xml::cascade_data);//day
		test_stages=16;
	}
	else if(flag==2)
	{
		cascade_LBP=(CvLBPClassifierCascade*)&(cascade_night_xml::cascade_data);//night
		test_stages=16;
	}
	else
	{
		cascade_LBP=(CvLBPClassifierCascade*)&(cascade_dusk_xml::cascade_data);//dusk
		test_stages=16;
	}

    if( !cascade_LBP )
    {
        return -1;
    }
	if(cascade_LBP->stage_count>test_stages)
	{
		cascade_LBP->stage_count=test_stages;
		cascade_LBP->node_count=cascade_LBP->stages[test_stages-1].first+cascade_LBP->stages[test_stages-1].ntrees;
	}

    return 0;
}

void alg_opencv_processPC(unsigned char* srcImg,unsigned char* maskImg,int w, int h,Rect* obj_rect,int  *obj_num,int flag,bool& initial_param_ok)
{
	int i;
	if(!initial_param_ok)
	{
		InitHaarParam(flag);
		initial_param_ok=true;
	}
    if( cascade_LBP )
    {
		Mat tempImg(h,w,CV_8U);
		memcpy(tempImg.data, srcImg, w*h * sizeof(unsigned char));
		std::vector<Rect> faces=detectMultiScale(tempImg,maskImg,cascade_LBP,40, 2, 0,
                                      Size(32, 40),Size(120,150));
		if(faces.size())
		{
			for( i = 0; i < faces.size(); i++ )
			{
				obj_rect[i + (*obj_num)].x= faces[i].x;
				obj_rect[i + (*obj_num)].width= faces[i].width;
				obj_rect[i + (*obj_num)].y= faces[i].y;
				obj_rect[i + (*obj_num)].height= faces[i].height;
			}
			(*obj_num)+=faces.size();
			
		}
	}
    return;

}








