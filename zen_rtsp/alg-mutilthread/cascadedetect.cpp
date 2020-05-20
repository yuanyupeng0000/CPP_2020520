//#include "stdafx.h"
//#define  HAVE_TBB 
#include "cascadedetect.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/imgproc/imgproc.hpp"


/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
using namespace std;

//namespace cv
//{

// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
        std::abs(r1.y - r2.y) <= delta &&
        std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
        std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};


void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<int>* levelWeights)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }
    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<int> rejectWeights(nclasses, 0);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();

    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
		int n1 = rweights[i];
		//int n1 = levelWeights ? rejectLevels[i] : rweights[i];
        int w1 = rejectWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold)
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(n1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void groupRectangles(vector<Rect>& rectList, vector<int>& weights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(vector<Rect>& rectList, vector<int>& rejectLevels, vector<int>& levelWeights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}

typedef int sumtype;
#define CALC_SUM_(p0, p1, p2, p3, offset) \
	((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])
inline int calc(sumtype* p[16], int _offset ) 
{
	int cval = CALC_SUM_( p[5], p[6], p[9], p[10], _offset );

	return (CALC_SUM_( p[0], p[1], p[4], p[5], _offset ) >= cval ? 128 : 0) |   // 0
		(CALC_SUM_( p[1], p[2], p[5], p[6], _offset ) >= cval ? 64 : 0) |    // 1
		(CALC_SUM_( p[2], p[3], p[6], p[7], _offset ) >= cval ? 32 : 0) |    // 2
		(CALC_SUM_( p[6], p[7], p[10], p[11], _offset ) >= cval ? 16 : 0) |  // 5
		(CALC_SUM_( p[10], p[11], p[14], p[15], _offset ) >= cval ? 8 : 0)|  // 8
		(CALC_SUM_( p[9], p[10], p[13], p[14], _offset ) >= cval ? 4 : 0)|   // 7
		(CALC_SUM_( p[8], p[9], p[12], p[13], _offset ) >= cval ? 2 : 0)|    // 6
		(CALC_SUM_( p[4], p[5], p[8], p[9], _offset ) >= cval ? 1 : 0);
}
int cvRunLBPClassifierCascade1(CvLBPClassifierCascade* _cascade,int start_stage, int end_stage, int p_offset,sumtype* g_ppsumRectP[300][16])
{
    int i = 0, j = 0;
    int iIW = 0;
    int ntrees;
	int nodeOfs = 0, leafOfs = 0;
	int subsetSize = 8/*(_cascade->ncategories + 31)/32*/;
	int* cascadeSubsets =&(_cascade->subsets[0]);
	int* cascadeLeaves = &(_cascade->leaves[0]);
	Stage* cascadeStages = &(_cascade->stages[0]);
	sumtype** p;
    // 以下计算每个classifier的用到的特征区域的特征值
    for (i = start_stage; i < end_stage; i++)
    {
        long long stage_sum = 0;
        iIW = cascadeStages[i].first;
		ntrees = cascadeStages[i].ntrees;
		nodeOfs=iIW;
		leafOfs=iIW*2;
		for (j = 0; j <ntrees; j++)
		{    
			int c;
			// 计算LBP特征
			//memcpy(p,g_ppsumRectP[nodeOfs],16*sizeof(sumtype));
			p = g_ppsumRectP[nodeOfs];
			c = calc(p,p_offset);
			const int* subset = &cascadeSubsets[nodeOfs*subsetSize];
			stage_sum += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
			nodeOfs++;
			leafOfs += 2;
		}
		if (stage_sum <cascadeStages[i].threshold)
		{
			return i;
		}
	}
    return end_stage;
}

int  
cvRunLBPClassifierCascade0(CvLBPClassifierCascade* _cascade,Size win_size,Point g_ptTemp1,int step,sumtype* g_ppsumRectP[300][16])
{
    int p_offset;
	int i=0,j=0;
    //计算特征点在积分图中的偏移，相当于移动窗口
    p_offset = g_ptTemp1.y * (step >> 2) + g_ptTemp1.x;
	//用检测窗口扫描两遍图像
	//第一遍通过级联两个stage粗略定位目标大致区域，对候选区域进行标定
    int split_stage = 2; //第一次分类用到的最大stage，第二次分类用到的起始stage
    int ret = 0; // 返回的遍历层数
    ret = cvRunLBPClassifierCascade1(_cascade, 0,split_stage, p_offset,g_ppsumRectP);
    if (ret < split_stage)
    {
    	return ret;
    }

    //第二遍对标定的候选区域进行完整筛选，将候选区域放置到队列中
    ret = cvRunLBPClassifierCascade1(_cascade,split_stage, _cascade->stage_count, p_offset,g_ppsumRectP);
    return ret;
}
CV_IMPL void
cvSetImagesForLBPClassifierCascade( CvLBPClassifierCascade* _cascade,
                                     Mat sum,
                                    int scale32x,sumtype* g_ppsumRectP[300][16])
{
    //CV_FUNCNAME("cvSetImagesForLBPClassifierCascade");

   // __BEGIN__;

    int i;
    sumtype *piSumTemp;
    
    unsigned int nSumStep;
    unsigned int nSumStepDiv4;
    piSumTemp = (sumtype*)(sum.data);
    
    nSumStep = sum.step;
    nSumStepDiv4 = nSumStep >> 2;
    
    /* init pointers in haar features according to real window size and
       given image pointers */
	DTreeNode*  node=_cascade->nodes;
	Feature* feature=_cascade->features;

	for(i=0;i<_cascade->node_count;i++)
	{
		int k;
		Rect tr,r;
		k=node[i].featureIdx;
		r.x=feature[k].x;
		r.y=feature[k].y;
		r.width=feature[k].width;
		r.height=feature[k].height;
		tr.x = ( r.x * scale32x +16)>>5;
		tr.width = ( r.width * scale32x+16 )>>5;
		tr.y = ( r.y * scale32x + 16 )>>5;
		tr.height = ( r.height * scale32x +16 )>>5;
	    g_ppsumRectP[i][0] = piSumTemp + nSumStepDiv4 * tr.y + tr.x;
		g_ppsumRectP[i][1] = piSumTemp + nSumStepDiv4 * tr.y + tr.x + tr.width;
		g_ppsumRectP[i][4] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x;
		g_ppsumRectP[i][5] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x + tr.width;
		tr.x += 2*tr.width;
		g_ppsumRectP[i][2] = piSumTemp + nSumStepDiv4 * tr.y + tr.x;
		g_ppsumRectP[i][3] = piSumTemp + nSumStepDiv4 * tr.y + tr.x + tr.width;
		g_ppsumRectP[i][6] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x;
		g_ppsumRectP[i][7] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x + tr.width;
		tr.y += 2*tr.height;
		g_ppsumRectP[i][10] = piSumTemp + nSumStepDiv4 * tr.y + tr.x;
		g_ppsumRectP[i][11] = piSumTemp + nSumStepDiv4 * tr.y + tr.x + tr.width;
		g_ppsumRectP[i][14] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x;
		g_ppsumRectP[i][15] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x + tr.width;
		tr.x -= 2*tr.width;
		g_ppsumRectP[i][8] = piSumTemp + nSumStepDiv4 * tr.y + tr.x;
		g_ppsumRectP[i][9] = piSumTemp + nSumStepDiv4 * tr.y + tr.x + tr.width;
		g_ppsumRectP[i][12] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x;
		g_ppsumRectP[i][13] = piSumTemp + nSumStepDiv4 * (tr.y + tr.height) + tr.x + tr.width;

	}
    //__END__;
}
vector<Rect>
	detectMultiScale( const Mat img, unsigned char* maskImg,
                     CvLBPClassifierCascade* cascade,int scale_factor32x,
                     int min_neighbors, int flags, Size min_size, Size max_size)
{
	const double GROUP_EPS = 0.2;
    int split_stage = 2;

    Mat sum ;
	vector<Rect> objects;
	objects.clear();
	vector<int> rejectLevels;
    int i;
    int result;
    Rect rectTemp; 
	//const int detect_win_size[10][2]={{36,45},{40,50},{48,60},{56,70},{64,80},{76,95},{92,115},{108,135},{128,160},{152,190}};
	const int detect_win_size[10][2]={{32,40},{36,45},{40,50},{48,60},{56,70},{64,80},{76,95},{92,115},{108,135},{128,160}};
	const int factor_array[10]={32,36,40,48,56,64,76,92,108,128};
    //__BEGIN__;

    int factor32x;
    int npass = 2, coi;
	Point g_ptTemp1;
	Mat integralSum;
	sumtype* g_ppsumRectP[300][16];


	sum.create(img.rows + 1, img.cols + 1, CV_32SC1);
	integral(img, sum);
	sum.copyTo(integralSum);
	if ((unsigned)split_stage >= (unsigned)cascade->stage_count)
	{
		split_stage = cascade->stage_count;
		npass = 1;
	}

	for(i=0;i<10;i++ )
	{
		factor32x=factor_array[i];
		//const int ystep32x =64;
		int ystep32x = ( factor32x >= 48 )? 32 : 64;
		Size win_size(detect_win_size[i][0],detect_win_size[i][1]);
        ystep32x = ystep32x * win_size.width /32;//32为分类器检测窗口宽度

		int stop_height = (((img.rows - win_size.height) << 5) + (ystep32x >> 1)) / ystep32x;
		int stop_width =(((img.cols - win_size.width) << 5) + (ystep32x >> 1)) / ystep32x;
		if( win_size.width < min_size.width || win_size.height < min_size.height )
		{
			continue;
		}
		if(win_size.width>max_size.width||win_size.height>max_size.height)
		{
			break;
		}
		if( win_size.width > img.cols-10 || win_size.height>img.rows-10 )
		{
			break;
		}

		cvSetImagesForLBPClassifierCascade( cascade, sum, factor32x,g_ppsumRectP);

        for( int _iy = 0; _iy < stop_height; _iy++)
        {
            int iy = (_iy*ystep32x+16)>>5; // 检测窗口纵坐标步长为2
			int _ix,_xstep = 1, preStep = 1;
            for( _ix = 0; _ix < stop_width; _ix += _xstep )
            {
                int ix = (_ix*ystep32x+16)>>5; // it really should be ystep
				if(maskImg)
				{
					if(*(maskImg+img.cols*iy+ix)==0)
						continue;
				}
                //_xstep = 2; // 检测窗口横坐标按步长为4开始移动
				_xstep = 1 + (preStep < 2); // 检测窗口横坐标按步长为4开始移动
                g_ptTemp1.x = ix;
                g_ptTemp1.y = iy;
				// 确保矩形的有效性，并防止计算窗口出边界
				if (g_ptTemp1.x < 0 || g_ptTemp1.y < 0 
					|| g_ptTemp1.x + win_size.width >= integralSum.cols - 2 
					|| g_ptTemp1.y + win_size.height >= integralSum.rows - 2)
				{
					result = -1;
				}
				else
                {
					result = cvRunLBPClassifierCascade0(cascade,win_size,g_ptTemp1,integralSum.step,g_ppsumRectP);
				}
                if (result ==0)
                {
                    _ix++; // 只通过了第一个分类器的，将步长设为2
                }
                else if (result>=cascade->stage_count) // 通过所有分类器，表示找到目标
                {
                    rectTemp = Rect(ix, iy, win_size.width, win_size.height);
					objects.push_back(rectTemp);
                }
	            preStep = _xstep;
            }

    // gather the results
        }
    }

    if( min_neighbors != 0 )
    {
		groupRectangles( objects, rejectLevels,min_neighbors, GROUP_EPS );
    }

    //__END__;


    return objects;
}


//} // namespace cv

