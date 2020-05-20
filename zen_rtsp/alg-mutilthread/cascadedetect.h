
#ifndef _CASCADEDETECT_H_
#define _CASCADEDETECT_H_
#include <vector>
#include "opencv2/core/core.hpp"
using namespace cv;
struct DTreeNode
{
	int featureIdx;
	int threshold; // for ordered features only
	int left;
	int right;
};

struct DTree
{
	int nodeCount;
};

struct Stage
{
	int first;
	int ntrees;
	int threshold;
};
struct Feature
{
	int x;
	int y;
	int width;
	int height;
};
typedef struct CvLBPClassifierCascade
{
	int stage_count;
	int node_count;
	int feature_count;
	bool isStumpBased;
	int stageType;
	int featureType;
	int ncategories;
	Size origWinSize;
	Stage* stages;
	DTreeNode* nodes;
	int* leaves;
	int* subsets;
	Feature* features;
}
CvLBPClassifierCascade;
std::vector<Rect>
	detectMultiScale( const Mat img, unsigned char* maskImg,
	CvLBPClassifierCascade* cascade,int scale_factor32x,
	int min_neighbors, int flags, Size min_size, Size max_size);

#endif

