/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __OPENCVUTILS_HPP__
#define __OPENCVUTILS_HPP__

#include "OpenCVUtils.hpp"

using std::cerr;
using std::endl;
using std::string;
using std::vector;

void matPrint(string s, const CvArr *M) {
  assert(M != NULL);

  if (!s.empty()) cerr << s;

  CvTypeInfo *info = cvTypeOf(M);
  if (!strcmp(info->type_name, CV_TYPE_NAME_IMAGE)) {
    CvScalar s;
    IplImage *I = const_cast<IplImage *>(reinterpret_cast<const IplImage *>(M));
    for (int i = 0; i < I->height; i++) {
      for (int j = 0; j < I->width; j++) {
        s = cvGet2D(I, i, j);
        cerr << s.val[0] << " ";
      }
      cerr << endl;
    }
  } else if (!strcmp(info->type_name, CV_TYPE_NAME_MAT)) {
    CvMat *M1 = const_cast<CvMat *>(reinterpret_cast<const CvMat *>(M));
    for (int i = 0; i < M1->height; i++) {
      for (int j = 0; j < M1->width; j++) cerr << cvmGet(M1, i, j) << " ";
      cerr << endl;
    }
  } else {
    assert(false);
  }
}

void matRotate(const CvArr *src, CvArr *dst, double angle) {
  float m[6];
  // double factor = (cos(angle*CV_PI/180.) + 1.1)*3;
  double factor = 1;
  CvMat M = cvMat(2, 3, CV_32F, m);
  int w = (const_cast<CvMat *>(reinterpret_cast<const CvMat *> (src)))->width;
  int h = (const_cast<CvMat *>(reinterpret_cast<const CvMat *> (src)))->height;

  m[0] = static_cast<float> (factor * cos(-angle * CV_PI / 180.));
  m[1] = static_cast<float> (factor * sin(-angle * CV_PI / 180.));
  m[2] = (w - 1) * 0.5f;
  m[3] = -m[1];
  m[4] = m[0];
  m[5] = (h - 1) * 0.5f;

  cvGetQuadrangleSubPix(src, dst, &M);
}

void matCopyStuffed(const CvArr *src, CvArr *dst) {
  // TODO(Brian Gerkey): get a flag for default value
  // double tMin, tMax;
  // cvMinMaxLoc(src, &tMin, &tMax);
  cvSet(dst, cvScalar(0));
  CvMat *SMat = const_cast<CvMat *>(reinterpret_cast<const CvMat *> (src));
  CvMat *DMat = const_cast<CvMat *>(reinterpret_cast<const CvMat *> (dst));
  int sRow, dRow, sCol, dCol;

  if (SMat->rows >= DMat->rows) {
    sRow = (SMat->rows - DMat->rows) / 2;
    dRow = 0;
  } else {
    sRow = 0;
    dRow = (DMat->rows - SMat->rows) / 2;
  }

  if (SMat->cols >= DMat->cols) {
    sCol = (SMat->cols - DMat->cols) / 2;
    dCol = 0;
  } else {
    sCol = 0;
    dCol = (DMat->cols - SMat->cols) / 2;
  }

  // cerr << "src start " << sRow << " " << sCol << " dst "  << dRow << " " <<
  // dCol << endl;

  /*
  for(int di =0; di < dRow; di++)
          for(int dj = 0; (dj < DMat->cols) && (dj < SMat->cols) ; dj++)
                  cvmSet(DMat, di, dj, cvmGet(SMat, sRow, dj));

  for(int dj =0; dj < dCol; dj++)
          for(int di = 0; (di < DMat->rows) && (di < SMat->rows) ; di++)
                  cvmSet(DMat, di, dj, cvmGet(SMat, di, sCol));
  */

  for (int si = sRow, di = dRow; (si < SMat->rows && di < DMat->rows);
       si++, di++)
    for (int sj = sCol, dj = dCol; (sj < SMat->cols && dj < DMat->cols);
         sj++, dj++)
      cvmSet(DMat, di, dj, cvmGet(SMat, si, sj));
}

void matNormalize(const CvArr *src, CvArr *dst, double minVal, double maxVal) {
  double tMin, tMax;
  cvMinMaxLoc(src, &tMin, &tMax);
  double scaleFactor = (maxVal - minVal) / (tMax - tMin);
  cvSubS(src, cvScalar(tMin), dst);
  cvConvertScale(dst, dst, scaleFactor, minVal);
}

double matMedian(const CvArr *M) {
  int starti = 0, startj = 0, height, width;
  CvTypeInfo *info = cvTypeOf(M);
  if (!strcmp(info->type_name, CV_TYPE_NAME_IMAGE)) {
    CvRect r =
      cvGetImageROI(const_cast<IplImage *>(reinterpret_cast<const IplImage *>(M)));
    height = r.height;
    width = r.width;
    startj = r.x;
    starti = r.y;
  } else if (!strcmp(info->type_name, CV_TYPE_NAME_MAT)) {
    height = (const_cast<CvMat *>(reinterpret_cast<const CvMat *>(M)))->height;
    width = (const_cast<CvMat *>(reinterpret_cast<const CvMat *>(M)))->width;
  } else {
    assert(false);
  }

  // push elements into a vector
  vector<double> v;
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      v.push_back(cvGet2D(M, i, j).val[0]);
    }

  // sort the vector and return the median element
  std::sort(v.begin(), v.end());

  return *(v.begin() + v.size() / 2);
}

void showImage(string title, const CvArr *M) {
  const char *s = title.c_str();
  cvNamedWindow(s, 0);
  cvMoveWindow(s, 100, 400);
  cvShowImage(s, M);
  cvWaitKey(0);
  cvDestroyWindow(s);
}

// like imagesc
void showImageSc(string title, const CvArr *M, int height, int width) {
  const char *s = title.c_str();
  IplImage *I1;

  CvTypeInfo *info = cvTypeOf(M);
  if (!strcmp(info->type_name, CV_TYPE_NAME_IMAGE)) {
    I1 = const_cast<IplImage *>(reinterpret_cast<const IplImage *>(M));
  } else if (!strcmp(info->type_name, CV_TYPE_NAME_MAT)) {
    CvMat *M2 = cvCloneMat(const_cast<CvMat *>(reinterpret_cast<const CvMat *>(M)));
    matNormalize(M, M2, 0, 255);
    double tMin, tMax;
    cvMinMaxLoc(M2, &tMin, &tMax);
    I1 = cvCreateImage(cvGetSize(M2), IPL_DEPTH_8U, 1);
    cvConvertScale(M2, I1);
  } else {
    assert(false);
  }

  IplImage *I = cvCreateImage(cvSize(height, width), I1->depth, 1);
  cvResize(I1, I);
  cvNamedWindow(s, 0);
  cvMoveWindow(s, 100, 400);
  cvShowImage(s, I);
  cvWaitKey(0);
  cvDestroyWindow(s);
  cvReleaseImage(&I);
  cvReleaseImage(&I1);
}

IplImage *readImage(const char *fileName, int useColorImage) {
#ifdef _WIN32
  IplImage *img = cvLoadImage(fileName, useColorImage);
#else
  // check the extension for jpg files; OpenCV has issues with reading jpg
  // files.
  int randInt = rand(); // NOLINT
  char randIntStr[128];
  sprintf(randIntStr, "%d", randInt); // NOLINT

  string tmpPPMFile("cacheReadImage");
  tmpPPMFile += randIntStr;
  tmpPPMFile += ".ppm";

  string sysCommand = "convert ";
  sysCommand += fileName;
  sysCommand += " " + tmpPPMFile;
  if (system(sysCommand.c_str()) != 0)
    cerr << " Execute sysCommmand failed: " << sysCommand << endl;

  IplImage *img = cvLoadImage(tmpPPMFile.c_str(), useColorImage);
  if (img == NULL) {
    cerr << " Could not read image" << endl;
  }

  sysCommand = "rm -f ";
  sysCommand += tmpPPMFile;
  if (system(sysCommand.c_str()) != 0)
    cerr << " Execute sysCommmand failed: " << sysCommand << endl;
#endif
  return img;
}

#endif
