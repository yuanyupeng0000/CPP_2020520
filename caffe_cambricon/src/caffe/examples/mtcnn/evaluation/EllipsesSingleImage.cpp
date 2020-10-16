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

#include "EllipsesSingleImage.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef __XCODE__
#include <cv.h>
#include <highgui.h>
#include "OpenCVUtils.hpp"
#else
#include <OpenCV/OpenCV.h>
#endif

#ifndef M_PI
#define M_PI 3.14156
#endif

using std::string;
using std::vector;
using std::ifstream;
using std::stringstream;

EllipsesSingleImage::EllipsesSingleImage(string fName)
    : RegionsSingleImage(fName) {}

EllipsesSingleImage::EllipsesSingleImage(IplImage *I) : RegionsSingleImage(I) {}

EllipsesSingleImage::~EllipsesSingleImage() {}

void EllipsesSingleImage::read(string rectFile) {
  ifstream fin(rectFile.c_str());
  if (fin.is_open()) {
    double x, y, t, w, h;

    while (fin >> w >> h >> t >> x >> y) {
      t = (M_PI - t) * 180 / M_PI;
      vector<double> *r = new vector<double>(5);
      double myarray[] = {x, y, t, w, h};
      r->insert(r->begin(), myarray, myarray + 5);
      EllipseR *ell = new EllipseR(NULL, r);
      list->push_back(reinterpret_cast<Region *> (ell));
      delete (r);
    }
  }
  fin.close();
}

void EllipsesSingleImage::read(ifstream &fin, int n) {
  for (int i = 0; i < n; i++) {
    double x, y, t, w, h, sc;

    string line;
    getline(fin, line);
    stringstream ss(line);
    ss >> w >> h >> t >> x >> y >> sc;

    t = (M_PI - t) * 180 / M_PI;
    vector<double> *r = new vector<double>(6);
    double myarray[] = {x, y, t, w, h, sc};
    r->insert(r->begin(), myarray, myarray + 6);
    EllipseR *ell = new EllipseR(NULL, r);
    list->push_back(reinterpret_cast<Region *> (ell));
    delete (r);
  }
}

void EllipsesSingleImage::show() {
  IplImage *mask = cvCreateImage(cvGetSize(im), im->depth, im->nChannels);
  cvCopy(im, mask, 0);
  for (unsigned int i = 0; i < list->size(); i++)
    mask =
        (reinterpret_cast<EllipseR *>(list->at(i)))->display(mask, CV_RGB(255, 0, 0),
         3, NULL);

  showImage("Ellipses", mask);
  cvReleaseImage(&mask);
}
