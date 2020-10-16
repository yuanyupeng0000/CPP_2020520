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

#include "RegionsSingleImage.hpp"

#include <iostream>
#include <vector>

using std::vector;
using std::cerr;
using std::cout;
using std::endl;

RegionsSingleImage::RegionsSingleImage(std::string fName) {
#ifdef __CVLOADIMAGE_WORKING__
  im = cvLoadImage(fName.c_str(), CV_LOAD_IMAGE_COLOR);
#else
  im = readImage(fName.c_str(), CV_LOAD_IMAGE_COLOR);
#endif
  if (im == NULL) {
    cerr << "Could not read image from " << fName << endl;
    assert(false);
  }
  list = new std::vector<Region *>;
}

RegionsSingleImage::RegionsSingleImage(IplImage *I) {
  assert(I != NULL);
  im = cvCreateImage(cvGetSize(I), I->depth, I->nChannels);
  cvCopy(I, im, 0);
  list = new std::vector<Region *>;
}

RegionsSingleImage::~RegionsSingleImage() {
  if (list) {
    for (unsigned int i = 0; i < list->size(); i++)
      if (list->at(i)) delete (list->at(i));
  }
  delete (list);
  cvReleaseImage(&im);
}

unsigned int RegionsSingleImage::length() {
  return (unsigned int)(list->size());
}

Region *RegionsSingleImage::get(int i) { return list->at(i); }

void RegionsSingleImage::set(int i, Region *r) { list->at(i) = r; }

const IplImage *RegionsSingleImage::getImage() { return (const IplImage *)im; }

std::vector<double> *RegionsSingleImage::getUniqueScores() {
  vector<double> *v = new vector<double>;
  v->reserve(list->size());
  for (unsigned int i = 0; i < list->size(); i++)
    v->push_back(list->at(i)->detScore);

  sort(v->begin(), v->end());
  vector<double>::iterator uniElem = unique(v->begin(), v->end());
  v->erase(uniElem, v->end());
  return v;
}
