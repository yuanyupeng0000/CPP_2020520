/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
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

#include "EllipseR.hpp"
#ifndef __XCODE__
#include <cv.h>
#endif
#include <iostream>
#include <vector>
#include "OpenCVUtils.hpp"

EllipseR::EllipseR(IplImage *I, std::vector<double> *v) : Region(I) {
  cx = v->at(0);
  cy = v->at(1);
  angle = v->at(2);
  ra = v->at(3);
  rb = v->at(4);
  detScore = v->at(5);
}

IplImage *EllipseR::display(IplImage *mask, CvScalar color, int lineWidth,
                            const char *text) {
  // draw the ellipse on the mask image
  cvEllipse(mask, cvPointFrom32f(cvPoint2D32f(cx, cy)),
            cvSize(static_cast<int> (ra), static_cast<int> (rb)),
            180 - angle, 0, 360, color, lineWidth);

  if (text != NULL) {
    // add text
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
    cvPutText(mask, text, cvPointFrom32f(cvPoint2D32f(cx, cy)), &font, color);
  }
  return mask;
}
