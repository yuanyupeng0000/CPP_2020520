/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#ifndef EXAMPLES_MTCNN_EVALUATION_ELLIPSER_HPP_
#define EXAMPLES_MTCNN_EVALUATION_ELLIPSER_HPP_

#include <vector>
#ifndef __XCODE__
#include <cv.h>
#endif

#include "common.hpp"
#include "Region.hpp"
/**
 * Specification of an elliptical region
 *  */
class EllipseR : public Region {
  private:
  /// x-position of the center
  double cx;
  /// y-position of the center
  double cy;
  /// orientation of the major axis
  double angle;
  /// half-length of the major axis
  double ra;
  /// half-length of the minor axis
  double rb;

  public:
  /// Constructor
  EllipseR(IplImage *, std::vector<double> *);
  /// Method to add this ellipse of a given color and
  /// line width to an image. If the
  /// last parameter is not NULL, display the text also.
  virtual IplImage *display(IplImage *I, CvScalar color, int lineWidth,
                            const char *text);
};

#endif  // EXAMPLES_MTCNN_EVALUATION_ELLIPSER_HPP_
