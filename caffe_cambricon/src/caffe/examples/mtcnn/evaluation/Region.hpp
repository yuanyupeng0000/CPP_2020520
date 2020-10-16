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

#ifndef EXAMPLES_MTCNN_EVALUATION_REGION_HPP_
#define EXAMPLES_MTCNN_EVALUATION_REGION_HPP_

#define REGION_MASK_VALUE 10

#include "common.hpp"

#ifdef __XCODE__
#include <OpenCV/OpenCV.h>
#else
#include <cv.h>
#endif

/**
 * Abstract class for the specification of a region
 *  */
class Region {
  private:
  /// Flag to specify if this region should be used
  bool valid;

  public:
  /// Image used for display and set operations
  IplImage *mask;
  /// Score assigned by an external detector
  double detScore;
  /// Constructor
  explicit Region(IplImage *I);
  /// Destructor
  ~Region();

  /// Returns if the region is valid for use
  bool isValid();
  /// Assigns the validity flag
  void setValid(bool);
  /// Computes the set-intersection between this->mask and r->mask
  double setIntersect(Region *r);
  /// Computes the set-union between this->mask and r->mask
  double setUnion(Region *r);
  /// Display this region -- Not implemented in this abstract class
  virtual IplImage *display(IplImage *, CvScalar color, int lineWidth,
                            const char *text) = 0;
};

#endif  // EXAMPLES_MTCNN_EVALUATION_REGION_HPP_
