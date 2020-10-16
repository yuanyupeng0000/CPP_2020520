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

#ifndef  EXAMPLES_MTCNN_EVALUATION_REGIONSSINGLEIMAGE_HPP_
#define  EXAMPLES_MTCNN_EVALUATION_REGIONSSINGLEIMAGE_HPP_

#include <string>
#include <vector>

#include "common.hpp"

#ifndef __XCODE__
#include <highgui.h>
#endif

#include <fstream>
#include <iostream>
#include "OpenCVUtils.hpp"
#include "Region.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

/**
 * Abstract class that specifies a set of regions for an image
 * */
class RegionsSingleImage {
  protected:
  /// Image associated with this set of regions
  IplImage *im;
  /// Vector to hold the list of regions
  std::vector<Region *> *list;

  public:
  /// Constructor: read an image from a file
  explicit RegionsSingleImage(std::string fName);
  /// Constructor: intialize the image for this set of ellipses as I
  explicit RegionsSingleImage(IplImage *I);
  /// Destructor
  ~RegionsSingleImage();

  /// Read the annotaion from the file fName -- Pure virtual
  virtual void read(std::string) = 0;
  /// Read N annotaion from the file stream fs -- Pure virtual
  virtual void read(std::ifstream &, int) = 0;
  /// Display all ellipses -- Pure virtual
  virtual void show() = 0;

  /// Returns the number of regions
  unsigned int length();
  /// Returns the pointer to the i-th region
  Region *get(int i);
  /// Adds the region r at the i-th position in the list
  void set(int i, Region *r);
  /// Returns a const pointer to the image associated with this set of regions
  const IplImage *getImage();
  /// Returns the set of unique detection scores for the regions in this set
  std::vector<double> *getUniqueScores();
};

#endif  // EXAMPLES_MTCNN_EVALUATION_REGIONSSINGLEIMAGE_HPP_
