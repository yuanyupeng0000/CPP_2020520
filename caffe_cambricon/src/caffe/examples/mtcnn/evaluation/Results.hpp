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

#ifndef EXAMPLES_MTCNN_EVALUATION_RESULTS_HPP_
#define EXAMPLES_MTCNN_EVALUATION_RESULTS_HPP_

#include <fstream>
#include <string>
#include <vector>
#include "MatchPair.hpp"
#include "RegionsSingleImage.hpp"

using std::vector;
using std::string;

/**
 * Specifies the cumulative result statistics
 * */
class Results {
  private:
  /// number of annotated regions (TP+FN)
  unsigned int N;
  /// threshold used for computing this result
  double scoreThreshold;
  /// True positives -- continuous
  double TPCont;
  /// True positives -- discrete
  double TPDisc;
  /// False positives -- discrete
  double FP;
  /// Name of the image
  string imName;

  public:
  /// Constructor
  Results();
  /// Constructor -- copy the contents of *r
  explicit Results(Results *r);
  /// Constructor -- copy the contents of *r and add N2 to N
  Results(Results *r, int N2);
  /// Constructor -- merge r1 and r2
  Results(Results *r1, Results *r2);
  /// Constructor
  Results(string imName, double threshold, vector<MatchPair *> *matchPairs,
          RegionsSingleImage *annot, RegionsSingleImage *det);
  /// Return a vector of results with combined statistics from the two
  /// vectors rv1 and rv2
  vector<Results *> *merge(vector<Results *> *rv1, vector<Results *> *rv2);
  /// print this result into the ostream os
  void print(std::ostream &os);
  /// save the ROC curve computed from rv into the file outFile
  void saveROC(string outFile, vector<Results *> *rv);

  /// get N
  unsigned int getN();
};

#endif  // EXAMPLES_MTCNN_EVALUATION_RESULTS_HPP_
