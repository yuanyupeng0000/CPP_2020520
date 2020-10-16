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

#ifndef EXAMPLES_MTCNN_EVALUATION_MATCHING_HPP_
#define EXAMPLES_MTCNN_EVALUATION_MATCHING_HPP_

#include <iostream>
#include <string>
#include <vector>

#include "Hungarian.hpp"
#include "MatchPair.hpp"
#include "Region.hpp"
#include "RegionsSingleImage.hpp"
#include "Results.hpp"

using std::vector;

/**
 * Class that computes the matching between annotated and
 * detected regions
 * */
class Matching {
  public:
  /// Name of the algorithm used for matching
  std::string matchingAlgoStr;
  /// Constructor
  Matching(RegionsSingleImage *, RegionsSingleImage *);
  /// Constructor
  Matching(std::string, RegionsSingleImage *, RegionsSingleImage *);
  /// Destructor
  ~Matching();
  /// Compute the matching pairs.
  /// Returns a vector of MatchPair pointers
  vector<MatchPair *> *getMatchPairs();

  private:
  /// Set of annotated regions
  RegionsSingleImage *annot;
  /// Set of detected regions
  RegionsSingleImage *det;
  /// Matrix of matching scores for annnotation and detections
  vector<vector<double> *> *pairWiseScores;

  /// Computes the score for a single pair of regions
  double computeScore(Region *, Region *);
  /// populate the pairWiseScores matrix
  void computePairWiseScores();
  /// Free the memory for the pairWiseScores matrix
  void clearPairWiseScores();
  /// Runs the Hungarian algorithm
  vector<MatchPair *> *runHungarian();
};

#endif  // EXAMPLES_MTCNN_EVALUATION_MATCHING_HPP_
