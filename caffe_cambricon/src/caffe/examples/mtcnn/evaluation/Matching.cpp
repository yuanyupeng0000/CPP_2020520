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

#include "Matching.hpp"

#include <iostream>
#include <limits>
#include <vector>

using std::cerr;
using std::endl;
using std::vector;

Matching::Matching(std::string s, RegionsSingleImage *a,
                   RegionsSingleImage *d) {
  matchingAlgoStr = s;
  Matching(a, d);
}

Matching::Matching(RegionsSingleImage *a, RegionsSingleImage *d) {
  annot = a;
  det = d;
  pairWiseScores = NULL;
}

Matching::~Matching() {
  // free memory for the score matrix
  clearPairWiseScores();
}

double Matching::computeScore(Region *r1, Region *r2) {
  double score = (r1->setIntersect(r2)) / (r1->setUnion(r2));
  return score;
}

void Matching::computePairWiseScores() {
  int nAnnot = annot->length();
  int nDet = det->length();

  if (pairWiseScores == NULL) {
    // first-time access of the score matrix, so initialize it
    pairWiseScores = new vector<vector<double> *>;
    pairWiseScores->reserve(nAnnot);
    for (int ai = 0; ai < nAnnot; ai++) pairWiseScores->push_back(NULL);
  }

  for (int ai = 0; ai < nAnnot; ai++) {
    const IplImage *im = annot->getImage();
    assert(im != NULL);
    IplImage *maskA = cvCreateImage(
      cvGetSize(const_cast<IplImage *>(reinterpret_cast<const IplImage *> (im))),
      IPL_DEPTH_8U, 1);
    cvSetZero(maskA);

    // Create the mask image corresponding to the annotated region
    Region *ar = annot->get(ai);
    maskA = ar->display(maskA, cvScalar(REGION_MASK_VALUE), CV_FILLED, NULL);
    ar->mask = maskA;

    vector<double> *v;
    if (pairWiseScores->at(ai) != NULL)
      v = pairWiseScores->at(ai);
    else
      v = new vector<double>(nDet, -1);

    for (int di = 0; di < nDet; di++) {
      Region *dr = det->get(di);
      if (dr->isValid() && v->at(di) == -1) {
        // Create the mask image corresponding to the detected region
        IplImage *maskD = cvCreateImage(cvGetSize(const_cast<IplImage *>(
                            reinterpret_cast<const IplImage *> (im))), IPL_DEPTH_8U, 1);
        cvSetZero(maskD);
        maskD =
            dr->display(maskD, cvScalar(REGION_MASK_VALUE), CV_FILLED, NULL);
        dr->mask = maskD;
        v->at(di) = computeScore(ar, dr);
        cvReleaseImage(&maskD);
        dr->mask = NULL;
      }
    }
    pairWiseScores->at(ai) = v;

    cvReleaseImage(&maskA);
    ar->mask = NULL;
  }
}

void Matching::clearPairWiseScores() {
  if (pairWiseScores == NULL) return;

  for (unsigned int ai = 0; ai < pairWiseScores->size(); ai++)
    delete (pairWiseScores->at(ai));
  delete (pairWiseScores);
}

vector<MatchPair *> *Matching::runHungarian() {
  int nAnnot = annot->length();
  int nDet = det->length();

  // Remove the annotations with no matching detections
  int *mapI = reinterpret_cast<int *> (calloc(nAnnot, sizeof(int)));
  int mI = 0;
  for (unsigned int i = 0; i < nAnnot; i++) {
    double sum = 0;
    for (unsigned int j = 0; j < nDet; j++)
      if (det->get(j)->isValid()) sum += pairWiseScores->at(i)->at(j);
    if (sum != 0) mapI[mI++] = i;
  }

  // Remove the detections with no matching annotations
  int *mapJ = reinterpret_cast<int *> (calloc(nDet, sizeof(int)));
  int mJ = 0;
  for (unsigned int j = 0; j < nDet; j++) {
    if (det->get(j)->isValid()) {
      double sum = 0;
      for (unsigned int i = 0; i < nAnnot; i++)
        sum += pairWiseScores->at(i)->at(j);
      if (sum != 0) mapJ[mJ++] = j;
    }
  }

  int nRow, nCol;
  bool useTranspose = (mI > mJ);
  if (useTranspose) {
    nCol = mI;
    nRow = mJ;
  } else {
    nRow = mI;
    nCol = mJ;
  }

  // Initialize the match matrix used in the hungarian algorithm
  double **matchMat = reinterpret_cast<double **> (calloc(nRow, sizeof(double *)));
  for (unsigned int i = 0; i < nRow; i++) {
    matchMat[i] = reinterpret_cast<double *> (calloc(nCol, sizeof(double)));
  }

  if (useTranspose) {
    for (unsigned int i = 0; i < nRow; i++)
      for (unsigned int j = 0; j < nCol; j++)
        matchMat[i][j] = pairWiseScores->at(mapI[j])->at(mapJ[i]);
  } else {
    for (unsigned int i = 0; i < nRow; i++)
      for (unsigned int j = 0; j < nCol; j++)
        matchMat[i][j] = pairWiseScores->at(mapI[i])->at(mapJ[j]);
  }

  hungarian_t prob;

  hungarian_init(&prob, matchMat, nRow, nCol, HUNGARIAN_MAX);
  // hungarian_init(&prob,matchMat,nAnnot,nDet,HUNGARIAN_MAX);
  // hungarian_print_rating(&prob);
  hungarian_solve(&prob);
  // hungarian_print_assignment(&prob);

  // Create the set of match pairs from the assignment matrix from the Hungarian
  // algo
  vector<MatchPair *> *mps = NULL;
  if (hungarian_check_feasibility(&prob)) {
    mps = new vector<MatchPair *>;
    if (useTranspose) {
      for (unsigned int i = 0; i < nRow; i++)
        for (unsigned int j = 0; j < nCol; j++) {
          double score = pairWiseScores->at(mapI[j])->at(mapJ[i]);
          if (prob.a[i] == j && score > 0) {
            Region *dr = det->get(mapJ[i]);
            Region *ar = annot->get(mapI[j]);
            mps->push_back(new MatchPair(ar, dr, score));
            // ar->setValid(false);
            // dr->setValid(false);
          }
        }
    } else {
      for (unsigned int i = 0; i < nRow; i++) {
        vector<double> *pwsI = pairWiseScores->at(mapI[i]);
        for (unsigned int j = 0; j < nCol; j++) {
          double score = pwsI->at(mapJ[j]);
          if (prob.a[i] == j && score > 0) {
            Region *dr = det->get(mapJ[j]);
            Region *ar = annot->get(mapI[i]);
            mps->push_back(new MatchPair(ar, dr, score));
            // ar->setValid(false);
            // dr->setValid(false);
          }
        }
      }
    }
  } else {
    cerr << "Not yet implemented" << endl;
    assert(false);
  }

  // free memory used by hungarian algorithm
  hungarian_fini(&prob);

  free(mapI);
  free(mapJ);
  for (unsigned int i = 0; i < nRow; i++) free(matchMat[i]);
  free(matchMat);

  return mps;
}

vector<MatchPair *> *Matching::getMatchPairs() {
  computePairWiseScores();
  return runHungarian();
}
