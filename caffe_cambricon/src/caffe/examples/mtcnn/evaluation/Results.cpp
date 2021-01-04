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

#include "Results.hpp"

#include <fstream>
#include <iostream>

using std::endl;
using std::ofstream;

#ifndef _WIN32
#include <limits>
#endif

Results::Results() {}

Results::Results(Results *r) {
  if (r) {
    N = r->N;
    TPCont = r->TPCont;
    TPDisc = r->TPDisc;
    FP = r->FP;
    scoreThreshold = r->scoreThreshold;
  }
}

Results::Results(Results *r, int N2) {
  if (r) {
    N = r->N + N2;
    TPCont = r->TPCont;
    TPDisc = r->TPDisc;
    FP = r->FP;
    scoreThreshold = r->scoreThreshold;
  }
}

Results::Results(Results *r1, Results *r2) {
  N = 0;
  TPDisc = 0;
  TPCont = 0;
  FP = 0;
#ifdef _WIN32
  scoreThreshold = 1e20;
#else
  scoreThreshold = std::numeric_limits<double>::max();
#endif

  if (r1) {
    N += r1->N;
    TPCont += r1->TPCont;
    TPDisc += r1->TPDisc;
    FP += r1->FP;
    if (r1->scoreThreshold < scoreThreshold)
      scoreThreshold = r1->scoreThreshold;
  }

  if (r2) {
    N += r2->N;
    TPCont += r2->TPCont;
    TPDisc += r2->TPDisc;
    FP += r2->FP;
    if (r2->scoreThreshold < scoreThreshold)
      scoreThreshold = r2->scoreThreshold;
  }
}

Results::Results(string s, double scoreThresh, vector<MatchPair *> *parallel,
                 RegionsSingleImage *annot, RegionsSingleImage *det) {
  imName = s;
  scoreThreshold = scoreThresh;

  N = annot->length();

  FP = 0;
  for (unsigned int i = 0; i < det->length(); i++)
    if ((det->get(i))->isValid()) FP++;

  TPCont = 0;
  TPDisc = 0;
  if (parallel) {
    for (unsigned int i = 0; i < parallel->size(); i++) {
      double score = parallel->at(i)->score;
      TPCont += score;
      if (score > 0.5) {
        TPDisc++;
        FP--;
      }
    }
  }
}

vector<Results *> *Results::merge(vector<Results *> *rv1,
                                  vector<Results *> *rv2) {
  vector<Results *> *mergeV = new vector<Results *>;
  unsigned int n1 = 0;
  if (rv1) n1 = rv1->size();
  unsigned int n2 = 0;
  if (rv2) n2 = rv2->size();

  unsigned int nAnnot1 = 0, nAnnot2 = 0;

  if (n1) {
    nAnnot1 = rv1->at(0)->getN();
    if (n2) nAnnot2 = rv2->at(0)->getN();

    unsigned int i1 = 0, i2 = 0;
    double score1, score2;

    Results *r1 = NULL;
    Results *r2 = NULL;

    while (i1 < n1) {
      r1 = rv1->at(i1);
      score1 = rv1->at(i1)->scoreThreshold;
      if (i2 < n2) {
        r2 = rv2->at(i2);
        score2 = rv2->at(i2)->scoreThreshold;
        Results *newR = new Results(r1, r2);
        mergeV->push_back(newR);
        if (score1 < score2) {
          i1++;
        } else if (score1 == score2) {
          i1++;
          i2++;
        } else {
          i2++;
        }
      } else {
        while (i1 < n1) {
          // add from rv1
          r1 = rv1->at(i1);

          Results *newR = new Results(r1, nAnnot2);
          mergeV->push_back(newR);
          i1++;
        }
      }
    }

    while (i2 < n2) {
      // add from rv2
      r2 = rv2->at(i2);
      Results *newR = new Results(r2, nAnnot1);
      mergeV->push_back(newR);
      i2++;
    }
  } else {
    if (n2) {
      for (unsigned int i = 0; i < n2; i++)
        mergeV->push_back(new Results(rv2->at(i)));
    }
  }
  return mergeV;
}

void Results::print(std::ostream &os) {
  os << imName << " Threshold = " << scoreThreshold << " N = " << N
     << " TP cont = " << TPCont << " TP disc = " << TPDisc << " FP = " << FP
     << endl;
}

void Results::saveROC(string outFile, vector<Results *> *rv) {
  string s = outFile + "ContROC.txt";
  ofstream osc(s.c_str());

  s = outFile + "DiscROC.txt";
  ofstream osd(s.c_str());

  for (unsigned int i = 0; i < rv->size(); i++) {
    Results *r = rv->at(i);
    if (r->N) {
      osc << (r->TPCont / r->N) << " " << r->FP << endl;
      osd << (r->TPDisc / r->N) << " " << r->FP << " " << r->scoreThreshold
          << endl;
    } else {
      osc << "0 0" << endl;
      osd << "0 0 " << r->scoreThreshold << endl;
    }
  }
  osc.close();
  osd.close();
}

unsigned int Results::getN() { return N; }
