// Copyright [2018] <cambricon>
#ifdef USE_MLU
#include <iostream>
#include <string>
#include <vector>
#include "offdemo.hpp"

DEFINE_int32(mludevice, 0,
    "set using mlu device number, default: 0");
int main(int argc, char* argv[]) {
::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if ( argc < 4 ) {
    LOG(INFO) << "USAGE: " << argv[0] <<" <cambricon_file>"
      << " <output_file> <function_name0> <function_name1> ...";
    exit(-1);
  }
  // program_name model_name output_file subnet ....
  OffEngine * engine = new OffEngine( string(argv[1]) );

  engine->OpenDevice(FLAGS_mludevice);
  engine->LoadModel();

  for ( int n = 3 ; n < argc ; n++ ) {
    // 3rd. create func
    engine->set_subnet_name(string(argv[n]));
    engine->CreateFunc();

    engine->AllocMemory();

    engine->FillInput();

    if (0 == engine->Run())
      engine->CopyOut(string(argv[2]));

    engine->FreeResources();
  }

  delete engine;

  return 0;
}

void OffEngine::FillInput() {
  int inputnum = inum();
  CHECK_GE(inputnum, 1);

  float** bufs = inputbufs();
  CHECK(bufs);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<> dis(0.1, 1);

  for (int i = 0; i < inputnum; i++) {
    float* databuf = bufs[i];
    CHECK(databuf);

    // n,c,h,w
    const vector<int> shape = in_shape(i);
    int total = shape[0] * shape[1] * shape[2] * shape[3];
    for (int j = 0; j < total; j++) {
      databuf[j] = static_cast<float>(dis(gen));
    }
  }
}
#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif
