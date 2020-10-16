#ifndef TEST_OFFDEMO_HPP_
#define TEST_OFFDEMO_HPP_

#include <random>
#include <string>
#include "offengine.hpp"

class OffEngine : public caffetool::OfflineEngine {
  public:
  explicit OffEngine(const string& name) : OfflineEngine(name) {}
  ~OffEngine() {}

  void FillInput();
};

#endif  // TEST_OFFDEMO_HPP_
