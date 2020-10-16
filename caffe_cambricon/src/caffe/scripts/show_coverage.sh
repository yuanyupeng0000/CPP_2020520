#!/bin/sh
# show coverage
# 1. export export CODE_COVERAGE_TEST=ON
# 2. run the test program.

set -e
lcov -c -d . -o clog
lcov -r clog  /usr/\* /cnml/\* -o caffe.coverage
genhtml caffe.coverage -o cloghtml
chromium-browser cloghtml/index.html
