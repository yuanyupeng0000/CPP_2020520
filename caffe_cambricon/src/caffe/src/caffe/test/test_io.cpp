/*
All modification made by Cambricon Corporation: © 2018-2019 Cambricon Corporation
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

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class IOTest : public ::testing::Test {};

bool ReadImageToDatumReference(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

TEST_F(IOTest, TestReadImageToDatum) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 576);
  EXPECT_EQ(datum.width(), 768);
}

TEST_F(IOTest, TestReadImageToDatumReference) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum, datum_ref;
  ReadImageToDatum(filename, 0, 0, 0, true, &datum);
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum.data();

  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}


TEST_F(IOTest, TestReadImageToDatumReferenceResized) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum, datum_ref;
  ReadImageToDatum(filename, 0, 100, 200, true, &datum);
  ReadImageToDatumReference(filename, 0, 100, 200, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum.data();

  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestReadImageToDatumContent) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(data[index++] ==
          static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumContentGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int h = 0; h < datum.height(); ++h) {
    for (int w = 0; w < datum.width(); ++w) {
      EXPECT_TRUE(data[index++] == static_cast<char>(cv_img.at<uchar>(h, w)));
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumResized) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 100, 200, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 100);
  EXPECT_EQ(datum.width(), 200);
}


TEST_F(IOTest, TestReadImageToDatumResizedSquare) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 256, 256, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToDatumGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 576);
  EXPECT_EQ(datum.width(), 768);
}

TEST_F(IOTest, TestReadImageToDatumResizedGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, 256, 256, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToCVMat) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 576);
  EXPECT_EQ(cv_img.cols, 768);
}

TEST_F(IOTest, TestReadImageToCVMatResized) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 100, 200);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 100);
  EXPECT_EQ(cv_img.cols, 200);
}

TEST_F(IOTest, TestReadImageToCVMatResizedSquare) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestReadImageToCVMatGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 576);
  EXPECT_EQ(cv_img.cols, 768);
}

TEST_F(IOTest, TestReadImageToCVMatResizedGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestCVMatToDatum) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 576);
  EXPECT_EQ(datum.width(), 768);
}

TEST_F(IOTest, TestCVMatToDatumContent) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  Datum datum_ref;
  ReadImageToDatum(filename, 0, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestCVMatToDatumReference) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestReadFileToDatum) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(datum.encoded());
  EXPECT_EQ(datum.label(), -1);
  EXPECT_EQ(datum.data().size(), 163759);
}

TEST_F(IOTest, TestDecodeDatum) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatum(&datum, true));
  EXPECT_FALSE(DecodeDatum(&datum, true));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMat) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 576);
  EXPECT_EQ(cv_img.cols, 768);
  cv_img = DecodeDatumToCVMat(datum, false);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 576);
  EXPECT_EQ(cv_img.cols, 768);
}

TEST_F(IOTest, TestDecodeDatumToCVMatContent) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadImageToDatum(filename, 0, std::string("jpg"), &datum));
  cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
  cv::Mat cv_img_ref = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img_ref.channels(), cv_img.channels());
  EXPECT_EQ(cv_img_ref.rows, cv_img.rows);
  EXPECT_EQ(cv_img_ref.cols, cv_img.cols);

  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(cv_img.at<cv::Vec3b>(h, w)[c]==
          cv_img_ref.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}

TEST_F(IOTest, TestDecodeDatumNative) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatumNative(&datum));
  EXPECT_FALSE(DecodeDatumNative(&datum));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMatNative) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 576);
  EXPECT_EQ(cv_img.cols, 768);
}

TEST_F(IOTest, TestDecodeDatumNativeGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatumNative(&datum));
  EXPECT_FALSE(DecodeDatumNative(&datum));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMatNativeGray) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 576);
  EXPECT_EQ(cv_img.cols, 768);
}

TEST_F(IOTest, TestDecodeDatumToCVMatContentNative) {
  string filename = string(TEST_SOURCE_DIR())+"dog.jpg";
  Datum datum;
  EXPECT_TRUE(ReadImageToDatum(filename, 0, std::string("jpg"), &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  cv::Mat cv_img_ref = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img_ref.channels(), cv_img.channels());
  EXPECT_EQ(cv_img_ref.rows, cv_img.rows);
  EXPECT_EQ(cv_img_ref.cols, cv_img.cols);

  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(cv_img.at<cv::Vec3b>(h, w)[c]==
          cv_img_ref.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
