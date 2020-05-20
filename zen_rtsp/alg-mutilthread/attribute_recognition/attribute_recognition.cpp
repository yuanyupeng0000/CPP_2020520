#include "attribute_recognition.h"
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

typedef struct NetParams{
	shared_ptr< Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;

};
NetParams AttriNet[NET_NUM];
int Net_gpu_index[NET_NUM];
void LoadAttriNet(const char* deploy_file,
				  const char* trained_file,
				  int gpu_idx, int net_idx)
{
  Caffe::SetDevice(gpu_idx);
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  Net_gpu_index[net_idx] = gpu_idx;
  /* Load the network. */
  AttriNet[net_idx].net_.reset(new Net<float>(deploy_file, TEST));
  AttriNet[net_idx].net_->CopyTrainedLayersFrom(trained_file);
  
  CHECK_EQ(AttriNet[net_idx].net_->num_inputs(), 1) << "Network should have exactly one input.";
 // CHECK_EQ(AttriNet[net_idx].net_->num_outputs(), 1) << "Network should have exactly one output.";
  //printf("output num = %d\n",AttriNet[net_idx].net_->num_outputs());
  Blob<float>* input_layer = AttriNet[net_idx].net_->input_blobs()[0];
  AttriNet[net_idx].num_channels_ = input_layer->channels();
  CHECK(AttriNet[net_idx].num_channels_ == 3 || AttriNet[net_idx].num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  AttriNet[net_idx].input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  input_layer->Reshape(1, AttriNet[net_idx].num_channels_,
	  AttriNet[net_idx].input_geometry_.height, AttriNet[net_idx].input_geometry_.width);
  /* Forward dimension change to all layers. */
  AttriNet[net_idx].net_->Reshape();//³õÊ¼»¯ÍøÂç¸÷²ã
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

void Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels, int net_idx) {
  /* Convert the input image to the input image format of the network. */
  	Blob<float>* input_layer = AttriNet[net_idx].net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
  cv::Mat sample_resized;
  if (img.size() != AttriNet[net_idx].input_geometry_)
    cv::resize(img, sample_resized, AttriNet[net_idx].input_geometry_);
  else
    sample_resized = img;
  cv::Mat sample_float;
  if (AttriNet[net_idx].num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);
 // cv::Mat sample_normalized;
 // cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == AttriNet[net_idx].net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
void AttriDetect(unsigned char* imgdata, int w, int h, int net_idx, int* result) {
	Caffe::SetDevice(Net_gpu_index[net_idx]);
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
  cv::Mat img(h, w, CV_8UC3, imgdata);
  //Blob<float>* input_layer = AttriNet[net_idx].net_->input_blobs()[0];
  //input_layer->Reshape(1, AttriNet[net_idx].num_channels_,
	//  AttriNet[net_idx].input_geometry_.height, AttriNet[net_idx].input_geometry_.width);
  /* Forward dimension change to all layers. */
  //AttriNet[net_idx].net_->Reshape();//³õÊ¼»¯ÍøÂç¸÷²ã
  std::vector<cv::Mat> input_channels;
  Preprocess(img, &input_channels, 0);//Ô¤´¦ÀíÍ¼ÏñÊý¾Ý
  const std::vector<caffe::Blob<float>*>& results = AttriNet[net_idx].net_->Forward();
  /* Copy the output layer to a std::vector */
  shared_ptr< caffe::Blob<float> > Male = AttriNet[net_idx].net_->blob_by_name("accuracy_Male");//ÐÔ±ð
  shared_ptr< caffe::Blob<float> > age1 = AttriNet[net_idx].net_->blob_by_name("accuracy_age1");//0-16
  shared_ptr< caffe::Blob<float> > age2 = AttriNet[net_idx].net_->blob_by_name("accuracy_age2");//17-30
  shared_ptr< caffe::Blob<float> > age3 = AttriNet[net_idx].net_->blob_by_name("accuracy_age3");//30-45
  shared_ptr< caffe::Blob<float> > up1 = AttriNet[net_idx].net_->blob_by_name("accuracy_up1");//ºÚ
  shared_ptr< caffe::Blob<float> > up2 = AttriNet[net_idx].net_->blob_by_name("accuracy_up2");//°×
  shared_ptr< caffe::Blob<float> > up3 = AttriNet[net_idx].net_->blob_by_name("accuracy_up3");//»Ò
  shared_ptr< caffe::Blob<float> > up4 = AttriNet[net_idx].net_->blob_by_name("accuracy_up4");//ºì
  shared_ptr< caffe::Blob<float> > up5 = AttriNet[net_idx].net_->blob_by_name("accuracy_up5");//ÂÌ
  shared_ptr< caffe::Blob<float> > up6 = AttriNet[net_idx].net_->blob_by_name("accuracy_up6");//À¶
  shared_ptr< caffe::Blob<float> > up7 = AttriNet[net_idx].net_->blob_by_name("accuracy_up7");//»Æ
  shared_ptr< caffe::Blob<float> > up8 = AttriNet[net_idx].net_->blob_by_name("accuracy_up8");//×Ø
  shared_ptr< caffe::Blob<float> > up9 = AttriNet[net_idx].net_->blob_by_name("accuracy_up9");//×Ï
  shared_ptr< caffe::Blob<float> > up10 = AttriNet[net_idx].net_->blob_by_name("accuracy_up10");//·Û
  shared_ptr< caffe::Blob<float> > up11 = AttriNet[net_idx].net_->blob_by_name("accuracy_up11");//³È
  shared_ptr< caffe::Blob<float> > up12 = AttriNet[net_idx].net_->blob_by_name("accuracy_up12");//»ìÉ«
  shared_ptr< caffe::Blob<float> > low1 = AttriNet[net_idx].net_->blob_by_name("accuracy_low1");//ºÚ
  shared_ptr< caffe::Blob<float> > low2 = AttriNet[net_idx].net_->blob_by_name("accuracy_low2");//°×
  shared_ptr< caffe::Blob<float> > low3 = AttriNet[net_idx].net_->blob_by_name("accuracy_low3");//»Ò
  shared_ptr< caffe::Blob<float> > low4 = AttriNet[net_idx].net_->blob_by_name("accuracy_low4");//ºì
  shared_ptr< caffe::Blob<float> > low5 = AttriNet[net_idx].net_->blob_by_name("accuracy_low5");//ÂÌ
  shared_ptr< caffe::Blob<float> > low6 = AttriNet[net_idx].net_->blob_by_name("accuracy_low6");//À¶
  shared_ptr< caffe::Blob<float> > low7 = AttriNet[net_idx].net_->blob_by_name("accuracy_low7");//»Æ
  shared_ptr< caffe::Blob<float> > low8 = AttriNet[net_idx].net_->blob_by_name("accuracy_low8");//»ìÉ«
  shared_ptr< caffe::Blob<float> > Hat = AttriNet[net_idx].net_->blob_by_name("accuracy_Hat");//Ã±×Ó
  shared_ptr< caffe::Blob<float> > Eyeglasses = AttriNet[net_idx].net_->blob_by_name("accuracy_Eyeglasses");//ÑÛ¾µ
  int age= 0, sex = 0, uppercolor = 0, lowercolor = 0, shape = 0, head = 0, glasses = 0, upstyle = 0, lowerstyle = 0, face = 0;
  /*const float* age1_data = results[0]->cpu_data();
  const float* age2_data = results[1]->cpu_data();
  const float* age3_data = results[2]->cpu_data();;
  const float* Male_data = results[3]->cpu_data();;
  const float* up1_data = results[4]->cpu_data();;
  const float* up2_data = results[5]->cpu_data();;
  const float* up3_data = results[6]->cpu_data();;
  const float* up4_data = results[7]->cpu_data();;
  const float* up5_data = results[8]->cpu_data();;
  const float* up6_data = results[9]->cpu_data();;
  const float* up7_data = results[10]->cpu_data();;
  const float* up8_data = results[11]->cpu_data();;*/
 /* if(*(age1_data) > *(age1_data + 1))
	  age = 0;
  else if(*(age2_data) > *(age2_data + 1))
	  age = 1;
  else if(*(age3_data) > *(age3_data + 1))
	  age =2;
  else
	  age = 3;*/

  if(*(age1->mutable_cpu_data()) > *(age1->mutable_cpu_data() + 1))
	  age = 0;//0-16
  else if(*(age2->mutable_cpu_data()) > *(age2->mutable_cpu_data() + 1))
	  age = 1;//17-30
  else if(*(age3->mutable_cpu_data()) > *(age3->mutable_cpu_data() + 1))
	  age = 2;//30-45
  else 
	  age = 3;//ÆäËû
  sex = (*(Male->mutable_cpu_data()) > *(Male->mutable_cpu_data() + 1))? 0 : 1;
  if(*(up1->cpu_data()) > *(up1->cpu_data() + 1))
	  uppercolor = 0;//ºÚ
  else if(*(up2->mutable_cpu_data()) > *(up2->mutable_cpu_data() + 1))
	  uppercolor = 1;//°×
  else if(*(up3->mutable_cpu_data()) > *(up3->mutable_cpu_data() + 1))
	  uppercolor = 2;//»Ò
  else if(*(up4->mutable_cpu_data()) > *(up4->mutable_cpu_data() + 1))
	  uppercolor = 3;//ºì
  else if(*(up5->mutable_cpu_data()) > *(up5->mutable_cpu_data() + 1))
	  uppercolor = 4;//ÂÌ
  else if(*(up6->mutable_cpu_data()) > *(up6->mutable_cpu_data() + 1))
	  uppercolor = 5;//À¶
  else if(*(up7->mutable_cpu_data()) > *(up7->mutable_cpu_data() + 1))
	  uppercolor = 6;//»Æ
  else if(*(up8->mutable_cpu_data()) > *(up8->mutable_cpu_data() + 1))
	  uppercolor = 7;//×Ø
  else if(*(up9->mutable_cpu_data()) > *(up9->mutable_cpu_data() + 1))
	  uppercolor = 8;//×Ï
  else if(*(up10->mutable_cpu_data()) > *(up10->mutable_cpu_data() + 1))
	  uppercolor = 9;//·Û
  else if(*(up11->mutable_cpu_data()) > *(up11->mutable_cpu_data() + 1))
	  uppercolor = 10;//³È
  else if(*(up12->mutable_cpu_data()) > *(up12->mutable_cpu_data() + 1))
	  uppercolor = 11;//»ìºÏ
  else 
	  uppercolor = 12;//ÆäËû
  if(*(low1->mutable_cpu_data()) > *(low1->mutable_cpu_data() + 1))
	  lowercolor = 0;//ºÚ
  else if(*(low2->mutable_cpu_data()) > *(low2->mutable_cpu_data() + 1))
	  lowercolor = 1;//°×
  else if(*(low3->mutable_cpu_data()) > *(low3->mutable_cpu_data() + 1))
	  lowercolor = 2;//»Ò
  else if(*(low4->mutable_cpu_data()) > *(low4->mutable_cpu_data() + 1))
	  lowercolor = 3;//ºì
  else if(*(low5->mutable_cpu_data()) > *(low5->mutable_cpu_data() + 1))
	  lowercolor = 4;//ÂÌ
  else if(*(low6->mutable_cpu_data()) > *(low6->mutable_cpu_data() + 1))
	  lowercolor = 5;//À¶
  else if(*(low7->mutable_cpu_data()) > *(low7->mutable_cpu_data() + 1))
	  lowercolor = 6;//»Æ
  else if(*(low8->mutable_cpu_data()) > *(low8->mutable_cpu_data() + 1))
	  lowercolor = 7;//»ìÉ«
  else
	  lowercolor = 8;//ÆäËû
  head = (*(Hat->mutable_cpu_data()) > *(Hat->mutable_cpu_data() + 1))? 1 : 0;//ÊÇ·ñ´øÃ±
  glasses = (*(Eyeglasses->mutable_cpu_data()) > *(Eyeglasses->mutable_cpu_data() + 1))? 1 : 0;//ÊÇ·ñ´øÑÛ¾µ
  result[0] = age;
  result[1] = sex;
  result[2] = uppercolor;
  result[3] = lowercolor;
  result[4] = shape;
  result[5] = head;
  result[6] = glasses;
  result[7] = upstyle;
  result[8] = lowerstyle;
  result[9] = face;
  printf("[%d   %d   %d   %d   %d   %d    %d    %d    %d    %d]\n",age,sex,uppercolor,lowercolor,shape,head,glasses,upstyle,lowerstyle,face);

  /*int i = 0, j = 0;
  for(i = 0; i < AttriNet[net_idx].net_->num_outputs(); i++)
  {
	  printf("\n");
	  Blob<float>* output_layer = AttriNet[net_idx].net_->output_blobs()[i];
	  for( j = 0; j < output_layer->channels(); j++)
	  {
		  printf("%f ",*(output_layer->cpu_data() + j));
	  }


  }
  Blob<float>* output_layer = AttriNet[net_idx].net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);*/
}



int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  const char* deploy_file  = argv[1];
  const char* trained_file = argv[2];
  LoadAttriNet(deploy_file, trained_file, 1, 0);

  string file = argv[3];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;
  cv::VideoCapture cap(file);
  while(1) 
  { 
      cv::Mat img; 
      cap>>img;
	  //cv::Mat img = cv::imread(file, -1);
	  CHECK(!img.empty()) << "Unable to decode image " << file;
	 int result[10];
	 AttriDetect(img.data, img.cols, img.rows, 0, result);
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
