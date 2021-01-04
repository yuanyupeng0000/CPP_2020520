/*
All modification made by Cambricon Corporation: © 2018--2020 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2020, the respective contributors
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

#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;

class QuantizeConfig;
class PreProcessor;
class Quantizer;

DEFINE_string(ini_file, "", "The ini_file used to show quantized information");
DEFINE_string(custom_config_file, "",
    "The custom config file used to show quantized information");
DEFINE_string(model, "", "The model definition protocol buffer text file.");
DEFINE_string(outputmodel, "", "The output file name of protocol buffer text file.");
DEFINE_string(
    weights,
    "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(
    mode,
    "common",
    "Optional; determine which mode to generate quantized pt, "
    "common: position + scale(default); scale: only scale;"
    "int8_channel: channel quantize + scales");
DEFINE_int32(iterations, 1,
    "Optional; determine to read images iterations ");
DEFINE_string(blobs_dtype, "INT8", "Set the quantized data type."
    "The optional data types include INT8 and INT16");
DEFINE_string(top_dtype, "FLOAT16", "Set the output data type."
    "The optional data types include FLOAT16 and FLOAT32");

class QuantizeConfig{
  public:
    QuantizeConfig(): use_ini_(false), use_custom_config_(false),
                  use_command_config_(false), custom_preprocess_option_(false),
                  use_firstconv_(false),
                  input_format_(ConvolutionParameter_InputFormat_UNSET),
                  filter_format_(ConvolutionParameter_FilterFormat_BGR),
                  iterations_(1), mode_("common"),
                  blobs_dtype_("INT8"), top_dtype_("FLOAT16") {
      getConfigs();
    }

    ~QuantizeConfig() {}

    bool has_mean() const;
    bool has_std() const;

    inline string config_filename() const { return config_filename_; }
    inline string ori_model_path() const { return ori_model_path_; }
    inline string save_model_path() const { return save_model_path_; }
    inline string ori_weights_path() const { return ori_weights_path_; }
    inline string img_file_list() const { return img_file_list_; }
    inline string img_db_path() const { return img_db_path_; }
    inline vector<float> mean_value() const { return mean_value_; }
    inline vector<float> std() const { return std_; }
    inline vector<float> scale_value() const { return scale_value_; }
    inline vector<float> crop() const { return crop_; }
    inline vector<string> quantize_op_list() const { return quantize_op_list_; }
    inline vector<string> int8_layer_list() const { return int8_layer_list_; }
    inline vector<string> int16_layer_list() const { return int16_layer_list_; }
    inline bool use_ini() const { return use_ini_; }
    inline bool use_custom_config() const { return use_custom_config_; }
    inline bool use_command_config() const { return use_command_config_; }
    inline bool custom_preprocess_option() const { return  custom_preprocess_option_; }
    inline bool use_firstconv() const { return use_firstconv_; }
    inline ConvolutionParameter_InputFormat input_format() const {
      return input_format_;
    }
    inline ConvolutionParameter_FilterFormat filter_format() const {
      return filter_format_;
    }
    inline int iterations() const { return iterations_; }
    inline string mode() const { return mode_; }
    inline string blobs_dtype() { return blobs_dtype_; }
    inline string top_dtype() { return top_dtype_; }


  private:
    int getConfigs();
    vector<float> parseData(string data);
    vector<string> parseString(string data);
    ConvolutionParameter_InputFormat convertInputFormat(
        string input_format) const;
    ConvolutionParameter_FilterFormat convertFilterFormat(
        string filter_format) const;


    int getConfigsFromIniFile(string filename);
    int getConfigsFromCommand();
    int getConfigsFromCustomFile(string filename) {
      LOG(FATAL) << "Please implement user-defined configuration file prasing.";
      return 0;
    }

  private:
    string config_filename_;
    string ori_model_path_;
    string save_model_path_;
    string ori_weights_path_;
    string img_file_list_;
    string img_db_path_;
    vector<float> mean_value_;
    vector<float> std_;
    vector<float> scale_value_;
    vector<float> crop_;
    vector<string> quantize_op_list_;
    vector<string> int8_layer_list_;
    vector<string> int16_layer_list_;
    bool use_ini_;
    bool use_custom_config_;
    bool use_command_config_;
    bool custom_preprocess_option_;
    bool use_firstconv_;
    ConvolutionParameter_InputFormat input_format_;
    ConvolutionParameter_FilterFormat filter_format_;
    int iterations_;
    string mode_;
    string blobs_dtype_;
    string top_dtype_;
};

ConvolutionParameter_InputFormat QuantizeConfig::convertInputFormat(
    string input_format) const {
  ConvolutionParameter_InputFormat format =
    ConvolutionParameter_InputFormat_BGRA;
  if (input_format == "RGBA") {
    format = ConvolutionParameter_InputFormat_RGBA;
  } else if (input_format == "BGRA") {
    format = ConvolutionParameter_InputFormat_BGRA;
  } else if (input_format == "ARGB") {
    format = ConvolutionParameter_InputFormat_ARGB;
  } else if (input_format == "ABGR") {
    format = ConvolutionParameter_InputFormat_ABGR;
  } else {
    LOG(FATAL) << "Unsupported InputFormat: " << input_format;
  }
  return format;
}

ConvolutionParameter_FilterFormat QuantizeConfig::convertFilterFormat(
    string filter_format) const {
  ConvolutionParameter_FilterFormat format =
    ConvolutionParameter_FilterFormat_BGR;
  if (filter_format == "BGR") {
    format = ConvolutionParameter_FilterFormat_BGR;
  } else if (filter_format == "RGB") {
    format = ConvolutionParameter_FilterFormat_RGB;
  } else {
    LOG(FATAL) << "Unsupported FilterFormat: " << filter_format;
  }
  return format;
}

int QuantizeConfig::getConfigs() {
  if (FLAGS_ini_file.size() && FLAGS_custom_config_file.size()) {
    LOG(FATAL) << "Config file is either ini file"
               << " or custom config file, Not both!";
  }
  int status = 0;
  if (FLAGS_ini_file.size()) {
    use_ini_ = true;
    config_filename_ = FLAGS_ini_file;
    LOG(INFO) << "ini_file path : " << FLAGS_ini_file;
    status = getConfigsFromIniFile(config_filename_);
  } else if (FLAGS_custom_config_file.size()) {
    use_custom_config_ = true;
    config_filename_ = FLAGS_custom_config_file;
    LOG(INFO) << "custom_file path : " << FLAGS_custom_config_file;
    status = getConfigsFromCustomFile(config_filename_);
  } else {
    LOG(INFO) << "Use command config for quantization.";
    use_command_config_ = true;
    status = getConfigsFromCommand();
  }

  mode_ = FLAGS_mode;
  blobs_dtype_ = FLAGS_blobs_dtype;
  top_dtype_ = FLAGS_top_dtype;

  return status;
}

int QuantizeConfig::getConfigsFromIniFile(string filename) {
  boost::property_tree::ptree pt;
  boost::property_tree::ini_parser::read_ini(filename, pt);
  if (FLAGS_model.size() > 0) {
    pt.put<string>("model.original_models_path", FLAGS_model);
  }
  if (FLAGS_weights.size() > 0) {
    pt.put<string>("weights.original_weights_path", FLAGS_weights);
  }
  if (FLAGS_outputmodel.size() > 0) {
    pt.put<string>("model.save_model_path", FLAGS_outputmodel);
  }
  ori_model_path_ = pt.get<string>("model.original_models_path");
  save_model_path_ = pt.get<string>("model.save_model_path");
  auto img_file_list_old = pt.get_optional<string>("data.images_folder_path");
  auto img_file_list_new = pt.get_optional<string>("data.images_list_path");
  if (img_file_list_old && img_file_list_new) {
    LOG(FATAL) << "Please use the image_folder_path or image_list_path to specify"
               << " the image file list, Not both. Recommend image_list_path.";
  }
  if (img_file_list_old) {
    img_file_list_ = img_file_list_old.get();
  }
  if (img_file_list_new) {
    img_file_list_ = img_file_list_new.get();
  }
  auto img_db_path = pt.get_optional<string>("data.images_db_path");
  if (img_db_path) {
    img_db_path_ = img_db_path.get();
  }
  ori_weights_path_ = pt.get<string>("weights.original_weights_path");
  auto used_img_num = pt.get_optional<int>("data.used_images_num");
  if (used_img_num) {
    iterations_ = used_img_num.get();
  }
  boost::optional<string> mean = pt.get_optional<string>("preprocess.mean");
  if (!mean) {
    mean_value_ = parseData("0, 0, 0");
  } else {
    mean_value_ = parseData(mean.get());
  }
  boost::optional<string> std = pt.get_optional<string>("preprocess.std");
  if (!std) {
    std_ = parseData("1");
  } else {
    std_ = parseData(std.get());
  }
  boost::optional<string> scale_value = pt.get_optional<string>("preprocess.scale");
  if (!scale_value) {
    scale_value_ = parseData("-1, -1");
  } else {
    scale_value_ = parseData(scale_value.get());
  }
  boost::optional<string> crop_value = pt.get_optional<string>("preprocess.crop");
  if (!crop_value) {
    crop_ = parseData("-1, -1");
  } else {
    crop_ = parseData(crop_value.get());
  }
  boost::optional<string> quantize_op_list =
    pt.get_optional<string>("config.quantize_op_list");
  if (quantize_op_list) {
    quantize_op_list_ = parseString(quantize_op_list.get());
  }
  string use_firstconv = pt.get<string>("config.use_firstconv");
  std::istringstream(use_firstconv) >> use_firstconv_;
  auto int8_layer_list = pt.get_optional<string>("config.int8_layer_list");
  auto int16_layer_list = pt.get_optional<string>("config.int16_layer_list");
  if (int8_layer_list) {
    int8_layer_list_ = parseString(int8_layer_list.get());
  }
  if (int16_layer_list) {
    int16_layer_list_ = parseString(int16_layer_list.get());
  }
  auto custom_preprocess_option = pt.get_optional<string>("custom.use_custom_preprocess");
  if (custom_preprocess_option) {
    std::istringstream(custom_preprocess_option.get()) >> custom_preprocess_option_;
  }
  auto input_format = pt.get_optional<string>("custom.input_format");
  if (input_format) {
    input_format_ = convertInputFormat(input_format.get());
  }
  auto filter_format = pt.get_optional<string>("custom.filter_format");
  if (filter_format) {
    filter_format_ = convertFilterFormat(filter_format.get());
  }
  return 0;
}

vector<float> QuantizeConfig::parseData(string data) {
  stringstream ss(data);
  vector<float> values;
  string item;
  while (getline(ss, item, ',')) {
    values.push_back(stof(item));
  }
  return values;
}

vector<string> QuantizeConfig::parseString(string data) {
  stringstream ss(data);
  vector<string> values;
  string item;
  while (getline(ss, item, ',')) {
    values.push_back(item.erase(0, item.find_first_not_of(" ")));
  }
  return values;
}
int QuantizeConfig::getConfigsFromCommand() {
  ori_model_path_ = FLAGS_model;
  ori_weights_path_ = FLAGS_weights;
  save_model_path_ = FLAGS_outputmodel;
  iterations_ = FLAGS_iterations;
  return 0;
}
bool QuantizeConfig::has_mean() const {
  if (mean_value_.size() > 0) {
    return true;
  }
  return false;
}

bool QuantizeConfig::has_std() const {
  if (std_.size() > 0) {
    return true;
  }
  return false;
}

enum class InputLayerType{ ImageData, Data, Input };

class PreProcessor{
  public:
    explicit PreProcessor(QuantizeConfig* quantize_config) :
                    use_custom_preprocess_(false), data_size_(0), in_num_(1),
                    in_channel_(3), input_layer_top_name_("data"),
                    quantize_config_(quantize_config) {
      ReadProtoFromTextFile(quantize_config_->ori_model_path(), &net_param_original_);
      if (quantize_config->use_command_config()) {
        net_param_processed_ = net_param_original_;
      } else {
        getNetInputSize();
      }
    }

    ~PreProcessor() {
      for (auto& ptr : data_ptrs_ )
      if (ptr != nullptr) {
        delete [] ptr;
        ptr = nullptr;
      }
    }

    int preProcess();
    inline caffe::NetParameter net_param_processed() {
      return net_param_processed_;
    }
    inline caffe::NetParameter net_param_original() {
      return net_param_original_;
    }
    inline bool use_custom_preprocess() const { return use_custom_preprocess_; }
    inline size_t data_size() const { return data_size_; }
    inline vector<float*> data_ptrs() const { return data_ptrs_; }
    inline vector<string> image_list() const { return image_list_;}
    int networkTrimmer();

  private:
    int getNetInputSize();
    int customPreProcess();
    void ReadImageList();
    void yoloPreProcess();
    void wrapInputLayer(vector<cv::Mat>* wrapped_image, float* input_data);

    int createNetInputLayer(LayerParameter* layer_param);
    int createImageDataLayer(LayerParameter* layer_param);
    int createDataLayer(LayerParameter* layer_param);
    int createInputLayer(LayerParameter* layer_param);

  private:
    InputLayerType input_layer_type_;
    bool use_custom_preprocess_;
    size_t data_size_;  // preprocess 1batch data size
    vector<float*> data_ptrs_;  // Input data, Layput: NCHW
    int in_num_;
    int in_channel_;
    int in_height_;
    int in_width_;
    string input_layer_top_name_;
    cv::Size in_geometry_;
    vector<string> image_list_;
    caffe::NetParameter net_param_original_;
    caffe::NetParameter net_param_processed_;
    QuantizeConfig* quantize_config_;
};

int PreProcessor::preProcess() {
  int status = 0;
  switch (input_layer_type_) {
    case InputLayerType::ImageData :
    case InputLayerType::Data:
      LOG(INFO) << "ImageData or Data layer don't require preprocessing!";
      break;
    case InputLayerType::Input:
      status = customPreProcess();
      break;
    default:
      LOG(INFO) << "No input layer type specified, Use origial input layer";
      break;
  }
  return status;
}

int PreProcessor::getNetInputSize() {
  if (quantize_config_->img_db_path().empty()) {
    if (quantize_config_->custom_preprocess_option()) {
      input_layer_type_ = InputLayerType::Input;
    } else {
      input_layer_type_ = InputLayerType::ImageData;
    }
  } else {
    input_layer_type_ = InputLayerType::Data;
  }
  if (input_layer_type_ == InputLayerType::ImageData ||
      input_layer_type_ == InputLayerType::Data) {
    in_height_ = quantize_config_->scale_value()[0];
    in_width_ = quantize_config_->scale_value()[1];
  }
  if (net_param_original_.input_dim_size() > 0) {
    input_layer_top_name_ = net_param_original_.input(0);
    in_num_ = net_param_original_.input_dim(0);
    in_channel_ = net_param_original_.input_dim(1);
    in_height_ = net_param_original_.input_dim(2);
    in_width_ = net_param_original_.input_dim(3);
  } else if (net_param_original_.input_shape_size() > 0) {
    input_layer_top_name_ = net_param_original_.input(0);
    in_num_ = net_param_original_.input_shape(0).dim(0);
    in_channel_ = net_param_original_.input_shape(0).dim(1);
    in_height_ = net_param_original_.input_shape(0).dim(2);
    in_width_ = net_param_original_.input_shape(0).dim(3);
  } else if (net_param_original_.layer(0).type() == "Input") {
    input_layer_top_name_ = net_param_original_.layer(0).top(0);
    auto shape = net_param_original_.layer(0).input_param().shape(0);
    in_num_ = shape.dim(0);
    in_channel_ = shape.dim(1);
    in_height_ = shape.dim(2);
    in_width_ = shape.dim(3);
  } else if (net_param_original_.layer(0).type() == "Data") {
    input_layer_top_name_ = net_param_original_.layer(0).top(0);
    in_height_ = in_width_ =
      net_param_original_.layer(0).transform_param().crop_size();
  } else if (net_param_original_.layer(0).type() == "ImageData") {
    input_layer_top_name_ = net_param_original_.layer(0).top(0);
    in_num_ = net_param_original_.layer(0).image_data_param().batch_size();
    if (!net_param_original_.layer(0).image_data_param().is_color()) {
      in_channel_ = 1;
    }
    in_height_ = net_param_original_.layer(0).image_data_param().new_height();
    in_width_ = net_param_original_.layer(0).image_data_param().new_width();
  } else {
    LOG(INFO) << "Quantize tools unsupported input layer.";
  }
  if (input_layer_type_ == InputLayerType::ImageData ||
      input_layer_type_ == InputLayerType::Data) {
    if (in_height_ != quantize_config_->scale_value()[0] ||
        in_width_ != quantize_config_->scale_value()[1]) {
      in_height_ = quantize_config_->scale_value()[0];
      in_width_ = quantize_config_->scale_value()[1];
    }
  }

  in_geometry_  = cv::Size(in_height_, in_width_);
  in_num_ = 1;  // quantize inference, set batchsize to 1
  return 0;
}

int PreProcessor::createNetInputLayer(LayerParameter* layer_param) {
  int status = 0;
  switch (input_layer_type_) {
    case InputLayerType::ImageData :
      LOG(INFO) << "Use \"ImageData\" as input layer.";
      status = createImageDataLayer(layer_param);
      break;
    case InputLayerType::Data:
      LOG(INFO) << "Use \"Data\" as input layer.";
      status = createDataLayer(layer_param);
      break;
    case InputLayerType::Input:
      LOG(INFO) << "Use \"Input\" as input layer.";
      status = createInputLayer(layer_param);
      break;
    default:
      LOG(FATAL) << "No input layer type specified!";
      break;
  }
  return status;
}

int PreProcessor::createImageDataLayer(LayerParameter* layer_param) {
  layer_param->Clear();
  layer_param->set_name("data");
  layer_param->set_type("ImageData");
  layer_param->add_top(input_layer_top_name_);
  layer_param->add_top("label");
  NetStateRule* net_rule = layer_param->add_include();
  net_rule->set_phase(TEST);
  TransformationParameter* trans_param = new TransformationParameter();
  if (!quantize_config_->use_firstconv()) {
    if (quantize_config_->has_mean()) {
      for (auto mean : quantize_config_->mean_value()) {
        trans_param->add_mean_value(mean);
      }
    }
    if (quantize_config_->has_std()) {
      for (auto std : quantize_config_->std()) {
        trans_param->add_scale(std);
      }
    }
  }
  trans_param->set_mirror(false);
  layer_param->set_allocated_transform_param(trans_param);
  ImageDataParameter* data_param = new ImageDataParameter();
  data_param->set_source(quantize_config_->img_file_list());
  data_param->set_batch_size(in_num_);
  data_param->set_new_height(in_height_);
  data_param->set_new_width(in_width_);
  if (in_channel_ == 1) {
    data_param->set_is_color(false);
  }
  layer_param->set_allocated_image_data_param(data_param);
  return 0;
}

int PreProcessor::createDataLayer(LayerParameter* layer_param) {
  layer_param->Clear();
  layer_param->set_name("data");
  layer_param->set_type("Data");
  layer_param->add_top(input_layer_top_name_);
  layer_param->add_top("label");
  NetStateRule* net_rule = layer_param->add_include();
  net_rule->set_phase(TEST);
  TransformationParameter* trans_param = new TransformationParameter();
  if (quantize_config_->crop()[0] != -1) {
    trans_param->set_crop_size(quantize_config_->crop()[0]);
  }
  if (!quantize_config_->use_firstconv()) {
    if (quantize_config_->has_mean()) {
      for (auto mean : quantize_config_->mean_value()) {
        trans_param->add_mean_value(mean);
      }
    }
    if (quantize_config_->has_std()) {
      for (auto std : quantize_config_->std()) {
        trans_param->add_scale(std);
      }
    }
  }
  trans_param->set_mirror(false);
  layer_param->set_allocated_transform_param(trans_param);
  DataParameter* data_param = new DataParameter();
  data_param->set_source(quantize_config_->img_db_path());
  data_param->set_batch_size(in_num_);
  data_param->set_backend(DataParameter::LMDB);
  layer_param->set_allocated_data_param(data_param);

  return 0;
}

int PreProcessor::createInputLayer(LayerParameter* layer_param) {
  layer_param->Clear();
  layer_param->set_name("data");
  layer_param->set_type("Input");
  layer_param->add_top(input_layer_top_name_);
  InputParameter* input_param = new InputParameter();
  input_param->Clear();
  BlobShape* blobshape = input_param->add_shape();
  blobshape->add_dim(in_num_);
  blobshape->add_dim(in_channel_);
  blobshape->add_dim(in_height_);
  blobshape->add_dim(in_width_);
  layer_param->set_allocated_input_param(input_param);
  return 0;
}

int PreProcessor::networkTrimmer() {
  LayerParameter* layer_param = net_param_processed_.add_layer();
  int status = 0;
  status = createNetInputLayer(layer_param);
  int input_layer_num = 1;
  if (net_param_original_.input_dim_size() > 0 ||
      net_param_original_.input_shape_size() > 0) {
    if (net_param_original_.input_size() == 2) {
      LayerParameter* input_layer_param = net_param_processed_.add_layer();
      input_layer_param->Clear();
      input_layer_param->set_name(net_param_original_.input(1));
      input_layer_param->set_type("Input");
      input_layer_param->add_top(net_param_original_.input(1));
      InputParameter* input_param1 = new InputParameter();
      input_param1->Clear();
      BlobShape* blobshape = input_param1->add_shape();
      blobshape->add_dim(net_param_original_.input_shape(1).dim(0));
      blobshape->add_dim(net_param_original_.input_shape(1).dim(1));
      input_layer_param->set_allocated_input_param(input_param1);
      input_layer_num++;
    }
    for (int i = 0; i < net_param_original_.layer_size(); i++) {
      net_param_processed_.add_layer()->CopyFrom(net_param_original_.layer(i));
    }
  } else {
    for (int i = 1; i < net_param_original_.layer_size(); i++) {
      net_param_processed_.add_layer()->CopyFrom(net_param_original_.layer(i));
    }
  }
  if (quantize_config_->use_firstconv()) {
    auto param = net_param_processed_.mutable_layer(
        input_layer_num)->mutable_convolution_param();
    auto param1 = net_param_processed_.layer(input_layer_num).convolution_param();
    if (param1.has_mean_file()) {
      param->clear_mean_file();
    } else if (param1.mean_value_size() > 0) {
      param->clear_mean_value();
    }
    if (quantize_config_->has_mean()) {
      for (auto mean : quantize_config_->mean_value()) {
        param->add_mean_value(mean);
      }
    }
    if (param1.std_size() > 0) {
      param->clear_std();
    }
    if (quantize_config_->has_std()) {
      for (auto std : quantize_config_->std()) {
        param->add_std(std);
      }
    }
    if (quantize_config_->input_format() == ConvolutionParameter_InputFormat_ARGB ||
        quantize_config_->input_format() == ConvolutionParameter_InputFormat_ABGR ||
        quantize_config_->input_format() == ConvolutionParameter_InputFormat_BGRA ||
        quantize_config_->input_format() == ConvolutionParameter_InputFormat_RGBA) {
      param->set_input_format(quantize_config_->input_format());
      param->set_filter_format(quantize_config_->filter_format());
    }
  }
  net_param_processed_.mutable_state()->set_phase(caffe::TEST);

  return status;
}

int PreProcessor::customPreProcess() {
  LOG(FATAL) << "Please implement custom preprocessing!";
  // yoloPreProcess();
  return 0;
}

void PreProcessor::yoloPreProcess() {
  ReadImageList();
  int iterations = quantize_config_->iterations();
  data_ptrs_.resize(iterations, nullptr);
  data_size_ = in_num_ * in_channel_ * in_height_ * in_width_;
  cv::Mat source_image;
  for (int i = 0; i < iterations; i++) {
    source_image = cv::imread(image_list_[i], -1);
    if (source_image.data) {
      cv::Mat sample;
      if (source_image.channels() == 4 && in_channel_ == 3)
        cv::cvtColor(source_image, sample, cv::COLOR_BGRA2BGR);
      else if (source_image.channels() == 1 && in_channel_ == 3)
        cv::cvtColor(source_image, sample, cv::COLOR_GRAY2BGR);
      else
        sample = source_image;
      cv::Mat sample_resized;
      if (sample.size() != in_geometry_) {
        int input_dim = 416;
        int new_h, new_w;
        float img_w, img_h, img_scale;
        cv::Mat sample_temp;
        cv::Mat sample_temp_416;
        img_w = sample.cols;
        img_h = sample.rows;
        img_scale = img_w < img_h ? (input_dim / img_h) : (input_dim / img_w);
        new_w = std::floor(img_w * img_scale);
        new_h = std::floor(img_h * img_scale);
        cv::resize(sample, sample_temp, cv::Size(new_w, new_h), CV_INTER_LINEAR);
        sample_temp_416 = cv::Mat(in_geometry_.height, in_geometry_.height,
                                  CV_8UC3, cv::Scalar(128, 128, 128));
        sample_temp.copyTo(sample_temp_416(
            cv::Range((static_cast<float>(in_geometry_.height) - new_h) / 2,
              (static_cast<float>(in_geometry_.height) - new_h) / 2 + new_h),
            cv::Range((static_cast<float>(in_geometry_.height) - new_w) / 2,
              (static_cast<float>(in_geometry_.height) - new_w) / 2 + new_w)));
        //  BGR -> RGB
        cv::cvtColor(sample_temp_416, sample_resized, cv::COLOR_BGR2RGB);
      } else {
        sample_resized = sample;
      }
      cv::Mat sample_float;
      sample_resized.convertTo(sample_float, CV_32FC3);
      data_ptrs_[i] = new float[data_size_];
      vector<cv::Mat> preprocessed_image;
      wrapInputLayer(&preprocessed_image, data_ptrs_[i]);
      cv::split(sample_float, preprocessed_image);
    }
  }
}

void PreProcessor::wrapInputLayer(vector<cv::Mat>* wrapped_image, float* input_data) {
  for (int j = 0; j < in_channel_; ++j) {
    cv::Mat channel(in_height_, in_width_, CV_32FC1, input_data);
    wrapped_image->push_back(channel);
    input_data += in_width_ * in_height_;
  }
}

void PreProcessor::ReadImageList() {
  if (!quantize_config_->img_file_list().empty()) {
    std::ifstream file_list(quantize_config_->img_file_list(), std::ios::in);
    CHECK(!file_list.fail()) << "Image file is invalid!";
    string line;
    while (getline(file_list, line)) {
      if (line.find(" ") != string::npos) {
        line = line.substr(0, line.find(" "));
      }
      image_list_.push_back(line);
    }
    file_list.close();
    LOG(INFO) << "There are a total of "
              << image_list_.size() << " in file_list.";
  }
}

class Quantizer{
  public:
      Quantizer(): net_(nullptr), quantize_config_(nullptr),
                pre_processor_(nullptr) {
        quantize_config_ = new QuantizeConfig();
        pre_processor_ = new PreProcessor(quantize_config_);
        if (!quantize_config_->use_command_config()) {
          pre_processor_->networkTrimmer();
        }
        Caffe::Caffe::set_mode(caffe::Caffe::CPU);
        net_ = new caffe::Net<float>(pre_processor_->net_param_processed());
        net_->CopyTrainedLayersFrom(quantize_config_->ori_weights_path());
      }

    ~Quantizer() {
      if (net_ != nullptr) {
        delete net_;
        net_ = nullptr;
      }
      if (quantize_config_ != nullptr) {
        delete quantize_config_;
        quantize_config_ = nullptr;
      }
      if (pre_processor_ != nullptr) {
        delete pre_processor_;
        pre_processor_ = nullptr;
      }
    }

    int absMaxGenerator();

    inline caffe::Net<float>* net() const { return net_; }
    inline PreProcessor* pre_processor() const { return pre_processor_; }

  private:
    caffe::Net<float>* net_;
    QuantizeConfig* quantize_config_;
    PreProcessor* pre_processor_;
};

int Quantizer::absMaxGenerator() {
  BaseDataType blobs_dtype;
  if (boost::iequals(quantize_config_->blobs_dtype(), "INT8")) {
    blobs_dtype = DT_INT8;
  } else if (boost::iequals(quantize_config_->blobs_dtype(), "INT16")) {
    blobs_dtype = DT_INT16;
  } else {
    LOG(FATAL) << "blobs_dtype: The specified data type is not supported.";
  }
  BaseDataType top_dtype;
  if (boost::iequals(quantize_config_->top_dtype(), "FLOAT16")) {
    top_dtype = DT_FLOAT16;
  } else if (boost::iequals(quantize_config_->top_dtype(), "FLOAT32")) {
    top_dtype = DT_FLOAT32;
  } else {
    LOG(FATAL) << "top_dtype: The specified data type is not supported.";
  }
  std::map<string, float> max_value;
  string save_model_path = quantize_config_->save_model_path();
  string mode = quantize_config_->mode();
  vector<string> int8_layers = quantize_config_->int8_layer_list();
  vector<string> int16_layers = quantize_config_->int16_layer_list();
  bool use_ini = quantize_config_->use_ini();
  int iterations = quantize_config_->iterations();
  ConvolutionParameter_InputFormat input_format = quantize_config_->input_format();
  ConvolutionParameter_FilterFormat filter_format = quantize_config_->filter_format();
  for (int i = 0; i < iterations; i++) {
    if (pre_processor_->net_param_processed().layer(0).type() == "Input") {
      auto input_blob = net_->input_blobs()[0];
      input_blob->set_cpu_data(pre_processor_->data_ptrs()[i]);
      net_->ForwardPrefilled();
    } else {
      net_->Forward();
    }
    if (i == iterations - 1) {
      net_->ToquantizedPrototxt(&max_value, save_model_path,
          mode, blobs_dtype, top_dtype, int8_layers, int16_layers,
          input_format, filter_format, use_ini, true);
    } else {
      net_->ToquantizedPrototxt(&max_value, save_model_path,
          mode, blobs_dtype, top_dtype, int8_layers, int16_layers,
          input_format, filter_format, use_ini, false);
    }
  }

  LOG(INFO) << "Output file is " << save_model_path
            << ", iterations: " << iterations;
  return 0;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  gflags::SetUsageMessage(
      "command line brew\n"
      "Usage: generate_quantized_pt -ini_file convert_quantized.ini [optional]\n"
      "  optional: if specified, covers the corresponding value setted in ini_file\n"
      "  -model net.prototxt"
      "  -weights net.caffemodel"
      "  -outputmodel new_net.prototxt");
  if (argc == 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/generate_quantized_pt");
    return 1;
  }
  // Google flags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Google logging.
  google::InitGoogleLogging(argv[0]);
  // Provide a backtrace on segfault.
  google::InstallFailureSignalHandler();

  Quantizer quantizer;
  quantizer.pre_processor()->preProcess();
  quantizer.absMaxGenerator();

  return 0;
}
