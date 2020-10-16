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

#ifndef INCLUDE_CAFFE_NET_HPP_
#define INCLUDE_CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/compile.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_MLU
#include "caffe/mlu/reshape_helper.hpp"
#include "caffe/mlu/spliter.hpp"
#include "caffe/mlu/subnet.hpp"
#endif  // USE_MLU

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 */
template <typename Dtype>
class Net {
  public:
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase, const int level = 0,
               const vector<string>* stages = NULL);
  explicit Net(void* buffer, int buffer_size, Phase phase);
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);
#ifdef USE_MLU
  /**
   * @brief Initialize MLU related "global" data(create subnet). Called at Init's ending.
   *
   */
  void InitSubnet();

  /**
   * @brief Optimize ssd to improve MLU performance
   */
  void OptimizeSsd(const NetParameter& param,
      NetParameter* const param_optimized);

  /**
   * @brief Optimize ConvBnScale to improve MLU performance
   */
  void OptimizeConvBnScale(NetParameter param,
      NetParameter* const param_optimized);

  /**
   * @brief Generate MLU offline model.
   *
   * @param name indicates the model name. The model file is "name.mef",
   *        and an description file "name.desc" will be generated.
   */
  void genOfflineModel(const std::string& name);


  /**
   * @brief Generate MLU offline model to memory buffer and return result.
   *
   * @param buffer[in] indicates input mem buffer used to save model,
   *        buffer_size[in] indicates input mem buffer size,
   *        model_size[out] return real size of model,
   *        hardware_reshape[in] specified reshape mode 0/1.
   *
   */
  bool genOfflineModelToMem(void* buffer, uint64_t* buffer_size, uint64_t* model_size);

  bool genOfflineModelToMem(void** buffer, uint64_t* model_size);
#endif  // USE_MLU

  /**
   * @brief Run Forward and return the result.
   *
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
    LOG_EVERY_N(WARNING, 1000)
        << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
#pragma GCC diagnostic pop
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
#ifdef USE_MLU
  Dtype ForwardFromTo_default(int start, int end);
  Dtype ForwardFromTo_mfus(int start, int end);
#endif
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>*>& bottom,
                                      Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();
  void blob_info() const;

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  void SetCompileShape(vector<int>& newshape);

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFrom(void* buffer, int buffer_size);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(void* buffer, int buffer_size);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;
  void ToquantizedPrototxt(map<string, Dtype>* max_value,
      string output_file, string mode,
      BaseDataType type, BaseDataType top_dtype,
      vector<string> int8_layers, vector<string> int16_layers,
      ConvolutionParameter_InputFormat input_format,
      ConvolutionParameter_FilterFormat filter_format,
      bool use_ini = false, bool write = false);

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the layer index
  inline map<string, int>& layer_names_index() { return layer_names_index_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype>>>& blobs() const { return blobs_; }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype>>>& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*>>& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*>>& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int>& top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int>& bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool>>& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype>>>& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype>> blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype>> layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  inline const NetParameter net_param_without_weights() {
    return net_param_without_weights_;
  }
  inline void set_net_param_without_weights(const NetParameter& net_param) {
    net_param_without_weights_ = net_param;
  }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the
   * current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
                        NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
                             const string& layer_name);

  // Invoked at specific points during an iteration
  class Callback {
    protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  const vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) { before_forward_.push_back(value); }
  const vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) { after_forward_.push_back(value); }
  const vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) { after_backward_.push_back(value); }

  const set<int>& dump_top_idx() { return dump_top_idx_; }
  void set_dump_top_idx(set<int> idx) { dump_top_idx_ = idx; }

#ifdef USE_MLU
  // @breif offline net execute
  void OfflineNetRun(const SegmentInfo& seg_info,
                     const cnrtModel_t& model,
                     const cnrtDataType_t& dtype,
                     void** cpuData);

  // @breif destroy offline resource
  void OfflineDestroy();

  // @breif control if append cpu info into offlinemodel file
  inline void set_cpu_info_flag(bool flag) { set_cpu_info_ = flag; }
  void RecalculateWeightsInt8Info(shared_ptr<Blob<Dtype>> weights_blob,
    const LayerParameter& param);
#endif

  protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);

  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);

  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);

  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);

  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

#ifdef USE_MLU
  // @brief Append cpu model, weights and segment info into offline model file.
  // File format: offline + caffe flag + cpu model and weights size
  //             + cpu model and weights + segment size + segment
  void AppendCpuInfo(const string& file,
                     const vector<vector<string>>& input_blob_array,
                     const vector<vector<string>>& output_blob_array);

  // @brief gennerated SegmentInfo
  SegmentInfo* GenSegmentInfo(const vector<vector<string>>& input_blob_array,
                              const vector<vector<string>>& output_blob_array);

  // @breif init memory
  void OfflineRunInit(const SegmentInfo& seg_info);

  // @brief mlu subnet execute
  void OfflineMluSubnetRun(const cnrtModel_t& model,
                           const cnrtDataType_t& dtype,
                           const SegmentInfoUnit& unit_info,
                           void** cpuData);

  // @breif cpu subnet execute
  void OfflineCpuSubnetRun(const SegmentInfoUnit& unit_info);

#endif
  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype>>> layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype>>> blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*>> bottom_vecs_;
  vector<vector<int>> bottom_id_vecs_;
  vector<vector<bool>> bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*>> top_vecs_;
  vector<vector<int>> top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int>> param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int>> param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype>>> params_;
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  // Callbacks
  vector<Callback*> before_forward_;
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

  set<int> dump_top_idx_;

  NetParameter net_param_without_weights_;
#ifdef USE_MLU
  shared_ptr<NetData<Dtype>> net_data_;
  shared_ptr<ReshapeHelper<Dtype>> reshape_helper_;
  vector<shared_ptr<SubNet<Dtype>>> subnets_;
  map<string, cnrtQueue_t> name_to_cnrt_queue_;
  map<string, cnrtFunction_t> name_to_cnrt_func_;
  map<string, Blob<Dtype>*> name_to_data_;
  bool offline_init_flag_ = false;
  bool set_cpu_info_ = false;
  bool int8_mode_flag_ = false;
  int opt_level_ = 0;
#endif  // USE_MLU

  DISABLE_COPY_AND_ASSIGN(Net);
};
}  // namespace caffe

#endif  // INCLUDE_CAFFE_NET_HPP_
