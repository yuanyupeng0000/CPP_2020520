#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=line-too-long

import os
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
from ssd_detector.readers.object_detector_json import ObjectDetectorJson

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "../../.."))

# See more details about parameters in TensorFlow documentation tf.estimator
class train:
  annotation_path = os.path.join(ROOT_DIR, "./data/vlp_test/annotations_train.json")  # Path to the annotation file
  cache_type = "ENCODED"  # Type of data to save in memory, possible options: 'FULL', 'ENCODED', 'NONE'

  batch_size = 32                    # Number of images in the batch
  steps = 65000                      # Number of steps for which to train model
  max_steps = None                   # Number of total steps for which to train model
  save_checkpoints_steps = 1000      # Number of training steps when checkpoint should be saved
  keep_checkpoint_every_n_hours = 6  # Checkpoint should be saved forever after every n hours
  save_summary_steps = 100           # Number of steps when the summary information should be saved
  random_seed = 666                  # Random seed

  fill_with_current_image_mean = True  # Parameter of data transformer

  class execution:
    CUDA_VISIBLE_DEVICES = "0"             # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 2
    inter_op_parallelism_threads = 8
    transformer_parallel_calls = 4  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 8   # Number of batches to prefetch


class eval:
  annotation_path = {
    "train": os.path.join(ROOT_DIR, "./data/vlp_test/annotations_train.json"),
    "test": os.path.join(ROOT_DIR, "./data/vlp_test/annotations_test.json")
  }  # Dictionary with paths to annotations and its short names which will be displayed in the TensorBoard
  datasets = ["train", "test"]  # List of names from annotation_path dictionary on which evaluation will be launched
  vis_num = 2                  # Select random images for visualization in the TensorBoard
  save_images_step = 2          # Save images every 2-th evaluation
  batch_size = 2                # Number of images in the batch

  class execution:
    CUDA_VISIBLE_DEVICES = "0"             # Environment variable to control CUDA device used for evaluation
    per_process_gpu_memory_fraction = 0.5  # Fix extra memory allocation issue
    allow_growth = True                    # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 1
    transformer_parallel_calls = 1  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 1   # Number of batches to prefetch


class infer:
  out_subdir = "predictions"  # Name of folder in model directory where output json files with detections will be saved
  batch_size = 32             # Number of images in the batch

  class execution:
    CUDA_VISIBLE_DEVICES = "0"             # Environment variable to control cuda device used for training
    per_process_gpu_memory_fraction = 0.5  # Fix extra memory allocation issue
    allow_growth = True                    # Option which attempts to allocate only as much GPU memory based on runtime allocations

    intra_op_parallelism_threads = 2
    inter_op_parallelism_threads = 8
    transformer_parallel_calls = 4  # Number of parallel threads in data transformer/augmentation
    transformer_prefetch_size = 8   # Number of batches to prefetch


input_shape = (256, 256, 3)  # Input shape of the model (width, height, channels)
classes = ObjectDetectorJson.get_classes_from_coco_annotation(os.path.join(CURRENT_DIR, train.annotation_path))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'model')  # Path to the folder where all training and evaluation artifacts will be located
if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)


def learning_rate_schedule():  # Function which controls learning rate during training
  import tensorflow as tf
  step = tf.train.get_or_create_global_step()
  lr = tf.case([(tf.less(step, 1000), lambda: tf.constant(0.0004)),
                (tf.less(step, 10000), lambda: tf.constant(0.01)),
                (tf.less(step, 40000), lambda: tf.constant(0.005)),
                (tf.less(step, 55000), lambda: tf.constant(0.0005)),
                (tf.less(step, 65000), lambda: tf.constant(0.00005))])
  return lr


def optimizer(learning_rate):
  import tensorflow as tf
  return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)


detector_params = {
  "num_classes": len(classes),  # Number of classes to detect
  "priors_rule": "custom",      # Prior boxes rule for SSD, possible options: 'caffe', 'object_detection_api', 'custom'
  "priors": [
    [(0.068, 0.03), (0.052, 0.097)],
    [(0.18, 0.087), (0.11, 0.33), (0.43, 0.1)],
    [(0.26, 0.27), (0.34, 0.4), (0.2, 0.55)],
    [(0.37, 0.52)],
    [(0.48, 0.45)],
    [(0.63, 0.64), (0.77, 0.77), (0.95, 0.95)]
  ],
  "mobilenet_version": "v2",                # Version of mobilenet backbone, possible options: 'v1', 'v2'
  "initial_weights_path": "",               # Path to initial weights
  "depth_multiplier": 0.35,                 # MobileNet channels multiplier
  "weight_regularization": 1e-3,            # L2 weight regularization
  "learning_rate": learning_rate_schedule,  # Learning rate
  "optimizer": optimizer,                   # Optimizer
  "collect_priors_summary": False,          # Option to collect priors summary for further analysis
}
