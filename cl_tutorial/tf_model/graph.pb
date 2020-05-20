node {
  name: "Input/Inputs"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "Target/Targets"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "ModCosh/cosh/Cosh"
  op: "Cosh"
  input: "Input/Inputs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Convolution/random_normal/RandomStandardNormal"
  input: "ModCosh/Convolution/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal"
  op: "Add"
  input: "ModCosh/Convolution/random_normal/mul"
  input: "ModCosh/Convolution/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_weights"
  input: "ModCosh/Convolution/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/read"
  op: "Identity"
  input: "ModCosh/Convolution/conv_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/ModCosh/Convolution/conv_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Convolution/ModCosh/Convolution/conv_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/ModCosh/Convolution/conv_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Convolution/ModCosh/Convolution/conv_weights_0/tag"
  input: "ModCosh/Convolution/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal_1/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution/random_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal_1/mul"
  op: "Mul"
  input: "ModCosh/Convolution/random_normal_1/RandomStandardNormal"
  input: "ModCosh/Convolution/random_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution/random_normal_1"
  op: "Add"
  input: "ModCosh/Convolution/random_normal_1/mul"
  input: "ModCosh/Convolution/random_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_biases"
  input: "ModCosh/Convolution/random_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/read"
  op: "Identity"
  input: "ModCosh/Convolution/conv_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/Conv2D"
  op: "Conv2D"
  input: "ModCosh/cosh/Cosh"
  input: "ModCosh/Convolution/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/add"
  op: "Add"
  input: "ModCosh/Convolution/Conv2D"
  input: "ModCosh/Convolution/conv_biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation/Relu6"
  op: "Relu6"
  input: "ModCosh/Convolution/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_1/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Convolution_1/random_normal/RandomStandardNormal"
  input: "ModCosh/Convolution_1/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal"
  op: "Add"
  input: "ModCosh/Convolution_1/random_normal/mul"
  input: "ModCosh/Convolution_1/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_weights"
  input: "ModCosh/Convolution_1/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/read"
  op: "Identity"
  input: "ModCosh/Convolution_1/conv_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/ModCosh/Convolution_1/conv_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Convolution_1/ModCosh/Convolution_1/conv_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/ModCosh/Convolution_1/conv_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Convolution_1/ModCosh/Convolution_1/conv_weights_0/tag"
  input: "ModCosh/Convolution_1/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal_1/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_1/random_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal_1/mul"
  op: "Mul"
  input: "ModCosh/Convolution_1/random_normal_1/RandomStandardNormal"
  input: "ModCosh/Convolution_1/random_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_1/random_normal_1"
  op: "Add"
  input: "ModCosh/Convolution_1/random_normal_1/mul"
  input: "ModCosh/Convolution_1/random_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_biases"
  input: "ModCosh/Convolution_1/random_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/read"
  op: "Identity"
  input: "ModCosh/Convolution_1/conv_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/Conv2D"
  op: "Conv2D"
  input: "ModCosh/Activation/Relu6"
  input: "ModCosh/Convolution_1/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/add"
  op: "Add"
  input: "ModCosh/Convolution_1/Conv2D"
  input: "ModCosh/Convolution_1/conv_biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_1/Relu6"
  op: "Relu6"
  input: "ModCosh/Convolution_1/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Pooling/MaxPool"
  op: "MaxPool"
  input: "ModCosh/Activation_1/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "ModCosh/cosh_1/Cosh"
  op: "Cosh"
  input: "ModCosh/Pooling/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_2/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Convolution_2/random_normal/RandomStandardNormal"
  input: "ModCosh/Convolution_2/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal"
  op: "Add"
  input: "ModCosh/Convolution_2/random_normal/mul"
  input: "ModCosh/Convolution_2/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_weights"
  input: "ModCosh/Convolution_2/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/read"
  op: "Identity"
  input: "ModCosh/Convolution_2/conv_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/ModCosh/Convolution_2/conv_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Convolution_2/ModCosh/Convolution_2/conv_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/ModCosh/Convolution_2/conv_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Convolution_2/ModCosh/Convolution_2/conv_weights_0/tag"
  input: "ModCosh/Convolution_2/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal_1/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_2/random_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal_1/mul"
  op: "Mul"
  input: "ModCosh/Convolution_2/random_normal_1/RandomStandardNormal"
  input: "ModCosh/Convolution_2/random_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_2/random_normal_1"
  op: "Add"
  input: "ModCosh/Convolution_2/random_normal_1/mul"
  input: "ModCosh/Convolution_2/random_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_biases"
  input: "ModCosh/Convolution_2/random_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/read"
  op: "Identity"
  input: "ModCosh/Convolution_2/conv_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/Conv2D"
  op: "Conv2D"
  input: "ModCosh/cosh_1/Cosh"
  input: "ModCosh/Convolution_2/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/add"
  op: "Add"
  input: "ModCosh/Convolution_2/Conv2D"
  input: "ModCosh/Convolution_2/conv_biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_2/Relu6"
  op: "Relu6"
  input: "ModCosh/Convolution_2/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_3/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Convolution_3/random_normal/RandomStandardNormal"
  input: "ModCosh/Convolution_3/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal"
  op: "Add"
  input: "ModCosh/Convolution_3/random_normal/mul"
  input: "ModCosh/Convolution_3/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_weights"
  input: "ModCosh/Convolution_3/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/read"
  op: "Identity"
  input: "ModCosh/Convolution_3/conv_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/ModCosh/Convolution_3/conv_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Convolution_3/ModCosh/Convolution_3/conv_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/ModCosh/Convolution_3/conv_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Convolution_3/ModCosh/Convolution_3/conv_weights_0/tag"
  input: "ModCosh/Convolution_3/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal_1/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_3/random_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal_1/mul"
  op: "Mul"
  input: "ModCosh/Convolution_3/random_normal_1/RandomStandardNormal"
  input: "ModCosh/Convolution_3/random_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_3/random_normal_1"
  op: "Add"
  input: "ModCosh/Convolution_3/random_normal_1/mul"
  input: "ModCosh/Convolution_3/random_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_biases"
  input: "ModCosh/Convolution_3/random_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/read"
  op: "Identity"
  input: "ModCosh/Convolution_3/conv_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/Conv2D"
  op: "Conv2D"
  input: "ModCosh/Activation_2/Relu6"
  input: "ModCosh/Convolution_3/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/add"
  op: "Add"
  input: "ModCosh/Convolution_3/Conv2D"
  input: "ModCosh/Convolution_3/conv_biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_3/Relu6"
  op: "Relu6"
  input: "ModCosh/Convolution_3/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Pooling_1/MaxPool"
  op: "MaxPool"
  input: "ModCosh/Activation_3/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "ModCosh/cosh_2/Cosh"
  op: "Cosh"
  input: "ModCosh/Pooling_1/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_4/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Convolution_4/random_normal/RandomStandardNormal"
  input: "ModCosh/Convolution_4/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal"
  op: "Add"
  input: "ModCosh/Convolution_4/random_normal/mul"
  input: "ModCosh/Convolution_4/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_weights"
  input: "ModCosh/Convolution_4/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/read"
  op: "Identity"
  input: "ModCosh/Convolution_4/conv_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/ModCosh/Convolution_4/conv_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Convolution_4/ModCosh/Convolution_4/conv_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/ModCosh/Convolution_4/conv_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Convolution_4/ModCosh/Convolution_4/conv_weights_0/tag"
  input: "ModCosh/Convolution_4/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal_1/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_4/random_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal_1/mul"
  op: "Mul"
  input: "ModCosh/Convolution_4/random_normal_1/RandomStandardNormal"
  input: "ModCosh/Convolution_4/random_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_4/random_normal_1"
  op: "Add"
  input: "ModCosh/Convolution_4/random_normal_1/mul"
  input: "ModCosh/Convolution_4/random_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_biases"
  input: "ModCosh/Convolution_4/random_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/read"
  op: "Identity"
  input: "ModCosh/Convolution_4/conv_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/Conv2D"
  op: "Conv2D"
  input: "ModCosh/cosh_2/Cosh"
  input: "ModCosh/Convolution_4/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/add"
  op: "Add"
  input: "ModCosh/Convolution_4/Conv2D"
  input: "ModCosh/Convolution_4/conv_biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_4/Relu6"
  op: "Relu6"
  input: "ModCosh/Convolution_4/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_5/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Convolution_5/random_normal/RandomStandardNormal"
  input: "ModCosh/Convolution_5/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal"
  op: "Add"
  input: "ModCosh/Convolution_5/random_normal/mul"
  input: "ModCosh/Convolution_5/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_weights"
  input: "ModCosh/Convolution_5/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/read"
  op: "Identity"
  input: "ModCosh/Convolution_5/conv_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/ModCosh/Convolution_5/conv_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Convolution_5/ModCosh/Convolution_5/conv_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/ModCosh/Convolution_5/conv_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Convolution_5/ModCosh/Convolution_5/conv_weights_0/tag"
  input: "ModCosh/Convolution_5/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal_1/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Convolution_5/random_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal_1/mul"
  op: "Mul"
  input: "ModCosh/Convolution_5/random_normal_1/RandomStandardNormal"
  input: "ModCosh/Convolution_5/random_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_5/random_normal_1"
  op: "Add"
  input: "ModCosh/Convolution_5/random_normal_1/mul"
  input: "ModCosh/Convolution_5/random_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_biases"
  input: "ModCosh/Convolution_5/random_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/read"
  op: "Identity"
  input: "ModCosh/Convolution_5/conv_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/Conv2D"
  op: "Conv2D"
  input: "ModCosh/Activation_4/Relu6"
  input: "ModCosh/Convolution_5/conv_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/add"
  op: "Add"
  input: "ModCosh/Convolution_5/Conv2D"
  input: "ModCosh/Convolution_5/conv_biases/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_5/Relu6"
  op: "Relu6"
  input: "ModCosh/Convolution_5/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Pooling_2/MaxPool"
  op: "MaxPool"
  input: "ModCosh/Activation_5/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "ModCosh/Flatten/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\000\200\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Flatten/Reshape"
  op: "Reshape"
  input: "ModCosh/Pooling_2/MaxPool"
  input: "ModCosh/Flatten/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Dense/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\200\000\000\310\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Dense/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Dense/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Dense/random_normal/RandomStandardNormal"
  input: "ModCosh/Dense/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense/random_normal"
  op: "Add"
  input: "ModCosh/Dense/random_normal/mul"
  input: "ModCosh/Dense/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32768
        }
        dim {
          size: 200
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Assign"
  op: "Assign"
  input: "ModCosh/Dense/dense_weights"
  input: "ModCosh/Dense/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/read"
  op: "Identity"
  input: "ModCosh/Dense/dense_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/ModCosh/Dense/dense_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Dense/ModCosh/Dense/dense_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/ModCosh/Dense/dense_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Dense/ModCosh/Dense/dense_weights_0/tag"
  input: "ModCosh/Dense/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense/dense_biases/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 200
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_biases/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_biases/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_biases/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Dense/dense_biases/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Dense/dense_biases/mul"
  op: "Mul"
  input: "ModCosh/Dense/dense_biases/RandomStandardNormal"
  input: "ModCosh/Dense/dense_biases/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense/dense_biases"
  op: "Add"
  input: "ModCosh/Dense/dense_biases/mul"
  input: "ModCosh/Dense/dense_biases/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 200
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Assign"
  op: "Assign"
  input: "ModCosh/Dense/Variable"
  input: "ModCosh/Dense/dense_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/read"
  op: "Identity"
  input: "ModCosh/Dense/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/MatMul"
  op: "MatMul"
  input: "ModCosh/Flatten/Reshape"
  input: "ModCosh/Dense/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "ModCosh/Dense/add"
  op: "Add"
  input: "ModCosh/Dense/MatMul"
  input: "ModCosh/Dense/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_6/Sigmoid"
  op: "Sigmoid"
  input: "ModCosh/Dense/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_1/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\310\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Dense_1/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Dense_1/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Dense_1/random_normal/RandomStandardNormal"
  input: "ModCosh/Dense_1/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_1/random_normal"
  op: "Add"
  input: "ModCosh/Dense_1/random_normal/mul"
  input: "ModCosh/Dense_1/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 200
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Assign"
  op: "Assign"
  input: "ModCosh/Dense_1/dense_weights"
  input: "ModCosh/Dense_1/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/read"
  op: "Identity"
  input: "ModCosh/Dense_1/dense_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/ModCosh/Dense_1/dense_weights_0/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "ModCosh/Dense_1/ModCosh/Dense_1/dense_weights_0"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/ModCosh/Dense_1/dense_weights_0"
  op: "HistogramSummary"
  input: "ModCosh/Dense_1/ModCosh/Dense_1/dense_weights_0/tag"
  input: "ModCosh/Dense_1/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_biases/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_biases/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_biases/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_biases/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Dense_1/dense_biases/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_biases/mul"
  op: "Mul"
  input: "ModCosh/Dense_1/dense_biases/RandomStandardNormal"
  input: "ModCosh/Dense_1/dense_biases/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_biases"
  op: "Add"
  input: "ModCosh/Dense_1/dense_biases/mul"
  input: "ModCosh/Dense_1/dense_biases/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Assign"
  op: "Assign"
  input: "ModCosh/Dense_1/Variable"
  input: "ModCosh/Dense_1/dense_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/read"
  op: "Identity"
  input: "ModCosh/Dense_1/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/MatMul"
  op: "MatMul"
  input: "ModCosh/Activation_6/Sigmoid"
  input: "ModCosh/Dense_1/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "ModCosh/Dense_1/add"
  op: "Add"
  input: "ModCosh/Dense_1/MatMul"
  input: "ModCosh/Dense_1/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_7/Sigmoid"
  op: "Sigmoid"
  input: "ModCosh/Dense_1/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_2/random_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: " \000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/random_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/random_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/random_normal/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Dense_2/random_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Dense_2/random_normal/mul"
  op: "Mul"
  input: "ModCosh/Dense_2/random_normal/RandomStandardNormal"
  input: "ModCosh/Dense_2/random_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_2/random_normal"
  op: "Add"
  input: "ModCosh/Dense_2/random_normal/mul"
  input: "ModCosh/Dense_2/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Assign"
  op: "Assign"
  input: "ModCosh/Dense_2/dense_weights"
  input: "ModCosh/Dense_2/random_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/read"
  op: "Identity"
  input: "ModCosh/Dense_2/dense_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_biases/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_biases/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_biases/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_biases/RandomStandardNormal"
  op: "RandomStandardNormal"
  input: "ModCosh/Dense_2/dense_biases/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_biases/mul"
  op: "Mul"
  input: "ModCosh/Dense_2/dense_biases/RandomStandardNormal"
  input: "ModCosh/Dense_2/dense_biases/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_biases"
  op: "Add"
  input: "ModCosh/Dense_2/dense_biases/mul"
  input: "ModCosh/Dense_2/dense_biases/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Assign"
  op: "Assign"
  input: "ModCosh/Dense_2/Variable"
  input: "ModCosh/Dense_2/dense_biases"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/read"
  op: "Identity"
  input: "ModCosh/Dense_2/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/MatMul"
  op: "MatMul"
  input: "ModCosh/Activation_7/Sigmoid"
  input: "ModCosh/Dense_2/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "ModCosh/Dense_2/add"
  op: "Add"
  input: "ModCosh/Dense_2/MatMul"
  input: "ModCosh/Dense_2/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ModCosh/Activation_8/softmax_output"
  op: "Softmax"
  input: "ModCosh/Dense_2/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/global_step/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/global_step"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "Optimization/global_step/Assign"
  op: "Assign"
  input: "Optimization/global_step"
  input: "Optimization/global_step/initial_value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/global_step"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/global_step/read"
  op: "Identity"
  input: "Optimization/global_step"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/global_step"
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Shape"
  op: "Shape"
  input: "ModCosh/Dense_2/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Shape_1"
  op: "Shape"
  input: "ModCosh/Dense_2/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Sub/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Sub"
  op: "Sub"
  input: "Optimization/softmax_cross_entropy_with_logits/Rank_1"
  input: "Optimization/softmax_cross_entropy_with_logits/Sub/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice/begin"
  op: "Pack"
  input: "Optimization/softmax_cross_entropy_with_logits/Sub"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice"
  op: "Slice"
  input: "Optimization/softmax_cross_entropy_with_logits/Shape_1"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice/begin"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/concat/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/concat"
  op: "ConcatV2"
  input: "Optimization/softmax_cross_entropy_with_logits/concat/values_0"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice"
  input: "Optimization/softmax_cross_entropy_with_logits/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Reshape"
  op: "Reshape"
  input: "ModCosh/Dense_2/add"
  input: "Optimization/softmax_cross_entropy_with_logits/concat"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Shape_2"
  op: "Shape"
  input: "Target/Targets"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Sub_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Sub_1"
  op: "Sub"
  input: "Optimization/softmax_cross_entropy_with_logits/Rank_2"
  input: "Optimization/softmax_cross_entropy_with_logits/Sub_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice_1/begin"
  op: "Pack"
  input: "Optimization/softmax_cross_entropy_with_logits/Sub_1"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice_1/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice_1"
  op: "Slice"
  input: "Optimization/softmax_cross_entropy_with_logits/Shape_2"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice_1/begin"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice_1/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/concat_1/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/concat_1/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/concat_1"
  op: "ConcatV2"
  input: "Optimization/softmax_cross_entropy_with_logits/concat_1/values_0"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice_1"
  input: "Optimization/softmax_cross_entropy_with_logits/concat_1/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Reshape_1"
  op: "Reshape"
  input: "Target/Targets"
  input: "Optimization/softmax_cross_entropy_with_logits/concat_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits"
  op: "SoftmaxCrossEntropyWithLogits"
  input: "Optimization/softmax_cross_entropy_with_logits/Reshape"
  input: "Optimization/softmax_cross_entropy_with_logits/Reshape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Sub_2/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Sub_2"
  op: "Sub"
  input: "Optimization/softmax_cross_entropy_with_logits/Rank"
  input: "Optimization/softmax_cross_entropy_with_logits/Sub_2/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice_2/begin"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice_2/size"
  op: "Pack"
  input: "Optimization/softmax_cross_entropy_with_logits/Sub_2"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Slice_2"
  op: "Slice"
  input: "Optimization/softmax_cross_entropy_with_logits/Shape"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice_2/begin"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice_2/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/softmax_cross_entropy_with_logits/Reshape_2"
  op: "Reshape"
  input: "Optimization/softmax_cross_entropy_with_logits"
  input: "Optimization/softmax_cross_entropy_with_logits/Slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/Mean"
  op: "Mean"
  input: "Optimization/softmax_cross_entropy_with_logits/Reshape_2"
  input: "Optimization/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/cost/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "Optimization/cost"
      }
    }
  }
}
node {
  name: "Optimization/cost"
  op: "ScalarSummary"
  input: "Optimization/cost/tags"
  input: "Optimization/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "Optimization/gradients/grad_ys_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "Optimization/gradients/Fill"
  op: "Fill"
  input: "Optimization/gradients/Shape"
  input: "Optimization/gradients/grad_ys_0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/Fill"
  input: "Optimization/gradients/Optimization/Mean_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Shape"
  op: "Shape"
  input: "Optimization/softmax_cross_entropy_with_logits/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Tile"
  op: "Tile"
  input: "Optimization/gradients/Optimization/Mean_grad/Reshape"
  input: "Optimization/gradients/Optimization/Mean_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Shape_1"
  op: "Shape"
  input: "Optimization/softmax_cross_entropy_with_logits/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Shape_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Prod"
  op: "Prod"
  input: "Optimization/gradients/Optimization/Mean_grad/Shape_1"
  input: "Optimization/gradients/Optimization/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Prod_1"
  op: "Prod"
  input: "Optimization/gradients/Optimization/Mean_grad/Shape_2"
  input: "Optimization/gradients/Optimization/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Maximum"
  op: "Maximum"
  input: "Optimization/gradients/Optimization/Mean_grad/Prod_1"
  input: "Optimization/gradients/Optimization/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/floordiv"
  op: "FloorDiv"
  input: "Optimization/gradients/Optimization/Mean_grad/Prod"
  input: "Optimization/gradients/Optimization/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/Cast"
  op: "Cast"
  input: "Optimization/gradients/Optimization/Mean_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/Mean_grad/truediv"
  op: "RealDiv"
  input: "Optimization/gradients/Optimization/Mean_grad/Tile"
  input: "Optimization/gradients/Optimization/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape"
  op: "Shape"
  input: "Optimization/softmax_cross_entropy_with_logits"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/Optimization/Mean_grad/truediv"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/zeros_like"
  op: "ZerosLike"
  input: "Optimization/softmax_cross_entropy_with_logits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims"
  op: "ExpandDims"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul"
  op: "Mul"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims"
  input: "Optimization/softmax_cross_entropy_with_logits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/LogSoftmax"
  op: "LogSoftmax"
  input: "Optimization/softmax_cross_entropy_with_logits/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/Neg"
  op: "Neg"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/LogSoftmax"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims_1"
  op: "ExpandDims"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul_1"
  op: "Mul"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/ExpandDims_1"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/Neg"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul"
  input: "^Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul_1"
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul"
  input: "^Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul"
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul_1"
  input: "^Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/mul_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_grad/Shape"
  op: "Shape"
  input: "ModCosh/Dense_2/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits_grad/tuple/control_dependency"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Dense_2/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_grad/Reshape"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/Optimization/softmax_cross_entropy_with_logits/Reshape_grad/Reshape"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_2/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul"
  op: "MatMul"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/control_dependency"
  input: "ModCosh/Dense_2/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "ModCosh/Activation_7/Sigmoid"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul"
  input: "^Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul_1"
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul"
  input: "^Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul_1"
  input: "^Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_2/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_7/Sigmoid_grad/SigmoidGrad"
  op: "SigmoidGrad"
  input: "ModCosh/Activation_7/Sigmoid"
  input: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Dense_1/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_7/Sigmoid_grad/SigmoidGrad"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_7/Sigmoid_grad/SigmoidGrad"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_1/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul"
  op: "MatMul"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/control_dependency"
  input: "ModCosh/Dense_1/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "ModCosh/Activation_6/Sigmoid"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul"
  input: "^Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul_1"
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul"
  input: "^Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul_1"
  input: "^Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense_1/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_6/Sigmoid_grad/SigmoidGrad"
  op: "SigmoidGrad"
  input: "ModCosh/Activation_6/Sigmoid"
  input: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Dense/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 200
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_6/Sigmoid_grad/SigmoidGrad"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_6/Sigmoid_grad/SigmoidGrad"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Dense/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Dense/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Dense/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Dense/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul"
  op: "MatMul"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/tuple/control_dependency"
  input: "ModCosh/Dense/dense_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "ModCosh/Flatten/Reshape"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul"
  input: "^Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul_1"
}
node {
  name: "Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul"
  input: "^Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul_1"
  input: "^Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Dense/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Flatten/Reshape_grad/Shape"
  op: "Shape"
  input: "ModCosh/Pooling_2/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Flatten/Reshape_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/control_dependency"
  input: "Optimization/gradients/ModCosh/Flatten/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Pooling_2/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "ModCosh/Activation_5/Relu6"
  input: "ModCosh/Pooling_2/MaxPool"
  input: "Optimization/gradients/ModCosh/Flatten/Reshape_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_5/Relu6_grad/Relu6Grad"
  op: "Relu6Grad"
  input: "Optimization/gradients/ModCosh/Pooling_2/MaxPool_grad/MaxPoolGrad"
  input: "ModCosh/Activation_5/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Convolution_5/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_5/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_5/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_5/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ModCosh/Activation_4/Relu6"
  input: "ModCosh/Convolution_5/conv_weights/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/ShapeN"
  input: "ModCosh/Convolution_5/conv_weights/read"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ModCosh/Activation_4/Relu6"
  input: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/ShapeN:1"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropInput"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropInput"
  input: "^Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_4/Relu6_grad/Relu6Grad"
  op: "Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/control_dependency"
  input: "ModCosh/Activation_4/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Convolution_4/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_4/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_4/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_4/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ModCosh/cosh_2/Cosh"
  input: "ModCosh/Convolution_4/conv_weights/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/ShapeN"
  input: "ModCosh/Convolution_4/conv_weights/read"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ModCosh/cosh_2/Cosh"
  input: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/ShapeN:1"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropInput"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropInput"
  input: "^Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/cosh_2/Cosh_grad/Sinh"
  op: "Sinh"
  input: "ModCosh/Pooling_1/MaxPool"
  input: "^Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/cosh_2/Cosh_grad/mul"
  op: "Mul"
  input: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/control_dependency"
  input: "Optimization/gradients/ModCosh/cosh_2/Cosh_grad/Sinh"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Pooling_1/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "ModCosh/Activation_3/Relu6"
  input: "ModCosh/Pooling_1/MaxPool"
  input: "Optimization/gradients/ModCosh/cosh_2/Cosh_grad/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_3/Relu6_grad/Relu6Grad"
  op: "Relu6Grad"
  input: "Optimization/gradients/ModCosh/Pooling_1/MaxPool_grad/MaxPoolGrad"
  input: "ModCosh/Activation_3/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Convolution_3/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_3/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_3/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_3/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ModCosh/Activation_2/Relu6"
  input: "ModCosh/Convolution_3/conv_weights/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/ShapeN"
  input: "ModCosh/Convolution_3/conv_weights/read"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ModCosh/Activation_2/Relu6"
  input: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/ShapeN:1"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropInput"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropInput"
  input: "^Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_2/Relu6_grad/Relu6Grad"
  op: "Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/control_dependency"
  input: "ModCosh/Activation_2/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Convolution_2/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_2/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_2/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_2/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ModCosh/cosh_1/Cosh"
  input: "ModCosh/Convolution_2/conv_weights/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/ShapeN"
  input: "ModCosh/Convolution_2/conv_weights/read"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ModCosh/cosh_1/Cosh"
  input: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/ShapeN:1"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropInput"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropInput"
  input: "^Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/cosh_1/Cosh_grad/Sinh"
  op: "Sinh"
  input: "ModCosh/Pooling/MaxPool"
  input: "^Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/cosh_1/Cosh_grad/mul"
  op: "Mul"
  input: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/control_dependency"
  input: "Optimization/gradients/ModCosh/cosh_1/Cosh_grad/Sinh"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Pooling/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "ModCosh/Activation_1/Relu6"
  input: "ModCosh/Pooling/MaxPool"
  input: "Optimization/gradients/ModCosh/cosh_1/Cosh_grad/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation_1/Relu6_grad/Relu6Grad"
  op: "Relu6Grad"
  input: "Optimization/gradients/ModCosh/Pooling/MaxPool_grad/MaxPoolGrad"
  input: "ModCosh/Activation_1/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Convolution_1/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_1/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation_1/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_1/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ModCosh/Activation/Relu6"
  input: "ModCosh/Convolution_1/conv_weights/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/ShapeN"
  input: "ModCosh/Convolution_1/conv_weights/read"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ModCosh/Activation/Relu6"
  input: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/ShapeN:1"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropInput"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropInput"
  input: "^Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Activation/Relu6_grad/Relu6Grad"
  op: "Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/control_dependency"
  input: "ModCosh/Activation/Relu6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/Shape"
  op: "Shape"
  input: "ModCosh/Convolution/Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Shape"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/Sum"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/Reshape"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Sum"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/Sum_1"
  op: "Sum"
  input: "Optimization/gradients/ModCosh/Activation/Relu6_grad/Relu6Grad"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/Reshape_1"
  op: "Reshape"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Sum_1"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution/add_grad/Reshape_1"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Reshape"
  input: "^Optimization/gradients/ModCosh/Convolution/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/Reshape_1"
  input: "^Optimization/gradients/ModCosh/Convolution/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ModCosh/cosh/Cosh"
  input: "ModCosh/Convolution/conv_weights/read"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/ShapeN"
  input: "ModCosh/Convolution/conv_weights/read"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ModCosh/cosh/Cosh"
  input: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/ShapeN:1"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropInput"
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropInput"
  input: "^Optimization/gradients/ModCosh/Convolution/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropFilter"
  input: "^Optimization/gradients/ModCosh/Convolution/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/gradients/ModCosh/Convolution/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "Optimization/beta1_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "Optimization/beta1_power"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "Optimization/beta1_power/Assign"
  op: "Assign"
  input: "Optimization/beta1_power"
  input: "Optimization/beta1_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/beta1_power/read"
  op: "Identity"
  input: "Optimization/beta1_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "Optimization/beta2_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "Optimization/beta2_power"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "Optimization/beta2_power/Assign"
  op: "Assign"
  input: "Optimization/beta2_power"
  input: "Optimization/beta2_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/beta2_power/read"
  op: "Identity"
  input: "Optimization/beta2_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution/conv_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_weights/Adam"
  input: "ModCosh/Convolution/conv_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution/conv_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\003\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution/conv_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_weights/Adam_1"
  input: "ModCosh/Convolution/conv_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution/conv_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_biases/Adam"
  input: "ModCosh/Convolution/conv_biases/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution/conv_biases/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_biases/Adam_1"
  input: "ModCosh/Convolution/conv_biases/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution/conv_biases/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution/conv_biases/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_1/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_1/conv_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_weights/Adam"
  input: "ModCosh/Convolution_1/conv_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_1/conv_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_biases/Adam"
  input: "ModCosh/Convolution_1/conv_biases/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_1/conv_biases/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_biases/Adam_1"
  input: "ModCosh/Convolution_1/conv_biases/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_1/conv_biases/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_1/conv_biases/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_2/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_2/conv_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_weights/Adam"
  input: "ModCosh/Convolution_2/conv_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_2/conv_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_biases/Adam"
  input: "ModCosh/Convolution_2/conv_biases/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_2/conv_biases/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_biases/Adam_1"
  input: "ModCosh/Convolution_2/conv_biases/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_2/conv_biases/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_2/conv_biases/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_3/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_3/conv_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_weights/Adam"
  input: "ModCosh/Convolution_3/conv_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_3/conv_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_biases/Adam"
  input: "ModCosh/Convolution_3/conv_biases/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_3/conv_biases/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_biases/Adam_1"
  input: "ModCosh/Convolution_3/conv_biases/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_3/conv_biases/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_3/conv_biases/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_4/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_4/conv_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_weights/Adam"
  input: "ModCosh/Convolution_4/conv_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_4/conv_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_biases/Adam"
  input: "ModCosh/Convolution_4/conv_biases/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_4/conv_biases/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_biases/Adam_1"
  input: "ModCosh/Convolution_4/conv_biases/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_4/conv_biases/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_4/conv_biases/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_5/conv_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_5/conv_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_weights/Adam"
  input: "ModCosh/Convolution_5/conv_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_5/conv_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\200\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 128
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_biases/Adam"
  input: "ModCosh/Convolution_5/conv_biases/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam/read"
  op: "Identity"
  input: "ModCosh/Convolution_5/conv_biases/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_biases/Adam_1"
  input: "ModCosh/Convolution_5/conv_biases/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Convolution_5/conv_biases/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Convolution_5/conv_biases/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\200\000\000\310\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Dense/dense_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Dense/dense_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32768
        }
        dim {
          size: 200
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Dense/dense_weights/Adam"
  input: "ModCosh/Dense/dense_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Dense/dense_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\200\000\000\310\000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Dense/dense_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Dense/dense_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32768
        }
        dim {
          size: 200
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Dense/dense_weights/Adam_1"
  input: "ModCosh/Dense/dense_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense/dense_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Dense/dense_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 200
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 200
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Dense/Variable/Adam"
  input: "ModCosh/Dense/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam/read"
  op: "Identity"
  input: "ModCosh/Dense/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 200
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 200
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Dense/Variable/Adam_1"
  input: "ModCosh/Dense/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense/Variable/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Dense/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\310\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Dense_1/dense_weights/Adam/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Dense_1/dense_weights/Adam/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 200
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Dense_1/dense_weights/Adam"
  input: "ModCosh/Dense_1/dense_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Dense_1/dense_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\310\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam_1/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam_1/Initializer/zeros"
  op: "Fill"
  input: "ModCosh/Dense_1/dense_weights/Adam_1/Initializer/zeros/shape_as_tensor"
  input: "ModCosh/Dense_1/dense_weights/Adam_1/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 200
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Dense_1/dense_weights/Adam_1"
  input: "ModCosh/Dense_1/dense_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_1/dense_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Dense_1/dense_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Dense_1/Variable/Adam"
  input: "ModCosh/Dense_1/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam/read"
  op: "Identity"
  input: "ModCosh/Dense_1/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Dense_1/Variable/Adam_1"
  input: "ModCosh/Dense_1/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_1/Variable/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Dense_1/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Dense_2/dense_weights/Adam"
  input: "ModCosh/Dense_2/dense_weights/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam/read"
  op: "Identity"
  input: "ModCosh/Dense_2/dense_weights/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Dense_2/dense_weights/Adam_1"
  input: "ModCosh/Dense_2/dense_weights/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_2/dense_weights/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Dense_2/dense_weights/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam/Assign"
  op: "Assign"
  input: "ModCosh/Dense_2/Variable/Adam"
  input: "ModCosh/Dense_2/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam/read"
  op: "Identity"
  input: "ModCosh/Dense_2/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam_1/Assign"
  op: "Assign"
  input: "ModCosh/Dense_2/Variable/Adam_1"
  input: "ModCosh/Dense_2/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ModCosh/Dense_2/Variable/Adam_1/read"
  op: "Identity"
  input: "ModCosh/Dense_2/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
}
node {
  name: "Optimization/Adam/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "Optimization/Adam/beta1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "Optimization/Adam/beta2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "Optimization/Adam/epsilon"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.99999993922529e-09
      }
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution/conv_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution/conv_weights"
  input: "ModCosh/Convolution/conv_weights/Adam"
  input: "ModCosh/Convolution/conv_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution/conv_biases/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution/conv_biases"
  input: "ModCosh/Convolution/conv_biases/Adam"
  input: "ModCosh/Convolution/conv_biases/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_1/conv_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_1/conv_weights"
  input: "ModCosh/Convolution_1/conv_weights/Adam"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_1/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_1/conv_biases/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_1/conv_biases"
  input: "ModCosh/Convolution_1/conv_biases/Adam"
  input: "ModCosh/Convolution_1/conv_biases/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_1/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_2/conv_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_2/conv_weights"
  input: "ModCosh/Convolution_2/conv_weights/Adam"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_2/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_2/conv_biases/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_2/conv_biases"
  input: "ModCosh/Convolution_2/conv_biases/Adam"
  input: "ModCosh/Convolution_2/conv_biases/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_2/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_3/conv_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_3/conv_weights"
  input: "ModCosh/Convolution_3/conv_weights/Adam"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_3/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_3/conv_biases/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_3/conv_biases"
  input: "ModCosh/Convolution_3/conv_biases/Adam"
  input: "ModCosh/Convolution_3/conv_biases/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_3/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_4/conv_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_4/conv_weights"
  input: "ModCosh/Convolution_4/conv_weights/Adam"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_4/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_4/conv_biases/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_4/conv_biases"
  input: "ModCosh/Convolution_4/conv_biases/Adam"
  input: "ModCosh/Convolution_4/conv_biases/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_4/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_5/conv_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_5/conv_weights"
  input: "ModCosh/Convolution_5/conv_weights/Adam"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_5/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Convolution_5/conv_biases/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Convolution_5/conv_biases"
  input: "ModCosh/Convolution_5/conv_biases/Adam"
  input: "ModCosh/Convolution_5/conv_biases/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Convolution_5/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Dense/dense_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Dense/dense_weights"
  input: "ModCosh/Dense/dense_weights/Adam"
  input: "ModCosh/Dense/dense_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Dense/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Dense/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Dense/Variable"
  input: "ModCosh/Dense/Variable/Adam"
  input: "ModCosh/Dense/Variable/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Dense/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Dense_1/dense_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Dense_1/dense_weights"
  input: "ModCosh/Dense_1/dense_weights/Adam"
  input: "ModCosh/Dense_1/dense_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Dense_1/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Dense_1/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Dense_1/Variable"
  input: "ModCosh/Dense_1/Variable/Adam"
  input: "ModCosh/Dense_1/Variable/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Dense_1/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Dense_2/dense_weights/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Dense_2/dense_weights"
  input: "ModCosh/Dense_2/dense_weights/Adam"
  input: "ModCosh/Dense_2/dense_weights/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Dense_2/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/update_ModCosh/Dense_2/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "ModCosh/Dense_2/Variable"
  input: "ModCosh/Dense_2/Variable/Adam"
  input: "ModCosh/Dense_2/Variable/Adam_1"
  input: "Optimization/beta1_power/read"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/learning_rate"
  input: "Optimization/Adam/beta1"
  input: "Optimization/Adam/beta2"
  input: "Optimization/Adam/epsilon"
  input: "Optimization/gradients/ModCosh/Dense_2/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Optimization/Adam/mul"
  op: "Mul"
  input: "Optimization/beta1_power/read"
  input: "Optimization/Adam/beta1"
  input: "^Optimization/Adam/update_ModCosh/Convolution/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_1/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_1/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_2/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_2/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_3/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_3/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_4/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_4/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_5/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_5/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense/dense_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_1/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_1/dense_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_2/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_2/dense_weights/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "Optimization/Adam/Assign"
  op: "Assign"
  input: "Optimization/beta1_power"
  input: "Optimization/Adam/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/Adam/mul_1"
  op: "Mul"
  input: "Optimization/beta2_power/read"
  input: "Optimization/Adam/beta2"
  input: "^Optimization/Adam/update_ModCosh/Convolution/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_1/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_1/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_2/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_2/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_3/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_3/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_4/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_4/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_5/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_5/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense/dense_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_1/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_1/dense_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_2/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_2/dense_weights/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
}
node {
  name: "Optimization/Adam/Assign_1"
  op: "Assign"
  input: "Optimization/beta2_power"
  input: "Optimization/Adam/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Optimization/Adam/update"
  op: "NoOp"
  input: "^Optimization/Adam/Assign"
  input: "^Optimization/Adam/Assign_1"
  input: "^Optimization/Adam/update_ModCosh/Convolution/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_1/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_1/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_2/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_2/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_3/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_3/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_4/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_4/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_5/conv_biases/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Convolution_5/conv_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense/dense_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_1/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_1/dense_weights/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_2/Variable/ApplyAdam"
  input: "^Optimization/Adam/update_ModCosh/Dense_2/dense_weights/ApplyAdam"
}
node {
  name: "Optimization/Adam/value"
  op: "Const"
  input: "^Optimization/Adam/update"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/global_step"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Optimization/Adam"
  op: "AssignAdd"
  input: "Optimization/global_step"
  input: "Optimization/Adam/value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/global_step"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "accuracy/ArgMax/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "accuracy/ArgMax"
  op: "ArgMax"
  input: "ModCosh/Activation_8/softmax_output"
  input: "accuracy/ArgMax/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "output_type"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "accuracy/ArgMax_1/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "accuracy/ArgMax_1"
  op: "ArgMax"
  input: "Target/Targets"
  input: "accuracy/ArgMax_1/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "output_type"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "accuracy/Equal"
  op: "Equal"
  input: "accuracy/ArgMax"
  input: "accuracy/ArgMax_1"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "accuracy/Cast"
  op: "Cast"
  input: "accuracy/Equal"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "accuracy/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "accuracy/Mean"
  op: "Mean"
  input: "accuracy/Cast"
  input: "accuracy/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "save/filename/input"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/filename"
  op: "PlaceholderWithDefault"
  input: "save/filename/input"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "save/Const"
  op: "PlaceholderWithDefault"
  input: "save/filename"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "save/SaveV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 57
          }
        }
        string_val: "ModCosh/Convolution/conv_biases"
        string_val: "ModCosh/Convolution/conv_biases/Adam"
        string_val: "ModCosh/Convolution/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution/conv_weights"
        string_val: "ModCosh/Convolution/conv_weights/Adam"
        string_val: "ModCosh/Convolution/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_1/conv_biases"
        string_val: "ModCosh/Convolution_1/conv_biases/Adam"
        string_val: "ModCosh/Convolution_1/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_1/conv_weights"
        string_val: "ModCosh/Convolution_1/conv_weights/Adam"
        string_val: "ModCosh/Convolution_1/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_2/conv_biases"
        string_val: "ModCosh/Convolution_2/conv_biases/Adam"
        string_val: "ModCosh/Convolution_2/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_2/conv_weights"
        string_val: "ModCosh/Convolution_2/conv_weights/Adam"
        string_val: "ModCosh/Convolution_2/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_3/conv_biases"
        string_val: "ModCosh/Convolution_3/conv_biases/Adam"
        string_val: "ModCosh/Convolution_3/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_3/conv_weights"
        string_val: "ModCosh/Convolution_3/conv_weights/Adam"
        string_val: "ModCosh/Convolution_3/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_4/conv_biases"
        string_val: "ModCosh/Convolution_4/conv_biases/Adam"
        string_val: "ModCosh/Convolution_4/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_4/conv_weights"
        string_val: "ModCosh/Convolution_4/conv_weights/Adam"
        string_val: "ModCosh/Convolution_4/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_5/conv_biases"
        string_val: "ModCosh/Convolution_5/conv_biases/Adam"
        string_val: "ModCosh/Convolution_5/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_5/conv_weights"
        string_val: "ModCosh/Convolution_5/conv_weights/Adam"
        string_val: "ModCosh/Convolution_5/conv_weights/Adam_1"
        string_val: "ModCosh/Dense/Variable"
        string_val: "ModCosh/Dense/Variable/Adam"
        string_val: "ModCosh/Dense/Variable/Adam_1"
        string_val: "ModCosh/Dense/dense_weights"
        string_val: "ModCosh/Dense/dense_weights/Adam"
        string_val: "ModCosh/Dense/dense_weights/Adam_1"
        string_val: "ModCosh/Dense_1/Variable"
        string_val: "ModCosh/Dense_1/Variable/Adam"
        string_val: "ModCosh/Dense_1/Variable/Adam_1"
        string_val: "ModCosh/Dense_1/dense_weights"
        string_val: "ModCosh/Dense_1/dense_weights/Adam"
        string_val: "ModCosh/Dense_1/dense_weights/Adam_1"
        string_val: "ModCosh/Dense_2/Variable"
        string_val: "ModCosh/Dense_2/Variable/Adam"
        string_val: "ModCosh/Dense_2/Variable/Adam_1"
        string_val: "ModCosh/Dense_2/dense_weights"
        string_val: "ModCosh/Dense_2/dense_weights/Adam"
        string_val: "ModCosh/Dense_2/dense_weights/Adam_1"
        string_val: "Optimization/beta1_power"
        string_val: "Optimization/beta2_power"
        string_val: "Optimization/global_step"
      }
    }
  }
}
node {
  name: "save/SaveV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 57
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/SaveV2"
  op: "SaveV2"
  input: "save/Const"
  input: "save/SaveV2/tensor_names"
  input: "save/SaveV2/shape_and_slices"
  input: "ModCosh/Convolution/conv_biases"
  input: "ModCosh/Convolution/conv_biases/Adam"
  input: "ModCosh/Convolution/conv_biases/Adam_1"
  input: "ModCosh/Convolution/conv_weights"
  input: "ModCosh/Convolution/conv_weights/Adam"
  input: "ModCosh/Convolution/conv_weights/Adam_1"
  input: "ModCosh/Convolution_1/conv_biases"
  input: "ModCosh/Convolution_1/conv_biases/Adam"
  input: "ModCosh/Convolution_1/conv_biases/Adam_1"
  input: "ModCosh/Convolution_1/conv_weights"
  input: "ModCosh/Convolution_1/conv_weights/Adam"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1"
  input: "ModCosh/Convolution_2/conv_biases"
  input: "ModCosh/Convolution_2/conv_biases/Adam"
  input: "ModCosh/Convolution_2/conv_biases/Adam_1"
  input: "ModCosh/Convolution_2/conv_weights"
  input: "ModCosh/Convolution_2/conv_weights/Adam"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1"
  input: "ModCosh/Convolution_3/conv_biases"
  input: "ModCosh/Convolution_3/conv_biases/Adam"
  input: "ModCosh/Convolution_3/conv_biases/Adam_1"
  input: "ModCosh/Convolution_3/conv_weights"
  input: "ModCosh/Convolution_3/conv_weights/Adam"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1"
  input: "ModCosh/Convolution_4/conv_biases"
  input: "ModCosh/Convolution_4/conv_biases/Adam"
  input: "ModCosh/Convolution_4/conv_biases/Adam_1"
  input: "ModCosh/Convolution_4/conv_weights"
  input: "ModCosh/Convolution_4/conv_weights/Adam"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1"
  input: "ModCosh/Convolution_5/conv_biases"
  input: "ModCosh/Convolution_5/conv_biases/Adam"
  input: "ModCosh/Convolution_5/conv_biases/Adam_1"
  input: "ModCosh/Convolution_5/conv_weights"
  input: "ModCosh/Convolution_5/conv_weights/Adam"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1"
  input: "ModCosh/Dense/Variable"
  input: "ModCosh/Dense/Variable/Adam"
  input: "ModCosh/Dense/Variable/Adam_1"
  input: "ModCosh/Dense/dense_weights"
  input: "ModCosh/Dense/dense_weights/Adam"
  input: "ModCosh/Dense/dense_weights/Adam_1"
  input: "ModCosh/Dense_1/Variable"
  input: "ModCosh/Dense_1/Variable/Adam"
  input: "ModCosh/Dense_1/Variable/Adam_1"
  input: "ModCosh/Dense_1/dense_weights"
  input: "ModCosh/Dense_1/dense_weights/Adam"
  input: "ModCosh/Dense_1/dense_weights/Adam_1"
  input: "ModCosh/Dense_2/Variable"
  input: "ModCosh/Dense_2/Variable/Adam"
  input: "ModCosh/Dense_2/Variable/Adam_1"
  input: "ModCosh/Dense_2/dense_weights"
  input: "ModCosh/Dense_2/dense_weights/Adam"
  input: "ModCosh/Dense_2/dense_weights/Adam_1"
  input: "Optimization/beta1_power"
  input: "Optimization/beta2_power"
  input: "Optimization/global_step"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_INT32
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/SaveV2"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/RestoreV2/tensor_names"
  op: "Const"
  device: "/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 57
          }
        }
        string_val: "ModCosh/Convolution/conv_biases"
        string_val: "ModCosh/Convolution/conv_biases/Adam"
        string_val: "ModCosh/Convolution/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution/conv_weights"
        string_val: "ModCosh/Convolution/conv_weights/Adam"
        string_val: "ModCosh/Convolution/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_1/conv_biases"
        string_val: "ModCosh/Convolution_1/conv_biases/Adam"
        string_val: "ModCosh/Convolution_1/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_1/conv_weights"
        string_val: "ModCosh/Convolution_1/conv_weights/Adam"
        string_val: "ModCosh/Convolution_1/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_2/conv_biases"
        string_val: "ModCosh/Convolution_2/conv_biases/Adam"
        string_val: "ModCosh/Convolution_2/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_2/conv_weights"
        string_val: "ModCosh/Convolution_2/conv_weights/Adam"
        string_val: "ModCosh/Convolution_2/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_3/conv_biases"
        string_val: "ModCosh/Convolution_3/conv_biases/Adam"
        string_val: "ModCosh/Convolution_3/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_3/conv_weights"
        string_val: "ModCosh/Convolution_3/conv_weights/Adam"
        string_val: "ModCosh/Convolution_3/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_4/conv_biases"
        string_val: "ModCosh/Convolution_4/conv_biases/Adam"
        string_val: "ModCosh/Convolution_4/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_4/conv_weights"
        string_val: "ModCosh/Convolution_4/conv_weights/Adam"
        string_val: "ModCosh/Convolution_4/conv_weights/Adam_1"
        string_val: "ModCosh/Convolution_5/conv_biases"
        string_val: "ModCosh/Convolution_5/conv_biases/Adam"
        string_val: "ModCosh/Convolution_5/conv_biases/Adam_1"
        string_val: "ModCosh/Convolution_5/conv_weights"
        string_val: "ModCosh/Convolution_5/conv_weights/Adam"
        string_val: "ModCosh/Convolution_5/conv_weights/Adam_1"
        string_val: "ModCosh/Dense/Variable"
        string_val: "ModCosh/Dense/Variable/Adam"
        string_val: "ModCosh/Dense/Variable/Adam_1"
        string_val: "ModCosh/Dense/dense_weights"
        string_val: "ModCosh/Dense/dense_weights/Adam"
        string_val: "ModCosh/Dense/dense_weights/Adam_1"
        string_val: "ModCosh/Dense_1/Variable"
        string_val: "ModCosh/Dense_1/Variable/Adam"
        string_val: "ModCosh/Dense_1/Variable/Adam_1"
        string_val: "ModCosh/Dense_1/dense_weights"
        string_val: "ModCosh/Dense_1/dense_weights/Adam"
        string_val: "ModCosh/Dense_1/dense_weights/Adam_1"
        string_val: "ModCosh/Dense_2/Variable"
        string_val: "ModCosh/Dense_2/Variable/Adam"
        string_val: "ModCosh/Dense_2/Variable/Adam_1"
        string_val: "ModCosh/Dense_2/dense_weights"
        string_val: "ModCosh/Dense_2/dense_weights/Adam"
        string_val: "ModCosh/Dense_2/dense_weights/Adam_1"
        string_val: "Optimization/beta1_power"
        string_val: "Optimization/beta2_power"
        string_val: "Optimization/global_step"
      }
    }
  }
}
node {
  name: "save/RestoreV2/shape_and_slices"
  op: "Const"
  device: "/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 57
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2/tensor_names"
  input: "save/RestoreV2/shape_and_slices"
  device: "/device:CPU:0"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_INT32
      }
    }
  }
}
node {
  name: "save/Assign"
  op: "Assign"
  input: "ModCosh/Convolution/conv_biases"
  input: "save/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_1"
  op: "Assign"
  input: "ModCosh/Convolution/conv_biases/Adam"
  input: "save/RestoreV2:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_2"
  op: "Assign"
  input: "ModCosh/Convolution/conv_biases/Adam_1"
  input: "save/RestoreV2:2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_3"
  op: "Assign"
  input: "ModCosh/Convolution/conv_weights"
  input: "save/RestoreV2:3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_4"
  op: "Assign"
  input: "ModCosh/Convolution/conv_weights/Adam"
  input: "save/RestoreV2:4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_5"
  op: "Assign"
  input: "ModCosh/Convolution/conv_weights/Adam_1"
  input: "save/RestoreV2:5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_6"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_biases"
  input: "save/RestoreV2:6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_7"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_biases/Adam"
  input: "save/RestoreV2:7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_8"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_biases/Adam_1"
  input: "save/RestoreV2:8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_9"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_weights"
  input: "save/RestoreV2:9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_10"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_weights/Adam"
  input: "save/RestoreV2:10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_11"
  op: "Assign"
  input: "ModCosh/Convolution_1/conv_weights/Adam_1"
  input: "save/RestoreV2:11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_1/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_12"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_biases"
  input: "save/RestoreV2:12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_13"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_biases/Adam"
  input: "save/RestoreV2:13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_14"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_biases/Adam_1"
  input: "save/RestoreV2:14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_15"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_weights"
  input: "save/RestoreV2:15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_16"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_weights/Adam"
  input: "save/RestoreV2:16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_17"
  op: "Assign"
  input: "ModCosh/Convolution_2/conv_weights/Adam_1"
  input: "save/RestoreV2:17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_2/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_18"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_biases"
  input: "save/RestoreV2:18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_19"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_biases/Adam"
  input: "save/RestoreV2:19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_20"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_biases/Adam_1"
  input: "save/RestoreV2:20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_21"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_weights"
  input: "save/RestoreV2:21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_22"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_weights/Adam"
  input: "save/RestoreV2:22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_23"
  op: "Assign"
  input: "ModCosh/Convolution_3/conv_weights/Adam_1"
  input: "save/RestoreV2:23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_3/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_24"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_biases"
  input: "save/RestoreV2:24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_25"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_biases/Adam"
  input: "save/RestoreV2:25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_26"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_biases/Adam_1"
  input: "save/RestoreV2:26"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_27"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_weights"
  input: "save/RestoreV2:27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_28"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_weights/Adam"
  input: "save/RestoreV2:28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_29"
  op: "Assign"
  input: "ModCosh/Convolution_4/conv_weights/Adam_1"
  input: "save/RestoreV2:29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_4/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_30"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_biases"
  input: "save/RestoreV2:30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_31"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_biases/Adam"
  input: "save/RestoreV2:31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_32"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_biases/Adam_1"
  input: "save/RestoreV2:32"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_33"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_weights"
  input: "save/RestoreV2:33"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_34"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_weights/Adam"
  input: "save/RestoreV2:34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_35"
  op: "Assign"
  input: "ModCosh/Convolution_5/conv_weights/Adam_1"
  input: "save/RestoreV2:35"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution_5/conv_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_36"
  op: "Assign"
  input: "ModCosh/Dense/Variable"
  input: "save/RestoreV2:36"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_37"
  op: "Assign"
  input: "ModCosh/Dense/Variable/Adam"
  input: "save/RestoreV2:37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_38"
  op: "Assign"
  input: "ModCosh/Dense/Variable/Adam_1"
  input: "save/RestoreV2:38"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_39"
  op: "Assign"
  input: "ModCosh/Dense/dense_weights"
  input: "save/RestoreV2:39"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_40"
  op: "Assign"
  input: "ModCosh/Dense/dense_weights/Adam"
  input: "save/RestoreV2:40"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_41"
  op: "Assign"
  input: "ModCosh/Dense/dense_weights/Adam_1"
  input: "save/RestoreV2:41"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_42"
  op: "Assign"
  input: "ModCosh/Dense_1/Variable"
  input: "save/RestoreV2:42"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_43"
  op: "Assign"
  input: "ModCosh/Dense_1/Variable/Adam"
  input: "save/RestoreV2:43"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_44"
  op: "Assign"
  input: "ModCosh/Dense_1/Variable/Adam_1"
  input: "save/RestoreV2:44"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_45"
  op: "Assign"
  input: "ModCosh/Dense_1/dense_weights"
  input: "save/RestoreV2:45"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_46"
  op: "Assign"
  input: "ModCosh/Dense_1/dense_weights/Adam"
  input: "save/RestoreV2:46"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_47"
  op: "Assign"
  input: "ModCosh/Dense_1/dense_weights/Adam_1"
  input: "save/RestoreV2:47"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_1/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_48"
  op: "Assign"
  input: "ModCosh/Dense_2/Variable"
  input: "save/RestoreV2:48"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_49"
  op: "Assign"
  input: "ModCosh/Dense_2/Variable/Adam"
  input: "save/RestoreV2:49"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_50"
  op: "Assign"
  input: "ModCosh/Dense_2/Variable/Adam_1"
  input: "save/RestoreV2:50"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_51"
  op: "Assign"
  input: "ModCosh/Dense_2/dense_weights"
  input: "save/RestoreV2:51"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_52"
  op: "Assign"
  input: "ModCosh/Dense_2/dense_weights/Adam"
  input: "save/RestoreV2:52"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_53"
  op: "Assign"
  input: "ModCosh/Dense_2/dense_weights/Adam_1"
  input: "save/RestoreV2:53"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Dense_2/dense_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_54"
  op: "Assign"
  input: "Optimization/beta1_power"
  input: "save/RestoreV2:54"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_55"
  op: "Assign"
  input: "Optimization/beta2_power"
  input: "save/RestoreV2:55"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ModCosh/Convolution/conv_biases"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/Assign_56"
  op: "Assign"
  input: "Optimization/global_step"
  input: "save/RestoreV2:56"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Optimization/global_step"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/Assign"
  input: "^save/Assign_1"
  input: "^save/Assign_10"
  input: "^save/Assign_11"
  input: "^save/Assign_12"
  input: "^save/Assign_13"
  input: "^save/Assign_14"
  input: "^save/Assign_15"
  input: "^save/Assign_16"
  input: "^save/Assign_17"
  input: "^save/Assign_18"
  input: "^save/Assign_19"
  input: "^save/Assign_2"
  input: "^save/Assign_20"
  input: "^save/Assign_21"
  input: "^save/Assign_22"
  input: "^save/Assign_23"
  input: "^save/Assign_24"
  input: "^save/Assign_25"
  input: "^save/Assign_26"
  input: "^save/Assign_27"
  input: "^save/Assign_28"
  input: "^save/Assign_29"
  input: "^save/Assign_3"
  input: "^save/Assign_30"
  input: "^save/Assign_31"
  input: "^save/Assign_32"
  input: "^save/Assign_33"
  input: "^save/Assign_34"
  input: "^save/Assign_35"
  input: "^save/Assign_36"
  input: "^save/Assign_37"
  input: "^save/Assign_38"
  input: "^save/Assign_39"
  input: "^save/Assign_4"
  input: "^save/Assign_40"
  input: "^save/Assign_41"
  input: "^save/Assign_42"
  input: "^save/Assign_43"
  input: "^save/Assign_44"
  input: "^save/Assign_45"
  input: "^save/Assign_46"
  input: "^save/Assign_47"
  input: "^save/Assign_48"
  input: "^save/Assign_49"
  input: "^save/Assign_5"
  input: "^save/Assign_50"
  input: "^save/Assign_51"
  input: "^save/Assign_52"
  input: "^save/Assign_53"
  input: "^save/Assign_54"
  input: "^save/Assign_55"
  input: "^save/Assign_56"
  input: "^save/Assign_6"
  input: "^save/Assign_7"
  input: "^save/Assign_8"
  input: "^save/Assign_9"
}
node {
  name: "Merge/MergeSummary"
  op: "MergeSummary"
  input: "ModCosh/Convolution/ModCosh/Convolution/conv_weights_0"
  input: "ModCosh/Convolution_1/ModCosh/Convolution_1/conv_weights_0"
  input: "ModCosh/Convolution_2/ModCosh/Convolution_2/conv_weights_0"
  input: "ModCosh/Convolution_3/ModCosh/Convolution_3/conv_weights_0"
  input: "ModCosh/Convolution_4/ModCosh/Convolution_4/conv_weights_0"
  input: "ModCosh/Convolution_5/ModCosh/Convolution_5/conv_weights_0"
  input: "ModCosh/Dense/ModCosh/Dense/dense_weights_0"
  input: "ModCosh/Dense_1/ModCosh/Dense_1/dense_weights_0"
  input: "Optimization/cost"
  attr {
    key: "N"
    value {
      i: 9
    }
  }
}
node {
  name: "init"
  op: "NoOp"
  input: "^ModCosh/Convolution/conv_biases/Adam/Assign"
  input: "^ModCosh/Convolution/conv_biases/Adam_1/Assign"
  input: "^ModCosh/Convolution/conv_biases/Assign"
  input: "^ModCosh/Convolution/conv_weights/Adam/Assign"
  input: "^ModCosh/Convolution/conv_weights/Adam_1/Assign"
  input: "^ModCosh/Convolution/conv_weights/Assign"
  input: "^ModCosh/Convolution_1/conv_biases/Adam/Assign"
  input: "^ModCosh/Convolution_1/conv_biases/Adam_1/Assign"
  input: "^ModCosh/Convolution_1/conv_biases/Assign"
  input: "^ModCosh/Convolution_1/conv_weights/Adam/Assign"
  input: "^ModCosh/Convolution_1/conv_weights/Adam_1/Assign"
  input: "^ModCosh/Convolution_1/conv_weights/Assign"
  input: "^ModCosh/Convolution_2/conv_biases/Adam/Assign"
  input: "^ModCosh/Convolution_2/conv_biases/Adam_1/Assign"
  input: "^ModCosh/Convolution_2/conv_biases/Assign"
  input: "^ModCosh/Convolution_2/conv_weights/Adam/Assign"
  input: "^ModCosh/Convolution_2/conv_weights/Adam_1/Assign"
  input: "^ModCosh/Convolution_2/conv_weights/Assign"
  input: "^ModCosh/Convolution_3/conv_biases/Adam/Assign"
  input: "^ModCosh/Convolution_3/conv_biases/Adam_1/Assign"
  input: "^ModCosh/Convolution_3/conv_biases/Assign"
  input: "^ModCosh/Convolution_3/conv_weights/Adam/Assign"
  input: "^ModCosh/Convolution_3/conv_weights/Adam_1/Assign"
  input: "^ModCosh/Convolution_3/conv_weights/Assign"
  input: "^ModCosh/Convolution_4/conv_biases/Adam/Assign"
  input: "^ModCosh/Convolution_4/conv_biases/Adam_1/Assign"
  input: "^ModCosh/Convolution_4/conv_biases/Assign"
  input: "^ModCosh/Convolution_4/conv_weights/Adam/Assign"
  input: "^ModCosh/Convolution_4/conv_weights/Adam_1/Assign"
  input: "^ModCosh/Convolution_4/conv_weights/Assign"
  input: "^ModCosh/Convolution_5/conv_biases/Adam/Assign"
  input: "^ModCosh/Convolution_5/conv_biases/Adam_1/Assign"
  input: "^ModCosh/Convolution_5/conv_biases/Assign"
  input: "^ModCosh/Convolution_5/conv_weights/Adam/Assign"
  input: "^ModCosh/Convolution_5/conv_weights/Adam_1/Assign"
  input: "^ModCosh/Convolution_5/conv_weights/Assign"
  input: "^ModCosh/Dense/Variable/Adam/Assign"
  input: "^ModCosh/Dense/Variable/Adam_1/Assign"
  input: "^ModCosh/Dense/Variable/Assign"
  input: "^ModCosh/Dense/dense_weights/Adam/Assign"
  input: "^ModCosh/Dense/dense_weights/Adam_1/Assign"
  input: "^ModCosh/Dense/dense_weights/Assign"
  input: "^ModCosh/Dense_1/Variable/Adam/Assign"
  input: "^ModCosh/Dense_1/Variable/Adam_1/Assign"
  input: "^ModCosh/Dense_1/Variable/Assign"
  input: "^ModCosh/Dense_1/dense_weights/Adam/Assign"
  input: "^ModCosh/Dense_1/dense_weights/Adam_1/Assign"
  input: "^ModCosh/Dense_1/dense_weights/Assign"
  input: "^ModCosh/Dense_2/Variable/Adam/Assign"
  input: "^ModCosh/Dense_2/Variable/Adam_1/Assign"
  input: "^ModCosh/Dense_2/Variable/Assign"
  input: "^ModCosh/Dense_2/dense_weights/Adam/Assign"
  input: "^ModCosh/Dense_2/dense_weights/Adam_1/Assign"
  input: "^ModCosh/Dense_2/dense_weights/Assign"
  input: "^Optimization/beta1_power/Assign"
  input: "^Optimization/beta2_power/Assign"
  input: "^Optimization/global_step/Assign"
}
versions {
  producer: 38
}
