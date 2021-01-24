import tensorflow as tf
import os
model_dir = "model/"
checkpoint_path = os.path.join(model_dir, "model.ckpt-99000.ckpt")
reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    f = open('temp.txt', 'a')
    tensor_name = "\n tensor_name:" + key
    f.write(tensor_name)
    f.write(str(reader.get_tensor(key)))
    
