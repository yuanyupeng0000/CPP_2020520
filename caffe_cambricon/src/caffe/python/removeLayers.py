from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import sys
import os

# the 0 to i-1 layer will be write
def write_pt(layers, id, outputdir):
    i = 0
    name = layers[id - 1].name
    name = outputdir + name + ".pt"
    print "dumping: " + name
    with open(name, "w") as f:
        for layer in layers:
            if(i < id):
                i+=1
                f.write("layer {\n")
                f.write("\n".join(["  " + line for line in str(layer).split("\n") if line != ""]))
                f.write("\n")
                f.write("}\n\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: removeLayers.py: input_pt layer_id outputdir")
        exit(1)
    net = caffe_pb2.NetParameter()
    filename = sys.argv[1]
    comment_id = int(sys.argv[2])
    outputdir = sys.argv[3]
    with open(filename) as f:
        s = f.read()
        txtf.Merge(s, net)
    layers = net.layer

    print("net layers: ", len(layers))
    #comment_id=input("input a layer to comment: ")

    if(comment_id > 0):
        write_pt(layers, comment_id, outputdir)
    else:
        for i in range(2, len(layers) + 1):
            write_pt(layers, i, outputdir)
