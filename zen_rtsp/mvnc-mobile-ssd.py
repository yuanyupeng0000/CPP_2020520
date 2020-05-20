#! /usr/bin/env python

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import numpy
import cv2
import cv
import sys
import time
import ctypes
from PIL import Image

dim=(300,300)
EXAMPLES_BASE_DIR='../../'
IMAGES_DIR = EXAMPLES_BASE_DIR + 'data/images/'
IMAGE_FULL_PATH = IMAGES_DIR + '09_59_48_5400.jpg'

#print(len(sys.argv))
#if (len(sys.argv)) < 3:
#    print("please input graph file and video file from commandline")
#    sys.exit(1)
#video_file = sys.argv[2]
#graph_file_name = sys.argv[1]
#if video_file == "":
#   print("please input video file from commandline 1st para")
#    sys.exit(1)
#if graph_file_name == "":
#    print("please input graph_file_name from commandline 2nd para")
#    sys.exit(1)  

# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************
"""
LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')
"""
"""
LABELS = ('background',
          'bus', 'car')
"""
LABELS = ('background',
          'bus', 'car', 'truck', 'bicycle', 'motorbike', 'person')

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, ssd_mobilenet_graph):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    time_start = time.time()
    resized_image = preprocess_image(image_to_classify)
    print("preprocess_image_use: %.6fs" % (time.time()-time_start))

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    time_start = time.time()
    ssd_mobilenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)
    print("send_image_to_NCS_use: %.6fs" % (time.time()-time_start))

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    time_start = time.time()
    output, userobj = ssd_mobilenet_graph.GetResult()
    print("get_result_from_NCS_use: %.6fs" % (time.time()-time_start))

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    time_start = time.time()
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))

    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
            x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
            y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            """print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')"""

            # overlay boxes and labels on the original image to classify
            overlay_on_image(image_to_classify, output[base_index:base_index + 7])
    print("post_process_use: %.6fs\n\n\n" % (time.time()-time_start))

def run_inference1(image_to_classify, ssd_mobilenet_graph):
    print "----------1-------------->"
    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    time_start = time.time()
    resized_image = preprocess_image(image_to_classify)
    print("preprocess_image_use: %.6fs" % (time.time()-time_start))

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    time_start = time.time()
    ssd_mobilenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)
    print("send_image_to_NCS_use: %.6fs" % (time.time()-time_start))

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    time_start = time.time()
    output, userobj = ssd_mobilenet_graph.GetResult()
    print("get_result_from_NCS_use: %.6fs" % (time.time()-time_start))

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0
    print "----------2-------------->"
    # number of boxes returned
    time_start = time.time()
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))
    objs=[];
    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
            x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
            y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))
            objs=objs+[x1,y1,x2,y2]

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            """print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')"""

            # overlay boxes and labels on the original image to classify
           # overlay_on_image(image_to_classify, output[base_index:base_index + 7])
    print("post_process_use: %.6fs\n\n\n" % (time.time()-time_start))
    print "----------3-------------->"
    return objs
# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def run_inference2(image_to_classify, ssd_mobilenet_graph):
    #print "----------1-------------->"
    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    #time_start = time.time()
    resized_image = preprocess_image(image_to_classify)
    #print("preprocess_image_use: %.6fs" % (time.time()-time_start))
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    #time_start = time.time()
    val=ssd_mobilenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)
    #print("send_image_to_NCS_use: %.6fs" % (time.time()-time_start))

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    #time_start = time.time()
    output, userobj = ssd_mobilenet_graph.GetResult()
    #print("get_result_from_NCS_use: %.6fs" % (time.time()-time_start))

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0
    #print "----------2-------------->"
    # number of boxes returned
    #time_start = time.time()
    num_valid_boxes = int(output[0])
    #print('total num boxes: ' + str(num_valid_boxes))
    objs=[];
    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            ClassID = int(output[base_index + 1])
            Confidence = output[base_index + 2]*100
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[1]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[0]))
            x2 = min(image_to_classify.shape[1], int(output[base_index + 5] * image_to_classify.shape[1]))
            y2 = min(image_to_classify.shape[0], int(output[base_index + 6] * image_to_classify.shape[0]))
            objs=objs+[ClassID,Confidence,x1,y1,x2 - x1,y2 - y1]

            #x1_ = str(x1)
            #y1_ = str(y1)
            #x2_ = str(x2)
            #y2_ = str(y2)
            #label_background_color = (125, 175, 75)
            #cv2.rectangle(image_to_classify, (x1, y1), (x2, y2),
            #      label_background_color, 1, 8, 0)
            """print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')"""

            # overlay boxes and labels on the original image to classify
           # overlay_on_image(image_to_classify, output[base_index:base_index + 7])
    #print("post_process_use: %.6fs\n\n\n" % (time.time()-time_start))
    #print "----------3-------------->"
    #if(frame_num > 500):
    #    out.release()
    #else:
    #    out.write(image_to_classify)
    #cv2.imwrite("1.jpg",image_to_classify);
    return objs
def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    # min_score_percent = 60
    min_score_percent =30 

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img

def preprocess_image1(src):

    # adjust values to range between -1.0 and + 1.0
    img = src - 127.5
    img = img * 0.007843
    return img
def HelloWorld():  
    print "Hello World"  
# This function is called from the entry point to do
# all the work of the program
def main111():
    # name of the opencv window
    cv_window_name = "SSD MobileNet - hit key 'q' to exit"

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    # graph_file_name = 'new_graph'

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    # read the image to run an inference on from the disk

    #####################################################
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    #cap = cv2.VideoCapture('/data/darknet/video/192.168.10.66_01_20160923193103251.mp4')
    cap = cv2.VideoCapture(video_file)


    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        loop_start = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

        # Display the resulting frame
        # cv2.imshow('Frame',frame)
        # run a single inference on the image and overwrite the
        # boxes and labels
            time_start = time.time()
            run_inference(frame, graph)
            fps = 1/(time.time()-time_start)
            print("FPS="+str(fps))

        #cv2.HoughLinesP
        #cv2.namedWindow("SSD-Mobilenet",0);
        #cv2.resizeWindow("SSD-Mobilenet", 640, 480);
        #cv2.imshow(cv_window_name, frame)

        # display the results and wait for user to hit a key
        #cv2.waitKey(0)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        loop_time = time.time() - loop_start
        print("loop_fps="+str(1/loop_time))

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    
    #####################################################
    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()

def process1(picture,width,height):
    print "processing-----------> in py"
    #objs=[];

    objs=run_inference1(picture, g_graph)
    return objs
def process2(picture, width, height):
    #print "processing-----------> in py"
    #objs=[];
    #objs=run_inference2(picture, width, height, g_graph)
    cv_img =cv.CreateImage((width, height), cv2.IPL_DEPTH_8U, 3)
    cv.SetData(cv_img, picture, 3 * width)
    cv_mat = cv_img[:]
    image_to_classify = numpy.asarray(cv_mat)
    objs=run_inference2(image_to_classify, g_graph)
    return objs
def init():
    global g_device
    global g_graph
    print "reading-----------> in py"
    gf="graph_file/iter_20000.graph"
    with open(gf, mode='rb') as f:
        graph_in_memory = f.read()
    devices = mvnc.EnumerateDevices()
    device_num = 0
    open_device_ok = 0
    if len(devices) == 0:
        print('No devices found')
        return -1
        quit()
    for device_ in devices:
        exception_accured = False
        #print(device_)
        g_device = mvnc.Device(device_)
        try:
            g_device.OpenDevice()
        except:
            exception_accured = True
            print("Cant open device {0} , seek next ...".format(device_))
            continue
        else:
            if(exception_accured):
                continue
            else:
                print("Open device {0} ok ".format(device_))
                device_num = device_num + 1
                g_graph = g_device.AllocateGraph(graph_in_memory)
                open_device_ok = 1
                break
    print "reading----1-------> in py"
    '''# Pick the first stick to run the network
    g_device = mvnc.Device(devices[0])
    print "reading-----22------> in py"
    # Open the NCS
    g_device.OpenDevice()'''
    if open_device_ok == 0:
        return -1

    print "reading----3-3------> in py"
    print "reading----44------> in py"
    # create the NCAPI graph instance from the memory buffer containing the graph file.
    #g_graph = g_device.AllocateGraph(graph_in_memory)

    print "reading#################  done---sss-------> in py"
    print "device_num" + str(len(devices))
    return device_num


def release():
    print "release start"
    g_graph.DeallocateGraph()
    print "release g_graph"
    g_device.CloseDevice()
    print "release g_device"



# main entry point for program. we'll call main() to do what needs to be done.
#if __name__ == "__main__":
#    sys.exit(main())
