# coding: utf-8
import numpy as np
import os
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2
import sys
import time

def match(resultsfile, groundtruthfile, show_images):
    results, num_detectedbox = load(resultsfile)
    groundtruth, num_groundtruthbox = load(groundtruthfile)

    assert len(results) == len(groundtruth), "The number of results does not match" % (
    len(groundtruth), len(results))

    maxiou_confidence = np.array([])

    for i in range(len(results)):

        #print(results[i][0])

        if show_images:
            fname = './' + results[i][0] + '.jpg'
            image = cv2.imread(fname)

        for j in range(2, len(results[i])):

            iou_array = np.array([])
            detectedbox = results[i][j]
            confidence = detectedbox[-1]

            if show_images:
                x_min, y_min = int(detectedbox[0]), int(detectedbox[1])
                x_max = int(detectedbox[0] + detectedbox[2])
                y_max = int(detectedbox[1] + detectedbox[3])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            for k in range(2, len(groundtruth[i])):
                groundtruthbox = groundtruth[i][k]
                iou = cal_IoU(detectedbox, groundtruthbox)
                iou_array = np.append(iou_array, iou)

                if show_images:
                    x_min, y_min = int(groundtruthbox[0]), int(groundtruthbox[1])
                    x_max = int(groundtruthbox[0] + groundtruthbox[2])
                    y_max = int(groundtruthbox[1] + groundtruthbox[3])
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            maxiou = np.max(iou_array)
            maxiou_confidence = np.append(maxiou_confidence, [maxiou, confidence])

        if show_images:
            cv2.imshow("Image",image)
            cv2.waitKey()

    maxiou_confidence = maxiou_confidence.reshape(-1, 2)
    maxiou_confidence = maxiou_confidence[np.argsort(-maxiou_confidence[:, 1])]

    return maxiou_confidence, num_detectedbox, num_groundtruthbox


def thres(maxiou_confidence, threshold = 0.5):
    maxious = maxiou_confidence[:, 0]
    confidences = maxiou_confidence[:, 1]
    true_or_flase = (maxious > threshold)
    tf_confidence = np.array([true_or_flase, confidences])
    tf_confidence = tf_confidence.T
    tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
    return tf_confidence


def plot(tf_confidence, num_groundtruthbox,filename):
    fp_list = []
    recall_list = []
    precision_list = []
    auc = 0
    mAP = 0
    for num in range(len(tf_confidence)):
        arr = tf_confidence[:(num + 1), 0]
        tp = np.sum(arr)
        fp = np.sum(arr == 0)
        recall = tp / num_groundtruthbox
        precision = tp / (tp + fp)
        auc = auc + recall
        mAP = mAP + precision

        fp_list.append(fp)
        recall_list.append(recall)
        precision_list.append(precision)

    auc = auc / len(fp_list)
    mAP = mAP * max(recall_list) / len(recall_list)

    plt.figure()
    plt.title('ROC')
    plt.xlabel('False Positives')
    plt.ylabel('True Positive rate')
    plt.ylim(0, 1)
    plt.plot(fp_list, recall_list, label = 'AUC: ' + str(auc))
    plt.legend()

    plt.figure()
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.plot(recall_list, precision_list, label = 'mAP: ' + str(mAP))
    plt.legend()
    plt.savefig(filename)
    #plt.show()
    return mAP

def ellipse_to_rect(ellipse):
    major_axis_radius, minor_axis_radius, angle, center_x, center_y, score = ellipse
    leftx = center_x - minor_axis_radius
    topy = center_y - major_axis_radius
    width = 2 * minor_axis_radius
    height = 2 * major_axis_radius
    rect = [leftx, topy, width, height, score]
    return rect


def cal_IoU(detectedbox, groundtruthbox):
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det + width_gt * height_gt - intersection
        iou = intersection / union
        #print(iou)
        return iou
    else:
        return 0


def load(txtfile):
    imagelist = []

    txtfile = open(txtfile, 'r')
    lines = txtfile.readlines()

    num_allboxes = 0
    i = 0
    while i < len(lines):
        image = []
        image.append(lines[i].strip())
        num_faces = int(lines[i + 1])
        num_allboxes = num_allboxes + num_faces
        image.append(num_faces)

        if num_faces > 0:
            for num in range(num_faces):
                boundingbox = lines[i + 2 + num].strip()
                boundingbox = boundingbox.split()
                boundingbox = list(map(float, boundingbox))

                if len(boundingbox) == 6:
                    boundingbox = ellipse_to_rect(boundingbox)

                image.append(boundingbox)

        imagelist.append(image)

        i = i + num_faces + 2

    txtfile.close()

    return imagelist, num_allboxes

def draw_curves(resultsfile, groundtruthfile, filename, show_images = False, threshold = 0.5):
  maxiou_confidence, num_detectedbox, num_groundtruthbox = match(resultsfile, groundtruthfile, show_images)
  tf_confidence = thres(maxiou_confidence, threshold)
  return plot(tf_confidence, num_groundtruthbox,filename)

def update_json_meanAp(json_data, meanAp, key):
    if isinstance(json_data, dict):
        for k in json_data:
            if k == key:
               json_data[k] = round(meanAp, 2)
            elif isinstance(json_data[k], dict):
               update_json_meanAp(json_data[k], meanAp, key)

if __name__ == '__main__':
  if len(sys.argv)!=4:
    print("Usage:{} detection_output ellipseList.txt jpg".format(sys.argv[0]))
    sys.exit()
  else:
    mAP=draw_curves(sys.argv[1],sys.argv[2],sys.argv[3])
    summary="mAP: {}".format(mAP)
    input_json_file = os.getenv('OUTPUT_JSON_FILE','')
    if os.path.isfile(input_json_file):
        file_in = open(input_json_file, "r")
        json_data = json.load(file_in)
        update_json_meanAp(json_data, mAP * 100, 'meanAp')
        file_in.close()
        file_out = open(input_json_file, "w")
        json.dump(json_data, file_out,indent=2)
        file_out.close()
    print(summary)
