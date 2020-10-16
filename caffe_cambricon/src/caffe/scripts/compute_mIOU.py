import numpy as np
import argparse
import json
import os
from PIL import Image


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def compute_mIoU(gt_dir, pred_dir, file_list):
    """
    Compute IoU given the predicted colorized images and
    """

    full_gt_dir = os.path.join(gt_dir, '../SegmentationClass/')
    num_classes = 21
    name_classes = ["backgroud", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    hist = np.zeros((num_classes, num_classes))

    im_full_list = open(file_list, 'r').read().splitlines()
    filename_list = [os.path.splitext(os.path.split(x)[1])[0] for x in im_full_list]
    gt_imgs = [os.path.join(full_gt_dir, x + '.png') for x in filename_list]
    pred_imgs = [os.path.join(pred_dir, x + '.png') for x in filename_list]

    for ind in range(len(gt_imgs)):
        label_org = Image.open(gt_imgs[ind])
        pred_rgb = Image.open(pred_imgs[ind]).resize(label_org.size)
        pred_p = pred_rgb.quantize(palette = label_org) # convert to P mode
        label = np.array(label_org)
        pred = np.array(pred_p)

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 100 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('mIOU:' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.file_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='directory which stores VOC Segmentation val gt images')
    parser.add_argument('--pred_dir', type=str, help='directory which stores VOC Segmentation val pred images')
    parser.add_argument('--file_list', type=str, help='file list which uses to compute mIOU')
    args = parser.parse_args()
    main(args)
