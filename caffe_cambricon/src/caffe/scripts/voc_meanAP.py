from __future__ import print_function
import os
import logging
import json
import argparse
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from eval_voc import voc_eval
except ImportError:
    from voc_eval import voc_eval

logging.basicConfig(format='%(asctime)s, %(levelname)s: %(message)s', level=logging.INFO)

def save_PR_curves(result, year, imageset, save_name, draw_11_point=False):
    def pr_curve_11_point(rec_, prec_):
        ''' calculate precision-recall curve '''
        rec = np.array(rec_)
        prec = np.array(prec_)
        recall = []
        precision = []
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            recall.append(t)
            precision.append(p)
            ap += p / 11.
        return recall, precision, ap

    def random_color():
        import random
        opt = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'orange', 'brown', 'fuchsia', 'lime', 'tomato']
        return opt[random.randint(0, len(opt)-1)]

    figure, ax = plt.subplots(4, 5)
    figure.set_size_inches(18, 15)
    APs = []
    for k, name in list(enumerate(result)):
        i = k / 5
        j = k % 5
        APs.append(result[name]['ap'])
        ax[i][j].plot(result[name]['rec'], result[name]['prec'],
                color=random_color())
        if int(year) < 2010 and draw_11_point:
            r, p, _ = pr_curve_11_point(result[name]['rec'], result[name]['prec'])
            # ax[i][j].plot(
            ax[i][j].scatter(
                    r, p,
                    marker="x", # linewidth=1, linestyle="-.",
                    color=random_color())
        ax[i][j].set_title("{:s}  AP={:s}".format(name, str(round(APs[-1], 4))), fontsize=15)
        ax[i][j].set_xlabel("recall")
        ax[i][j].set_ylabel("precision")
        ax[i][j].set_xlim(-0.1, 1.1)
        ax[i][j].set_ylim(-0.1, 1.1)
        ax[i][j].grid(True)
    mAP = sum(APs) / len(APs)
    title = "VOC{:s} {:s}, mAP={:s}".format(
        year, imageset, str(round(mAP,4)))
    if int(year) < 2010:
        title = title + "   11-point interpolated average precision"
    plt.suptitle("{:s}".format(title), y=0.97, fontsize=20)
    figure.tight_layout(pad=3)
    figure.subplots_adjust(top=0.92)
    plt.savefig(save_name, dpi=120)

#
# official format result should save like this path
# VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
#
# but we just give input_dir rather than
# generate voc_results_file path by using VOCdevkit path
#

def do_python_eval(input_dir, devkit_path, year, comp_id, image_set, output_dir):
    def get_voc_results_file_template(input_dir, devkit_path, year, comp_id, image_set):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = comp_id + '_det_' + image_set + '_{:s}.txt'
        if input_dir != "":
            path = os.path.join(
                    input_dir,
                    filename)
        else:
            path = os.path.join(
                    devkit_path,
                    'results',
                    'VOC' + year,
                    'Main',
                    filename)
        return path


    logger = logging.getLogger()
    classes = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
    annopath_t = os.path.join(
            devkit_path,
            'VOC' + year,
            'Annotations',
            '{:s}.xml')
    imagesetfile = os.path.join(
            devkit_path,
            'VOC' + year,
            'ImageSets',
            'Main',
            image_set + '.txt')

    cachedir = os.path.join('./', 'annotations_cache')
    aps = []
    pr_full = collections.OrderedDict()
    pr_11point = collections.OrderedDict()
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        detpath_t = get_voc_results_file_template(
                input_dir, devkit_path, year, comp_id, image_set).format(cls)
        rec, prec, ap = voc_eval(
                detpath_t, annopath_t, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls, ap))
        pr_message = {'rec': rec, 'prec': prec, 'ap': ap}
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
            pickle.dump(pr_message, f)
        pr_full[cls] = {'rec': list(rec.flatten()), 'prec': list(prec.flatten()), 'ap': ap}
    logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for ap in aps:
        logger.info('{:.3f}'.format(ap))
    logger.info('{:.3f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('--------------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB eval code.')
    logger.info('-- Thanks, The Management')
    logger.info('--------------------------------------------------------------')
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    if logger.isEnabledFor(logging.DEBUG):
        json_name = os.path.join(output_dir, 'PR_curve.json')
        photo_name = os.path.join(output_dir, 'PR_curve.jpg')
        with open(json_name, 'w') as json_file:
            json.dump(pr_full, json_file, sort_keys = False, indent = 2)
        logger.debug("Save PR curves in {:s}".format(json_name))
        if len(pr_full) == 20:
            save_PR_curves(pr_full, year, image_set, photo_name)
            logger.debug("Save PR curves in {:s}".format(photo_name))
            if int(year) < 2010:
                photo_name = os.path.join(output_dir, 'PR_curve_and_11_points.jpg')
                save_PR_curves(pr_full, year, image_set, photo_name, True)
                logger.debug("Save PR curves in {:s}".format(photo_name))

def parse_args():
    parser = argparse.ArgumentParser(description="convert result")
    parser.add_argument("--input-dir", dest="input_dir",
            help="path of the directory of the result", type=str, required=True)
    parser.add_argument("--devkit-path", dest="devkit_path",
            help="path of the directory of the devkit", type=str, required=True)
    parser.add_argument("--year", dest="year",
            help="which year VOC", type=str, default="2007")
    parser.add_argument("--comp-id", dest="comp_id",
            help="which competition", type=str, default="comp4")
    parser.add_argument("--image-set", dest="image_set",
            help="which image_set, train, val, test", type=str, default="test")
    parser.add_argument("--output-dir", dest="output_dir",
            help="path of the directory of the converted result", type=str, default="./mAP_result")
    parser.add_argument("--debug", dest="debug",
            help="whether show debug message", action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
    do_python_eval(args.input_dir,
            args.devkit_path, args.year,
            args.comp_id, args.image_set,
            args.output_dir)
