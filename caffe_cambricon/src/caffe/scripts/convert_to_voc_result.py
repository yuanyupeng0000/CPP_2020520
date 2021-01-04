import os
import logging
import argparse
import xml.etree.ElementTree as ET

logging.basicConfig(format='%(asctime)s, %(levelname)s: %(message)s', level=logging.INFO)

def parseresult(annopath_t, result_file_path, input_type, score_threshold=0.00):
    """ Parse a result file like 000001.txt """

    def parse_rec(filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('size'):
            width = obj.find('width').text
            height = obj.find('height').text
        return float(width), float(height)

    def autotune(result, annopath, input_type):
        """ normalized position to absoluted position """
        if input_type == 'normalized':
            new_result = {}
            width, height = parse_rec(annopath)
            # width -= 1
            # height -= 1
            for k, v in result.items():
                if len(v) == 0:
                    continue
                new_result[k] = []
                for line in v:
                    new_result[k].append([line[0],
                                          line[1],
                                          line[2] * width,
                                          line[3] * height,
                                          line[4] * width,
                                          line[5] * height])
            return new_result
        else:
            return result

    result = {}
    dirname, filename = os.path.split(result_file_path)
    fname, extension = os.path.splitext(filename)
    lines = [line.strip().split() for line in open(result_file_path, "r").readlines()]
    for splitline in lines:
        predict_id = splitline[0]
        score = float(splitline[1])
        bb = [float(x) for x in splitline[2:]]
        if score <= score_threshold:
            continue
        if not predict_id in result:
            result[predict_id] = []
        result[predict_id].append([fname, score] + bb)
    ret = autotune(result, annopath_t.format(fname), input_type)
    return ret

def dump2file(all_boxes, output_dir, output_filename_t):
    for k, v in all_boxes.items():
        if len(v) == 0:
            continue
        with open(os.path.join(output_dir, output_filename_t.format(k)), "w") as f:
            for line in v:
                f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                    line[0], line[1],
                    line[2], line[3], line[4], line[5]))
    return True

def convert(to_convert_result_dir, filename_extension, input_type,
            devkit_path, year, comp_id, image_set,
            output_dir, allow_missing):

    def insert(all_boxes, result):
        for k, v in result.items():
            if not k in all_boxes:
                all_boxes[k] = []
            all_boxes[k].extend(v)
        return all_boxes

    logger = logging.getLogger()
    all_boxes = {}
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        image_set + '.txt')
    annopath_t = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
            '{:s}.xml')
    output_filename_t = comp_id + '_det_' + image_set + '_{:s}.txt'
    logger.info("imagesetfile: {}".format(imagesetfile))
    logger.info("annopath_template: {}".format(annopath_t))
    logger.info("output_filename_template: {}".format(
        os.path.join(output_dir, output_filename_t)))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_list = [line.strip() for line in open(imagesetfile, "r").readlines()]
    for idx, filename_ in enumerate(file_list):
        if idx % 100 == 0:
            logger.info("{:5d}/{:5d}".format(idx, len(file_list)))
        filepath = os.path.join(to_convert_result_dir, filename_ + filename_extension)
        if not os.path.isfile(filepath):
            if allow_missing:
                logger.warning("Invalid file {}".format(filepath))
            else:
                raise ValueError("Invalid file {}".format(filepath))
        else:
            result = parseresult(
                    annopath_t, filepath, input_type)
            all_boxes = insert(all_boxes, result)
    dump2file(all_boxes, output_dir, output_filename_t)

def parse_args():
    parser = argparse.ArgumentParser(description="convert result")
    parser.add_argument("--input-dir", dest="input_dir",
            help="path of the directory of the result", type=str, required=True)
    parser.add_argument("--filename-extension", dest="filename_extension",
            help="filename extention of the result files", type=str, default=".txt")
    parser.add_argument("--input-type", dest="input_type",
            help="input data position type, absoluted or normalized", type=str,
            choices=['normalized', 'absoluted'], required=True)
    parser.add_argument("--allow-missing", dest="allow_missing",
            help="whether allow missing", action='store_true', default=False)
    parser.add_argument("--devkit-path", dest="devkit_path",
            help="path of the directory of the devkit", type=str, required=True)
    parser.add_argument("--year", dest="year",
            help="which year VOC", type=str, default="2007")
    parser.add_argument("--comp-id", dest="comp_id",
            help="which competition", type=str, default="comp4")
    parser.add_argument("--image-set", dest="image_set",
            help="which image_set, train, val, test", type=str, default="test")
    parser.add_argument("--output-dir", dest="output_dir",
            help="path of the directory of the converted result", type=str, default="./result")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    convert(args.input_dir, args.filename_extension, args.input_type,
            args.devkit_path, args.year,
            args.comp_id, args.image_set,
            args.output_dir, args.allow_missing)
