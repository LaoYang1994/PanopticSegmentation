#!/usr/bin/env python2
'''
This script converts panoptic COCO format to detection COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data. All segments will be stored in RLE format.

Additional option:
- using option '--things_only' the script can discard all stuff
segments, saving segments of things classes only.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing

import PIL.Image as Image

from utils import get_traceback, rgb2id

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
except:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

@get_traceback
def convert_panoptic_to_detection_coco_format_single_core(
    proc_id, annotations_set, categories, segmentations_folder, things_only
):
    annotations_detection = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(annotations_set)))

        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])
        try:
            pan_format = np.array(
                Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32
            )
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['image_id']))
        pan = rgb2id(pan_format)

        for segm_info in annotation['segments_info']:
            if things_only and categories[segm_info['category_id']]['isthing'] != 1:
                continue
            mask = (pan == segm_info['id']).astype(np.uint8)
            mask = np.expand_dims(mask, axis=2)
            segm_info.pop('id')
            segm_info['image_id'] = annotation['image_id']
            segm_info['segmentation'] = COCOmask.encode(np.asfortranarray(mask))[0]
            annotations_detection.append(segm_info)

    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))
    return annotations_detection


def convert_panoptic_to_detection_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              categories_json_file,
                                              things_only):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = input_json_file.rsplit('.', 1)[0]

    print("CONVERTING...")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO detection format")
    print("\tJSON file: {}".format(output_json_file))
    if things_only:
        print("Saving only segments of things classes.")
    print('\n')

    print("Reading annotation information from {}".format(input_json_file))
    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(convert_panoptic_to_detection_coco_format_single_core,
                                (proc_id, annotations_set, categories, segmentations_folder, things_only))
        processes.append(p)
    annotations_coco_detection = []
    for p in processes:
        annotations_coco_detection.extend(p.get())
    for idx, ann in enumerate(annotations_coco_detection):
        ann['id'] = idx

    d_coco['annotations'] = annotations_coco_detection
    categories_coco_detection = []
    for category in d_coco['categories']:
        if things_only and category['isthing'] != 1:
            continue
        category.pop('isthing')
        category.pop('color')
        categories_coco_detection.append(category)
    d_coco['categories'] = categories_coco_detection
    with open(output_json_file, 'w') as f:
        json.dump(d_coco, f)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script converts panoptic COCO format to detection \
         COCO format. See this file's head for more information."
    )
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with panoptic COCO format")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if input_json_file is \
         X.json"
    )
    parser.add_argument('--output_json_file', type=str,
                        help="JSON file with detection COCO format")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--things_only', action='store_true',
                        help="discard stuff classes")
    args = parser.parse_args()
    convert_panoptic_to_detection_coco_format(args.input_json_file,
                                              args.segmentations_folder,
                                              args.output_json_file,
                                              args.categories_json_file,
                                              args.things_only)
