#!/usr/bin/python

# @Time    : 2019/3/1 10:42
# @Author  : LaoYang
# @Email   : lhy_ustb@pku.edu.cn
# @File    : cal_panoptic.py
# @Software: PyCharm

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import time
import json
import copy
import argparse
import multiprocessing
import numpy as np
import PIL.Image as Image
from collections import defaultdict
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO
import _init_path
from panopticapi.evaluation import pq_compute
from panopticapi.converters.two_channels2panoptic_coco_format import converter

from config import ConfigVal, ConfigTest
from utils import get_traceback


# -----------------------------------------------------------------------
# config choice
# -----------------------------------------------------------------------
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',
                        dest='test',
                        action='store_true')
    parser.add_argument('-e', '--eval',
                        dest='eval',
                        action='store_true')
    parser.add_argument('-sc', '--skip_convert',
                        dest='skip_convert',
                        action='store_true')

    args = parser.parse_args()
    return args


args = parse()
cfg = ConfigTest if args.test else ConfigVal


# -----------------------------------------------------------------------
# global information
# -----------------------------------------------------------------------
with open(cfg.ins_result, 'r') as f:
    det_inses = json.load(f)
ins_res = defaultdict(list)
for ins in det_inses:
    img_id = ins['image_id']
    ins_res[img_id].append(ins)

with open(cfg.panoptic_info, 'r') as f:
    cate_info = json.load(f)['categories']
label_id2channel_id = {cate['id']: i for i, cate in enumerate(cate_info)}
stuff_channel_id2label_id = [cate['id'] for cate in cate_info if not cate['isthing']]
stuff_channel_id2label_id.append(0)

# these values can be changed via experiments.
sem_threshold = 0. # if the max stuff probability is lower than this threshold, the directly set it to void.
stuff_area_limit = 0 # filter the stuff tha the area is lower than the threshold. According to the official panopticapi, the value
                     # is set to 64 * 64

# if semantic result judges one pixel to be background, but the max stuff probability
# is larger than the threshold, then the pixel will be judged to be the stuff category.
# Otherwise it will be judged to be void. 
threshold_stuff = 0.2

# -----------------------------------------------------------------------
# overlap relations
# -----------------------------------------------------------------------
class CateSet(object):
    def __init__(self, val, lv1=None, lv2=None):
        self.val = val
        self.lv1 = lv1
        self.lv2 = lv2


table_lv2 = CateSet([31, 41, 44, 46, 48,
                     49, 50, 52, 53, 54,
                     55, 56, 57, 58, 60,
                     61, 74, 76, 84], None, None)
table_lv1 = CateSet([47, 51, 59, 73], None, table_lv2)
table = CateSet([67], table_lv1, table_lv2)

person_lv2 = CateSet([33, 46, 62, 89, 75,
                      27, 31, 77, 35, 32,
                      2, 19, 4, 28, 34,
                      36, 39, 40, 41, 42,
                      43, 90, 58, 60, 61,
                      87, 37, 38, 47, 59,
                      84], None, None)
person_lv1 = CateSet([1], None, person_lv2)
person = CateSet([15, 62, 63, 65, 6,
                  3, 8, 7, 9], person_lv1, person_lv2)


class GetChannel(object):
    def __init__(self, vec, ins_num, label):
        super(GetChannel, self).__init__()
        self.vec = vec
        self.ins_num = ins_num
        self.label = label

    def get_channel(self, cls_lv, pre_channel):
        self.vec[pre_channel] = 0
        now_channel = self.vec.argmax()
        if now_channel == self.ins_num:
            return pre_channel
        else:
            if cls_lv.lv1 is not None and self.label[now_channel] in cls_lv.lv1.val:
                now_channel = self.get_channel(cls_lv.lv1, now_channel)
            elif cls_lv.lv2 is not None and self.label[now_channel] in cls_lv.lv2.val:
                now_channel = self.get_channel(cls_lv.lv2, now_channel)
            else:
                now_channel = pre_channel
            return now_channel


# -----------------------------------------------------------------------
# two channel png generator
# -----------------------------------------------------------------------
class Generator(object):
    def __init__(self, gpu_id):
        super(Generator, self).__init__()
        # gpu_id is used to select which gpu to run the semantic model to generate the semantic prediction matrix.

    def cal_Q(self, img_name):
        # This is just an example that generates the semantic prediction matrix.
        img_path = os.path.join(cfg.img_path, img_name + '.jpg')
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        Q = np.random.rand(h, w, 54)
        return Q

    def generate_png(self, img_id, inses):
        img_name = '%012d' % img_id
        Q = self.cal_Q(img_name)
        h, w, _ = Q.shape

        ins_num = len(inses)

        png_1st_channel = np.zeros((h, w), dtype=np.int32)
        png_2nd_channel = np.zeros((h, w), dtype=np.int32)
        label_list = []

        # generate score map
        score_map = 0.01 * np.ones((h, w, ins_num + 1))
        # in this place, all instances is list by the order of category_id
        for i, ins in enumerate(inses):
            label_list.append(ins['category_id'])
            score = ins['score']
            mask = COCOmask.decode(ins['segmentation'])
            score_map[..., i] = score * mask

        ins_id_list = []
        if ins_num > 0:
            for cat_id in sorted(set(label_list)):
                ins_id_list += list(range(label_list.count(cat_id)))

        ins_id_map = score_map.argmax(axis=2)
        Q_argmax = Q.argmax(axis=2)
        Q_max = Q.max(axis=2)

        for i in range(h):
            for j in range(w):
                ins_max_id = ins_id_map[i, j]
                Q_max_id = Q_argmax[i, j]
                if ins_max_id != ins_num:
                    vec = copy.deepcopy(score_map[i, j])
                    if label_list[ins_max_id] in table.val + table_lv1.val:
                        channel_gettor = GetChannel(vec, ins_num, label_list)
                        if label_list[ins_max_id] in table.val:
                            cls = table
                        else:
                            cls = table_lv1
                        ins_max_id = channel_gettor.get_channel(cls, ins_max_id)
                    elif label_list[ins_max_id] in person.val + person_lv1.val:
                        channel_gettor = GetChannel(vec, ins_num, label_list)
                        if label_list[ins_max_id] in person.val:
                            cls = person
                        else:
                            cls = person_lv1
                        ins_max_id = channel_gettor.get_channel(cls, ins_max_id)
                    png_1st_channel[i, j] = label_list[ins_max_id]
                    png_2nd_channel[i, j] = ins_id_list[ins_max_id]
                else:
                    if Q_max_id == 53:
                        q = Q[i, j][:-1]
                        q_max_id = q.argmax()
                        q_max = q.max()
                        if q_max > threshold_stuff:
                            png_1st_channel[i, j] = stuff_channel_id2label_id[q_max_id]
                        else:
                            png_1st_channel[i, j] = 0
                    elif Q_max[i, j] < sem_threshold:
                        png_1st_channel[i, j] = 0
                    else:
                        png_1st_channel[i, j] = stuff_channel_id2label_id[Q_max_id]

        label_list = np.unique(png_1st_channel)
        for label_id in label_list:
            if label_id <= 90:
                # thing categories are passed
                continue
            selected = png_1st_channel == label_id
            area = np.sum(selected)
            if area > stuff_area_limit:
                continue
            else:
                png_1st_channel[selected] = 0
        return png_1st_channel, png_2nd_channel


# -----------------------------------------------------------------------
# multiprocess single core
# -----------------------------------------------------------------------
@get_traceback
def convert_single_core(proc_id, img_ids):
    generator = Generator(proc_id % cfg.gpu_num)
    for i, img_id in enumerate(img_ids):
        if i % 20 == 0:
            print('Core: {}, {} from {} images converted'.format(proc_id, i, len(img_ids)))
        img_name = '%012d' % img_id
        print('processing: ' + img_name + '.jpg')
        detected_inses = list()
        if img_id not in ins_res:
            print('There is no instance detected in ' + img_name)
        else:
            detected_inses = ins_res[img_id]
        detected_inses = sorted(detected_inses, key=lambda info: info['category_id'])
        png_1st_channel, png_2nd_channel = generator.generate_png(img_id, detected_inses)

        save_png(png_1st_channel, png_2nd_channel, img_name)


# -----------------------------------------------------------------------
# save two channel pngs
# -----------------------------------------------------------------------
def save_png(c1, c2, img_name):
    h, w = c1.shape
    png = np.zeros((h, w, 3), dtype=np.uint8)
    png[..., 0] = c1
    png[..., 1] = c2
    png_PIL = Image.fromarray(png)
    if not os.path.exists(cfg.two_channel_pngs):
        os.makedirs(cfg.two_channel_pngs)
    png_PIL.save(os.path.join(cfg.two_channel_pngs, img_name + '.png'))


# -----------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------
def evaluate():
    # generate panoptic pictures
    print("Convert to panoptic format!!!")
    converter(cfg.two_channel_pngs, cfg.img_info, cfg.panoptic_cate,
              cfg.generated_panoptic, cfg.panoptic_res)
    print("Finished!!!")
    # evaluate
    print("Start to evaluate!!!")
    pq_compute(cfg.panoptic_info, cfg.panoptic_res, cfg.panoptic_gt, cfg.generated_panoptic)


def main():
    img_info = COCO(cfg.img_info)
    img_ids = img_info.getImgIds()
    img_ids.sort()

    # start
    start_time = time.time()

    cpu_num = cfg.cpu_num
    img_split = np.array_split(img_ids, cpu_num)
    print('Number of cores: {}, imgs per core: {}'.format(cpu_num, len(img_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    for proc_id, sub_img_ids in enumerate(img_split):
        workers.apply_async(convert_single_core,
                            (proc_id, sub_img_ids))
    workers.close()
    workers.join()
    print('Two channel pngs have been generated!!!')
    print('Running time: {:0.2f} seconds'.format(time.time() - start_time))

    # before evaluate, we'd better check whether the 2 channel pngs are generated correctly.
    if args.eval:
        evaluate()


if __name__ == '__main__':
    if not args.skip_convert:
        main()
    else:
        evaluate()
