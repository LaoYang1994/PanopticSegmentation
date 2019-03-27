#!/usr/bin/python

# @Time    : 2019/3/1 20:53
# @Author  : LaoYang
# @Email   : lhy_ustb@pku.edu.cn
# @File    : config.py
# @Software: PyCharm

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp

_ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

class ConfigBase(object):
    def __init__(self):
        super(ConfigBase, self).__init__()

    base_path = osp.join(_ROOT_DIR, 'files')
    panoptic_info = osp.join(base_path, 'annotations', 'panoptic_sample.json')
    panoptic_cate = osp.join(base_path, 'annotations', 'panoptic_coco_categories.json')
    panoptic_gt = osp.join(base_path, 'annotations', 'panoptic_val2017')
    generated_panoptic = osp.join('Outputs', 'panoptic_results')
    gpu_num = 8
    cpu_num = 4


class ConfigVal(ConfigBase):
    def __init__(self):
        super(ConfigVal, self).__init__()

    base_path = ConfigBase.base_path
    img_path = osp.join(base_path, 'images', 'val2017')
    # img_info = osp.join(base_path, 'annotations', 'instances_val2017.json')
    img_info = osp.join(base_path, 'annotations', 'instances_sample.json')  # annotations for example: instances_val2017.json
    ins_result = osp.join(base_path, 'detresults', 'det_res.json')          # detection result
    two_channel_pngs = osp.join('Outputs', 'val_two_channel_pngs')
    panoptic_res = osp.join('Outputs', 'val_panoptic_result.json')


class ConfigTest(ConfigBase):
    def __init__(self):
        super(ConfigTest, self).__init__()

    base_path = ConfigBase.base_path
    img_path = osp.join(base_path, 'images', 'test2017')
    img_info = osp.join(base_path, 'annotations', 'image_info_test-dev2017.json')
    ins_result = osp.join(base_path, 'annotations', 'test-dev.json')
    two_channel_pngs = osp.join('Outputs', 'test_two_channel_pngs')
    panoptic_res = osp.join('Outputs', 'test_panoptic_result.json')


class ConfigModel(object):
    all_cls_num = 54


def main():
    cfg = ConfigVal()
    print(cfg.img_path)


if __name__ == '__main__':
    main()

