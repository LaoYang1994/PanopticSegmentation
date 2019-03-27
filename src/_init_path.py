#!/usr/bin/python
# @Author  : LaoYang
# @Email   : lhy_ustb@pku.edu.cn
# @Software: VsCode

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'tools')
add_path(lib_path)