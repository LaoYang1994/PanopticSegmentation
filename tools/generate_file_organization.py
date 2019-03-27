#!/usr/bin/python
# @Author  : LaoYang
# @Email   : lhy_ustb@pku.edu.cn
# @Software: VsCode

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os

subfolder = ["annotations", "detresults", "images"]
folder = "files"           # the name must be files!!!

# file folder
for sub_folder in subfolder:
    path = os.path.join(folder, sub_folder)
    if not os.path.exists(path):
        print(path + 'is be created!')
        os.makedirs(path)

# output folder
if not os.path.exists('Outputs'):
    os.makedirs('Outputs')