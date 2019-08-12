# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs


class_names = [
        'back_ground', 'scrap','fixedshape','reflect','scratch']

classes_originID = {
    'smallscrap': 1, 'bigscrap': 2, 'fixedshape': 3, 'reflect': 4,
    'scratch': 5}




if cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'scrap': 1,
        'fixedshape': 2,
        'reflect': 3,
        'scratch': 4,
       
    }

else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict


LABEl_NAME_MAP = get_label_name_map()
