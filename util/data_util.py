#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_util.py    
@Contact :   liu.8948@osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/6/19 3:59 PM   Haohe Liu      1.0         None
'''

import pickle
import json
import logging

def save_pickle(obj,fname):
    print("Save pickle at "+fname)
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(fname):
    print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

def write_json(dict,fname):
    print("Save json file at"+fname)
    json_str = json.dumps(dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data