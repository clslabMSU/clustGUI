# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:08:52 2017

@author: thy1995
"""
import numpy as np
def table_result (result, x_axis, y_axis):
    result = np.concatenate((x_axis, result), axis = 0)
    result = np.concatenate((result, np.transpose(y_axis)), axis = 1)
    return result