# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:03:00 2017

@author: thy1995
"""

import pandas as pd


def extract_csv(filename, ranges):
    f_temp = pd.read_csv(filepath_or_buffer = filename )
    x_start, x_end, y_start, y_end = ranges
    return (f_temp.iloc[x_start : x_end +1 , y_start : y_end + 1 ]).as_matrix()

def extract_table(filename, ranges, delim = ","):
    f_temp = pd.read_table(filepath_or_buffer = filename,  delim_whitespace = True )
    x_start, x_end, y_start, y_end = ranges
    return (f_temp.iloc[x_start : x_end  , y_start : y_end  ]).as_matrix()

def extract_xls(filename, ranges):
    f_temp = pd.read_excel(filepath_or_buffer = filename )
    x_start, x_end, y_start, y_end = ranges
    return (f_temp.iloc[x_start : x_end +1 , y_start : y_end + 1 ])
