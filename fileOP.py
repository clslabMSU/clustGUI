# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:51:02 2017

@author: thy1995
"""
import csv

def writeRows(result_file_name, rows):
    with open(result_file_name, 'w', newline='', encoding='utf-8') as text_file:
        csv_file= csv.writer(text_file)
        csv_file.writerows(rows)
        
def new_names(label_names, postfix):
    output_names = []
    for i in label_names:
        new_name = i.split('.')[0] + postfix + ".csv"
        output_names.append(new_name)
    return output_names
    
def new_name(label_name, postfix):
    return "/".join((label_name.split('.')[0]).split("/")[:-1]) + "/" + postfix + ".csv"
