# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:35:26 2020

@author: T530
"""

import h5py
import os
import argparse

def main(args):
    directory = os.fsencode(args.dataset)
    data_size = 0
    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         if filename.endswith(".hdf5") and "training" in filename: 
             print(filename)
             f = h5py.File(os.path.join(args.dataset, filename), 'r')
             data_size += len(f["input_ids"])
             print(type(f["input_ids"]))
             continue
         else:
             continue
    
    print(data_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--dataset",
                        default="wiki_70k",
                        type=str,
                        required=False,
                        help="Specify a input filename!")
    args = parser.parse_args()
    main(args)