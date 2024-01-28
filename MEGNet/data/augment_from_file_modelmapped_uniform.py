# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:47:30 2022

@author: ashro
"""

import os
import shutil
from pymatgen.ext.matproj import MPRester
import csv
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import random
import pickle as pkl
from sklearn.mixture import GaussianMixture
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from predict_energy_batch import predict_energy

np.random.seed(19)
random.seed(19)



def write_structs(low, high, scale, num_per, input_dir, output_dir, modelpath):
    """Write original and perturbed structure to a directory
    Parameters
    ----------
    results: List or np.ndarray
      query results from materials project
    dir_name: String
      Name of directory to write cif files to
    Returns
    -------
    E: List
      list in format to write id_prop.csv file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cif_files = [x for x in os.listdir(input_dir) if x[-4:] == '.cif']

    for file in cif_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))
        
    if os.path.exists('./temp_data/'):
        shutil.rmtree('./temp_data/')
    
    os.makedirs('./temp_data/')

    shutil.copy('./matbench_allcifs_unperturbed_testremoved/atom_init.json', './temp_data/atom_init.json')
    shutil.copy('./temp_data/atom_init.json', os.path.join(output_dir, 'atom_init.json'))

    perturbed_cif_files = [x for x in cif_files if 'per' in x]
    print(len(cif_files))
    id_prop = pd.read_csv(os.path.join(input_dir, 'id_prop.csv'), header = None)
    id_prop.columns = ['struct_idx', 'form_energy']
    
    original_mask = []
    for idx in id_prop['struct_idx']:
        if 'per' in idx:
            original_mask.append(False)
        else:
            original_mask.append(True)

    id_prop_original = id_prop[original_mask]

    for file in perturbed_cif_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join('./temp_data/', file))

    df_perturbed = predict_energy(modelpath, './temp_data/')

    #print(np.mean(df.iloc[:, 1]))
    complete_df = pd.concat([id_prop_original, df_perturbed], axis = 0, ignore_index = True)
    print(complete_df.shape)
    complete_df.to_csv(os.path.join(output_dir, 'id_prop.csv'), header = False, index = False)
    




if __name__ == "__main__":
    low = 0.0
    high = 0.3
    scale_per = 1 #Scaling the perturbation distance
    input_dir = 'matbench_allcifs_unperturbed_testremoved'
    num_per = 1   
    intermediate_dir = input_dir + '_per{}'.format(num_per) + '_scale{}'.format(scale_per) + '_uniform_low{}_high{}'.format(low, high)
    output_dir = intermediate_dir + 'modelmapped' + 'bybestM1'
    modelpath = '../models/model_best_train132krelaxedtrain0.7val0.15test0.15epochs1000batch128patience500trial2/'
    E = write_structs(low, high, scale_per, num_per, intermediate_dir, output_dir, modelpath)
    
