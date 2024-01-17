# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:52:37 2022

@author: ashro
"""
import os
import numpy as np
import tensorflow as tf
from megnet.models import MEGNetModel, GraphModel
from megnet.data.crystal import CrystalGraph
#from data_preparation import MatBenchDataset
from tqdm import tqdm
from tensorflow.keras.models import load_model
from megnet.layers import _CUSTOM_OBJECTS
from pymatgen.io.cif import CifParser


def predict_energy(model, cifdir):
    # Uncomment the following line if you want to run with a specific model and not the best validation model.
    #model_path = os.path.join(model_path, 'val_mae_00372_0.032587.hdf5')
    # if os.path.isdir(model_path):
    #     model_files = os.listdir(model_path)
    #     model_files = [mf for mf in model_files if '.hdf5' in mf]
    #     #model_files.sort(key=lambda fn: os.path.getmtime(model_path + '/' + fn))
    #     val_errors = [float(x.split('_')[-1][:-5]) for x in model_files]
    #     min_val = min(val_errors)
    #     print(min_val)
    #     idx = val_errors.index(min_val)
    #     model_file_path = os.path.join(model_path, model_files[idx])
    # elif os.path.isfile(model_path):
    #     model_file_path = model_path
    # else:
    #     print('Input a GN model path!')
        
    # nfeat_bond = 100
    # r_cutoff = 5
    # gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    # gaussian_width = 0.5
    # graph_converter = CrystalGraph(cutoff=r_cutoff)

    # print(model_file_path)

    # model = load_model(model_file_path, custom_objects = _CUSTOM_OBJECTS)

    # gn_model = GraphModel(model=model,
    #                   graph_converter=graph_converter,
    #                   centers=gaussian_centers,
    #                   width=gaussian_width)


    cif_files = [x for x in os.listdir(cifdir) if x[-4:] == '.cif']
    #assert len(cif_files) == 1
    parser = CifParser(cifdir + '/' + cif_files[0])
    test_structure = parser.get_structures(primitive=False)[0]

    
    try:
        pred_target = model.predict_structure(test_structure).reshape(-1)[0]
        return pred_target
    except:
        return 999


if __name__=='__main__':
    modelpath = './model_best_trainmpdata75-230_testmpdata75-230.pth.tar'
    cifdir = '../Data/One_Item/'
    predict_energy(modelpath, cifdir)
    
    
