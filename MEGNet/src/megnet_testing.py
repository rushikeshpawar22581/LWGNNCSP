# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:05:27 2023

@author: ashro
"""


import os
import numpy as np
import tensorflow as tf
from megnet.models import MEGNetModel, GraphModel
from megnet.data.crystal import CrystalGraph
from data_preparation import MatBenchDataset
from tqdm import tqdm
from tensorflow.keras.models import load_model
from megnet.layers import _CUSTOM_OBJECTS


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

model_path = './models/matbenchallcifs_unperturbed_testremoved_150epochs'
if os.path.isdir(model_path):
    model_files = os.listdir(model_path)
    model_files = [mf for mf in model_files if '.hdf5' in mf]
    #model_files.sort(key=lambda fn: os.path.getmtime(model_path + '/' + fn))
    val_errors = [float(x.split('_')[-1][:-5]) for x in model_files]
    min_val = min(val_errors)
    print(min_val)
    idx = val_errors.index(min_val)
    model_file_path = os.path.join(model_path, model_files[idx])
elif os.path.isfile(model_path):
    model_file_path = model_path
else:
    print('Input a GN model path!')
    
nfeat_bond = 100
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

print(model_file_path)

model = load_model(model_file_path, custom_objects = _CUSTOM_OBJECTS)

gn_model = GraphModel(model=model,
                  graph_converter=graph_converter,
                  centers=gaussian_centers,
                  width=gaussian_width)


dataset = MatBenchDataset(n_train_rate = 0.70, n_val_rate = 0.15, is_shuffle = True,
                          rand_seed = 100)
test_structures, test_targets = dataset.test_set
# pred_targets = model.predict_structures(test_targets).reshape(-1)
print('Data loading is complete')
ae = []
for i in tqdm(range(len(test_targets))):
    try:
        pred_target = gn_model.predict_structure(test_structures[i]).reshape(-1)[0]
        true_target = test_targets[i]
        error = pred_target - true_target
        ae.append(abs(error))
    except:
        continue

mae = np.sum(ae) / len(ae)
print('Test MAE:', mae)
