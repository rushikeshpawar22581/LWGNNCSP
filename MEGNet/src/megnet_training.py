# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:05:05 2023

@author: ashro
"""

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
from megnet.models import MEGNetModel, GraphModel
from megnet.data.crystal import CrystalGraph
from data_preparation import MatBenchDataset
from tensorflow.keras.models import load_model
from megnet.layers import _CUSTOM_OBJECTS
from tqdm import tqdm

def args_parse():
    parser = argparse.ArgumentParser(description='MEGNet')
    parser.add_argument('--data-path', type=str, default='../data/', help='Data Path')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',help='number of total epochs to run (default: 30)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--patience', default=500, type=int, metavar='N',help='Patience For Early Stopping')
    parser.add_argument('--train-ratio', default=0.70, type=float, metavar='N',help='Training Data Ratio')
    parser.add_argument('--val-ratio', default=0.15, type=float, metavar='N',help='Validation Data Ratio')
    parser.add_argument('--gpu-name', default=0, type=int, metavar='N',help='Which GPU to use for running program')
    parser.add_argument('--trial-num', default=None, type=int, help='Trial Number To Identify Multiple Runs With Same Hyperparameters')
    args = parser.parse_args(sys.argv[1:])
    return args
    
global args
args = args_parse()

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_visible_devices(physical_devices[args.gpu_name], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[args.gpu_name], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

nfeat_bond = 100
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, 
                    width=gaussian_width, metrics = ['mae'], loss = 'mse')



#Reading Data
# dataset = MatBenchDataset(n_train_rate = 0.70, n_val_rate = 0.15, is_shuffle = True, rand_seed = 100)
path_to_data = args.data_path
dataset = MatBenchDataset(path_to_data, n_train_rate = args.train_ratio, n_val_rate = args.val_ratio, is_shuffle = True, rand_seed = 100)


training_structures, training_targets = dataset.training_set
val_structures, val_targets = dataset.val_set
test_structures, test_targets = dataset.test_set

print(test_targets[0])

del dataset

epochs = args.epochs
batch_size = args.batch_size
patience = args.patience
trial = args.trial_num
model_dir = '../models/model_best_train132krelaxedtrain0.7val0.15test0.15epochs{}batch{}patience{}trial{}'.format(epochs, batch_size,
                                                                                                                  patience, trial)


model.train(train_structures = training_structures, train_targets = training_targets,
            validation_structures = val_structures, validation_targets = val_targets,
            epochs = epochs, batch_size = batch_size, scrub_failed_structures = True,
            patience = patience, dirname = model_dir, save_checkpoint = True, verbose = 2)


# Model Testing

if os.path.isdir(model_dir):
    model_files = os.listdir(model_dir)
    model_files = [mf for mf in model_files if '.hdf5' in mf]
    #model_files.sort(key=lambda fn: os.path.getmtime(model_path + '/' + fn))
    val_errors = [float(x.split('_')[-1][:-5]) for x in model_files]
    min_val = min(val_errors)
    print(min_val)
    idx = val_errors.index(min_val)
    model_file_path = os.path.join(model_dir, model_files[idx])
elif os.path.isfile(model_path):
    model_file_path = model_path
else:
    print('Input a GN model path!')
    
nfeat_bond = 100
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

model = load_model(model_file_path, custom_objects = _CUSTOM_OBJECTS)

gn_model = GraphModel(model=model,
                  graph_converter=graph_converter,
                  centers=gaussian_centers,
                  width=gaussian_width)

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