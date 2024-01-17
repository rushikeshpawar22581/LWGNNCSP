# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:52:37 2022

@author: ashro
"""
import argparse
import os
import shutil
import sys
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_prop import CIFData, collate_pool
from model import CrystalGraphConvNet

class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def string_to_float(x):
    try:
        return float(x.split('[')[1].split(']')[0])
    except:
        return float(x)
    
    
def predict_energy(modelpath, cifdir):
    #We can either pass structure or the cif file as input along with model.
    #Output must be a single number with energy value.
    
    #Creating fake id_prop.csv to bypass error from cgcnn.
    #cif_files = [x.split('.')[0] for x in os.listdir(cifdir) if x.endswith('.cif')]
    #id_prop_df = pd.DataFrame({'cifs' : cif_files, 'energy' : [0] * len(cif_files)})
    #id_prop_df.to_csv(os.path.join(cifdir, 'id_prop.csv'), header = False, index = False)
    
    cuda = torch.cuda.is_available()
    
    model_checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    
    dataset = CIFData(cifdir)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False,
                             num_workers=1, collate_fn=collate_fn,
                             pin_memory=cuda)
    
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=False)
    
    if cuda:
        model.cuda()
        
    normalizer = Normalizer(torch.zeros(3))
    model.load_state_dict(model_checkpoint['state_dict'])
    normalizer.load_state_dict(model_checkpoint['normalizer'])
    #print("=> loaded model '{}' (epoch {}, validation {})".format(modelpath, model_checkpoint['epoch'], model_checkpoint['best_mae_error']))
    
    
    model.eval()
    all_cif_ids = []
    all_ground_truth = []
    all_preds = []
    for i, (input, target, batch_cif_ids) in enumerate(test_loader):
        print('Batch : {}'.format(i))
        batch_start_time = time.time()
        with torch.no_grad():
            if cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
                
        output, _ = model(*input_var)
        final_output = normalizer.denorm(output.data.cpu())
        batch_end_time = time.time()
        print('Per batch inference time (Batch Size = 128) : {}s'.format(round(batch_end_time - batch_start_time, 2)))
        #print(len(list(batch_cif_ids)), len(list(final_output)))
        
        
        all_cif_ids.extend(list(batch_cif_ids))
        
        final_output_float = []
        for i in range(len(list(final_output))):
            final_output_float.append(string_to_float(list(final_output)[i]))
        all_preds.extend(final_output_float)
        
        target_float = []
        for i in range(len(list(target))):
            target_float.append(string_to_float(list(target)[i]))
        all_ground_truth.extend(target_float)
    
    
    df = pd.DataFrame()
    df['struct_idx'] = all_cif_ids
    df['ground_truth_relaxed_mapping'] = all_ground_truth
    df['pred_form_energy'] = all_preds
    return df

if __name__=='__main__':
    modelpath = '../model/model_best_nopretrainingtrain132krelaxed132kunrelaxedmodelmappingM1train0.7val0.15test0.15epochs1000lr0.02lrmilestone800optimadamnconv4.pth.tar'
    cifdir = '../data/2unrelaxeddataperstructuremaxatomcount8_using132krelaxedstructure/'
    base_output_dir = '../results/Holdout_Unrelaxed/'
    pid = 'nopretrainingtrain132krelaxed132kunrelaxedmodelmappingM1train0.7val0.15test0.15epochs1000lr0.02lrmilestone800optimadamnconv4'
    output_dir = os.path.join(base_output_dir, pid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = predict_energy(modelpath, cifdir)
    df.to_csv(os.path.join(output_dir, 'pred_results.csv'), header = None, index = None)
    
