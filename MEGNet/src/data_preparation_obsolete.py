# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:18:00 2023

@author: ashro
"""

import random
import numpy as np
from abc import abstractmethod
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class MatBenchDataset:
    def __init__(self,
                 n_train_rate: float = 0.70, n_val_rate: float = 0.15,
                 n_train: int = None, n_val: int = None,
                 is_shuffle: bool = True, rand_seed: int = None):
        structures, targets = self.get_dataset()
        print("Read the data.")
        self.structures, self.targets = np.array(structures), np.array(targets)
        print(self.structures.shape, self.targets.shape)
        data_count = len(targets)
        indexes = list(range(data_count))

        if is_shuffle:
            if rand_seed is not None:
                random.seed(rand_seed)
            random.shuffle(indexes)
            self.structures = list(self.structures[indexes])
            self.targets = list(self.targets[indexes])
        

        if n_train is None:
            self.n_train = int(data_count * n_train_rate)
        if n_val is None:
            self.n_val = int(data_count * n_val_rate)


    @property
    def training_set(self):
        return self.structures[:self.n_train], self.targets[:self.n_train]

    @property
    def val_set(self):
        return self.structures[self.n_train:self.n_train + self.n_val], self.targets[self.n_train:self.n_train + self.n_val]

    @property
    def test_set(self):
        return self.structures[self.n_train + self.n_val:], self.targets[self.n_train + self.n_val:]


    def get_dataset(self):
        from matminer.datasets import load_dataset

        print('I am within get_dataset() function.')
        df = load_dataset("matbench_mp_e_form")
        
        materials_to_remove = ['LiF', 'NaF', 'KF', 'RbF', 'CsF', 'LiCl', 'NaCl', 'KCl', 'RbCl', 'CsCl',
                              'BeO', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'CdO', 'BeS', 'MgS', 'CaS', 'SrS',
                              'BaS', 'ZnS', 'CdS', 'C', 'Si', 'GaAs', 'CdTe', 'CsPbI3', 'BN', 'GaN', 'ZrC',
                              'SiC', 'WC', 'MnO', 'GaP', 'AlP', 'InP', 'LiBr', 'NaBr', 'KBr', 'RbBr', 'BeSe',
                              'MgSe', 'CaSe', 'SrSe', 'BaSe']
        
        conventional_structures = []
        targets = []
        
        for i in range(len(df)):
            SGA = SpacegroupAnalyzer(df.iloc[i, 0])
            conventional_structure = SGA.get_conventional_standard_structure()
            if conventional_structure.composition.reduced_formula not in materials_to_remove:
                conventional_structures.append(conventional_structure)
                targets.append(df.iloc[i, 1])
            else:
                pass
        #structures, targets = df['structure'], df['e_form']

        return conventional_structures, targets
    
    
if __name__=='__main__':
    dataset = MatBenchDataset(n_train_rate = 0.70, n_val_rate = 0.15, is_shuffle = True,
                              rand_seed = 100)
    
    training_structures, training_targets = dataset.training_set
    val_structures, val_targets = dataset.val_set
    test_structures, test_targets = dataset.test_set
    
    print(len(training_structures), type(training_structures[0]))
    print(len(training_targets), type(training_targets[0]))
    
    
    
    
    
