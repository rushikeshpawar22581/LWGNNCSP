import numpy as np
import random
import os
import csv
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure

class MatBenchDataset:
    def __init__(self, path_to_data,
                 n_train_rate: float = 0.70, n_val_rate: float = 0.15,
                 is_shuffle: bool = True, rand_seed: int = None):
        structures, targets = self.get_dataset(path_to_data)
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
        

        
        self.n_train = int(data_count * n_train_rate)
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


    def get_dataset(self, root_dir):
        assert os.path.exists(root_dir)
        id_prop_file = os.path.join(root_dir, 'id_prop.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        
        
        conventional_structures = []
        targets = []
        
        for i in range(len(self.id_prop_data)):
            cif_id, target = self.id_prop_data[i]
            conventional_structure = Structure.from_file(os.path.join(root_dir, cif_id+'.cif'))
            conventional_structures.append(conventional_structure)
            targets.append(float(target))
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
    