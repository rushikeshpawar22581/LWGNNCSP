import pandas as pd
import numpy as np
import os

path_to_dir = '../results/Holdout_Unrelaxed/'
subdirs = sorted(list(os.walk(path_to_dir))[0][1])
thresholds = [-5]

for subdir in subdirs:
    pred_dir = os.path.join(path_to_dir, subdir)
    df = pd.read_csv(os.path.join(pred_dir, 'pred_results.csv'), header = None)
    pred_form = np.array(df.iloc[:, 2])
    num_pred_less_than_relaxed = np.sum(np.array(df.iloc[:, 1]) > np.array(df.iloc[:, 2]))
    print('Model : {}'.format(subdir))

    print('{} out of {} perturbed structures were predicted to have formation energy less than that of relaxed mapping energy.'.format(num_pred_less_than_relaxed, len(pred_form)))
    for threshold in thresholds:
        num_pred_outliers = np.sum(pred_form < threshold)
        print('{} out of {} perturbed structures were predicted to have formation energy less than {} eV/atom.'.format(num_pred_outliers, len(pred_form), threshold))
    
