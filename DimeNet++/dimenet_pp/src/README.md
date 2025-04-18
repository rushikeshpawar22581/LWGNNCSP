1. Use `train_dimenet_pp_kgcnn.py` to train DimeNet++ model on Matbench dataset. The hyperparameters used for training are available in `hyperparams.json`.
2. Command line args are as below:
    2.1. `--data_path` : Path to the dataset dir, the dir should contain subdir with name `cif_files` that has cif files and a csv file `id_prop.csv` with headers `filename` and `formation_energy`, `filename` should end with .cif.
    2.2. `--model_name` : Name of the model to be trained, for example `M1_Trial1`
    2.3. `--random_seed` : Random seed for used for training, for example `0`, different random seeds will give us different trials of the model
    2.4. `--hyperparams_path` : Path to the hyperparams.json file, for example `hyperparams.json`