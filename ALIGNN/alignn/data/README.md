1. The file `matbench_demo.py` can be used to fetch Matbench formation energy dataset.
2. The file `predict_energy_batch.py` can be used to obtain batch inference of GNN model.
3. The file `augment_from_file_uniform.py` can be used to augment the dataset with controlled perturbation without affecting the energy labels.
4. The file `augment_from_file_modelmapped_uniform.py` can be used to augment the dataset with controlled perturbation while using an existing model to generate energy labels.
5. Use `predict_energy_holdout_unrelaxed.py` to predict energy of holdout unrelaxed structures (as mentioned in manuscript).
6. Use `analyze_holdout_unrelaxed.py` to compute (Metric)<sub>1</sub> and (Metric)<sub>2</sub> as mentioned in manuscript.
