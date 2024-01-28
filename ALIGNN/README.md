# Crystal Structure Prediction with ALIGNN and Bayesian Optimization
Crystal structure prediction with ALIGNN as GNN backbone involves three key steps: (a) Training ALIGNN model (b) Geometry Optimization with Bayesian Optimization (c) Performance Evaluation

## Training ALIGNN model
ALIGNN model was trained on Matbench dataset and its augmented counterpart as required for formation energy prediction. Use https://github.com/usnistgov/alignn on how to train ALIGNN model.

## Geometry Optimization
Update `gnoa.in` as required with composition, optimal GNN model path, geometry optimization output path and other parameters. Run `python predict_structure.py` for geometry optimization. The final structure predicted by the algorithm is in `best.cif` file in the geometry optimization output directory.

## Performance Evaluation
Run `python crystal_matcher.py --compound <compound-name>` for performance evaluation. For example, `<compound-name>` can be `NaCl`.
