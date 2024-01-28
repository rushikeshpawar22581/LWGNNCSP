# Crystal Structure Prediction with CGCNN and Bayesian Optimization
Crystal structure prediction with CGCNN as GNN backbone involves three key steps: (a) Training CGCNN model (b) Geometry Optimization with Bayesian Optimization (c) Performance Evaluation

## Training CGCNN model
CGCNN model was trained on Matbench dataset and its augmented counterpart as required for formation energy prediction. Use https://github.com/txie-93/cgcnn on how to train CGCNN model. In order to train pre-trained variants like CrysXPP and CrysGNN, follow https://github.com/kdmsit/crysxpp and https://github.com/kdmsit/crysgnn respectively. Instructions to augment the dataset as mentioned in manuscript and select the best model as per our framework can be found in `alignn/data/README.md` and `alignn/src/README.md` respectively.


## Geometry Optimization
Update `gnoa.in` as required with composition, optimal GNN model path, geometry optimization output path and other parameters. Run `python predict_structure.py` for geometry optimization. The final structure predicted by the algorithm is in `best.cif` file in the geometry optimization output directory.

## Performance Evaluation
Run `python crystal_matcher.py --compound <compound-name>` for performance evaluation. For example, `<compound-name>` can be `NaCl`.
