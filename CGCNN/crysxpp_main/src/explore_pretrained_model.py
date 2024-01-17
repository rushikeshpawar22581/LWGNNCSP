import numpy as np
import pandas as pd
import torch
from model import CrystalGraphConvNet
from collections import OrderedDict

pretrained_model_path = '../model/model_pretrain_crysxpppretrainon132krelaxedlr0.003nconv4epoch200atomfealen64batch64globallosstruelocallosstrue.pth'

ae_model = torch.load(pretrained_model_path)
ae_model_dict = ae_model.state_dict()
#ae_model_dict = ae_model['state_dict']
print(ae_model_dict.keys())
print(type(ae_model_dict))

for key in ae_model_dict.keys():
    if key != 'fc_adj.weight':
        print('Frobenius Norm of {} is {}'.format(key, torch.norm(ae_model_dict[key].to(dtype = torch.float64), p = 'fro')))
    else:
        break

'''
new_ae_model_dict = OrderedDict()
for key in ae_model_dict.keys():
    if key != 'conv_logvar.0.fc_full.weight':
        new_ae_model_dict[key] = ae_model_dict[key]
    else:
        break

print(new_ae_model_dict.keys())
'''
#print(ae_model_dict['conv_mean.0.bn1.num_batches_tracked'] == new_ae_model_dict['conv_mean.0.bn1.num_batches_tracked'])

# model = CrystalGraphConvNet(92, 41,atom_fea_len=64,n_conv=3,
#                                 h_fea_len=128,n_h=1,classification=False)
# model.to('cuda')
# model_dict = model.state_dict()
# print(model_dict.keys())
'''
print(model_dict['conv_shared.0.fc_full.weight'] == new_ae_model_dict['conv_shared.0.fc_full.weight'])

model_dict.update(new_ae_model_dict)

print(model_dict['conv_shared.0.fc_full.weight'] == new_ae_model_dict['conv_shared.0.fc_full.weight'])

model.load_state_dict(model_dict)

new_model_dict = model.state_dict()

print(ae_model_dict['conv_mean.0.bn1.num_batches_tracked'] == new_model_dict['conv_mean.0.bn1.num_batches_tracked'])

print(new_model_dict.keys())

print(new_model_dict)
'''






