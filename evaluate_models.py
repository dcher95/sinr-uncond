import os
import numpy as np
import torch

import eval

eval_params = {}
eval_params['exp_base'] = './experiments'
eval_params['experiment_name'] = 'model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000'
save_out = ''
# evaluate:
# for eval_type in ['snt', 'iucn', 'geo_prior', 'geo_feature']:
#     eval_params['eval_type'] = eval_type
    
#     if eval_type == 'iucn':
#         eval_params['device'] = torch.device('cpu') # for memory reasons
#     cur_results = eval.launch_eval_run(eval_params)
#     np.save(os.path.join(eval_params['exp_base'], eval_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)

# for eval_type in ['snt', 'iucn', 'geo_prior', 'geo_feature']:
eval_type = 'geo_prior'
eval_params['eval_type'] = eval_type

# if eval_type == 'iucn':
#     eval_params['device'] = torch.device('cpu') # for memory reasons
cur_results = eval.launch_eval_run(eval_params)
np.save(os.path.join('/data/cher/sinr-uncond/experiments/unconditional', f'results_{eval_type}.npy'), cur_results)


'''
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
'''
