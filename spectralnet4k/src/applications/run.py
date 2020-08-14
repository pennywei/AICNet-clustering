import sys, os

# add directories in src/ to path
sys.path.insert(0, 'path_to_spectralnet/src/')
#os.environ['CUDA_VISIBLE_DEVICES']="2,3"
# import run_net and get_data
from spectralnet import run_net
from core.data import get_data

# PARSE ARGUMENTS
import argparse

from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, help='gpu number to use', default='')

parser.add_argument('--dset', type=str, help='gpu number to use', default='data4000-4')

args = parser.parse_args()



# SELECT GPU

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



params = defaultdict(lambda: None)


# define hyperparameters
params = {
    'dset': 'data4000-4',
    'use_code_space': True,
    'val_set_fraction': 0.2,
    'siam_batch_size': 64,
    'n_clusters': 4,
    'affinity': 'siamese',
    'n_nbrs': 3,
    'scale_nbr': 2,
    'siam_k': 2,
    'siam_ne': 20,
    'spec_ne': 100,
    'siam_lr': 1e-3,
    'spec_lr': 1e-6,
    'siam_patience': 10,
    'spec_patience': 20,
    'siam_drop': 0.1,
    'spec_drop': 0.001,
    'batch_size': 64,
    'siam_reg': None,
    'spec_reg': None ,
    'siam_n':  None,
    'siamese_tot_pairs': 3200,
    'arch': [
        {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 4},
    ],
    'use_approx': False,
    'use_all_data': True,
}



# preprocess dataset
data = get_data(params)

# run spectral net
x_spectralnet, y_spectralnet = run_net(data, params)
