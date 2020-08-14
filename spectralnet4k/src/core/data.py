'''
data.py: contains all data generating code for datasets used in the script
'''

import os, sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import numpy as np
from sklearn import preprocessing
from keras import layers
from keras import backend as K
from keras.models import model_from_json

from core import pairs


def get_data(params, data=None):
    '''
    Convenience function: preprocesses all data in the manner specified in params, and returns it
    as a nested dict with the following keys:

    the permutations (if any) used to shuffle the training and validation sets
    'p_train'                           - p_train
    'p_val'                             - p_val

    the data used for spectral net
    'spectral'
        'train_and_test'                - (x_train, y_train, x_val, y_val, x_test, y_test)
        'train_unlabeled_and_labeled'   - (x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled)
        'val_unlabeled_and_labeled'     - (x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled)

    the data used for siamese net, if the architecture uses the siamese net
    'siamese'
        'train_and_test'                - (pairs_train, dist_train, pairs_val, dist_val)
        'train_unlabeled_and_labeled'   - (pairs_train_unlabeled, dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
        'val_unlabeled_and_labeled'     - (pairs_val_unlabeled, dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)
    '''
    ret = {}

    # get data if not provided
    if data is None:
        x_train, x_test, y_train, y_test = load_data(params)
    else:
        print("WARNING: Using data provided in arguments. Must be tuple or array of format (x_train, x_test, y_train, y_test)")
        x_train, x_test, y_train, y_test = data

    ret['spectral'] = {}
    if params.get('use_all_data'):
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)
        x_test = np.zeros((0,) + x_train.shape[1:])
        y_test = np.zeros((0,))
    m = np.concatenate((y_train, y_test), axis=0)
    print('mshiy:', m)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)
    # split x training, validation, and test subsets
    if 'val_set_fraction' not in params:
        print("NOTE: Validation set required, setting val_set_fraction to 0.1")
        train_val_split = (.8, .2)
    elif params['val_set_fraction'] > 0 and params['val_set_fraction'] <= 1:
        train_val_split = (1 - params['val_set_fraction'], params['val_set_fraction'])
    else:
        raise ValueError("val_set_fraction is invalid! must be in range (0, 1]")

    p = np.arange(len(x_train) + len(x_test))
    print('p:', p)
    (x_train, y_train, p_train), (x_val, y_val, p_val) = split_data(x_train, y_train, train_val_split, permute=p[:len(x_train)])
    print('ytrain:', y_train)

    # further split each training and validation subset into its supervised and unsupervised sub-subsets
    if params.get('train_labeled_fraction'):
        train_split = (1 - params['train_labeled_fraction'], params['train_labeled_fraction'])
    else:
        train_split = (1, 0)
    (x_train_unlabeled, y_train_unlabeled, p_train_unlabeled), (x_train_labeled, y_train_labeled, _) = split_data(x_train, y_train, train_split)

    if params.get('val_labeled_fraction'):
        val_split = (1 - params['val_labeled_fraction'], params['val_labeled_fraction'])
    else:
        val_split = (1, 0)
    (x_val_unlabeled, y_val_unlabeled, p_val_unlabeled), (x_val_labeled, y_val_labeled, _) = split_data(x_val, y_val, val_split)
    print('ytrain:', y_train)

    # embed data in code space, if necessary
    all_data = [x_train, x_val, x_test, x_train_unlabeled, x_train_labeled, x_val_unlabeled, x_val_labeled]
    if params.get('use_code_space'):
       for i, d in enumerate(all_data):
            all_data[i] = embed_data(d, dset=params['dset'])
    else:
        # otherwise just flatten it
        for i, d in enumerate(all_data):
            all_data[i] = all_data[i].reshape((-1, np.prod(all_data[i].shape[1:])))
    x_train, x_val, x_test, x_train_unlabeled, x_train_labeled, x_val_unlabeled, x_val_labeled = all_data
    print('ytrain:',y_train)
    # collect everything into a dictionary
    ret['spectral']['train_and_test'] = (x_train, y_train, x_val, y_val, x_test, y_test)
    ret['spectral']['train_unlabeled_and_labeled'] = (x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled)
    ret['spectral']['val_unlabeled_and_labeled'] = (x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled)

    ret['p_train'] = p_train
    ret['p_val'] = p_val

    # get siamese data if necessary
    if 'siamese' in params['affinity']:
        ret['siamese'] = {}

        if params.get('precomputedKNNPath'):
            # if we use precomputed knn, we cannot shuffle the data; instead
            # we pass the permuted index array and the full matrix so that
            # create_pairs_from_unlabeled data can keep track of the indices
            p_train_unlabeled = p_train[:len(x_train_unlabeled)]
            train_path = params.get('precomputedKNNPath', '')
            if train_val_split[1] < 0.09 or params['siam_k'] > 100:
                # if the validation set is very small, the benefit of
                # the precomputation is small, and there will be a high miss
                # rate in the precomputed neighbors (neighbors that are not
                # in the validation set) so we just recomputed neighbors
                p_val_unlabeled = None
                val_path = ''
            else:
                p_val_unlabeled = p_val[:len(x_val_unlabeled)]
                val_path = params.get('precomputedKNNPath', '')
        else:
            # if we do not use precomputed knn, then this does not matter
            p_train_unlabeled = None
            train_path = params.get('precomputedKNNPath', '')
            p_val_unlabeled = None
            val_path = params.get('precomputedKNNPath', '')

        pairs_train_unlabeled, dist_train_unlabeled = pairs.create_pairs_from_unlabeled_data(
            x1=x_train_unlabeled,
            p=p_train_unlabeled,
            k=params.get('siam_k'),
            tot_pairs=params.get('siamese_tot_pairs'),
            precomputed_knn_path=train_path,
            use_approx=params.get('use_approx', False),
            pre_shuffled=True,
        )
        pairs_val_unlabeled, dist_val_unlabeled = pairs.create_pairs_from_unlabeled_data(
            x1=x_val_unlabeled,
            p=p_val_unlabeled,
            k=params.get('siam_k'),
            tot_pairs=params.get('siamese_tot_pairs'),
            precomputed_knn_path=val_path,
            use_approx=params.get('use_approx', False),
            pre_shuffled=True,
        )

        #get pairs for labeled data
        class_indices = [np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])]
        pairs_train_labeled, dist_train_labeled = pairs.create_pairs_from_labeled_data(x_train_labeled, class_indices)
        class_indices = [np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])]
        pairs_val_labeled, dist_val_labeled = pairs.create_pairs_from_labeled_data(x_train_labeled, class_indices)

        ret['siamese']['train_unlabeled_and_labeled'] = (pairs_train_unlabeled, dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
        ret['siamese']['val_unlabeled_and_labeled'] = (pairs_val_unlabeled, dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)

        #combine labeled and unlabeled pairs for training the siamese
        pairs_train = np.concatenate((pairs_train_unlabeled, pairs_train_labeled), axis=0)
        dist_train = np.concatenate((dist_train_unlabeled, dist_train_labeled), axis=0)
        pairs_val = np.concatenate((pairs_val_unlabeled, pairs_val_labeled), axis=0)
        dist_val = np.concatenate((dist_val_unlabeled, dist_val_labeled), axis=0)

        ret['siamese']['train_and_test'] = (pairs_train, dist_train, pairs_val, dist_val)

    return ret

def load_data(params):
    '''
    Convenience function: reads from disk, downloads, or generates the data specified in params
    '''
    if params['dset'] == 'reuters':
        with h5py.File('../../data/reuters/reutersidf_total.h5', 'r') as f:
            x = np.asarray(f.get('data'), dtype='float32')
            y = np.asarray(f.get('labels'), dtype='float32')

            n_train = int(0.9 * len(x))
            x_train, x_test = x[:n_train], x[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
    elif params['dset'] == 'data4000-4':
        train_dataset = h5py.File('../../data4000-4.h5', 'r')
        x_train = np.array(train_dataset['x_train'][:])
        y_train = np.array(train_dataset['y_train'][:]).astype('int')
        x_test = np.array(train_dataset['x_test'][:])
        y_test = np.array(train_dataset['y_test'][:]).astype('int')
        x_train = np.expand_dims(x_train, -1) / 255
        x_test = np.expand_dims(x_test, -1) / 255
        print('xtrain:', x_train.shape)
        print('xtest:', x_test.shape)
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    return x_train, x_test, y_train, y_test



def embed_data(x, dset):
    '''
    Convenience function: embeds x into the code space using the corresponding
    autoencoder (specified by dset).
    '''
    if not len(x):
        return np.zeros(shape=(0, 10))
    if dset == 'reuters':
        dset = 'reuters10k'

    json_path = 'H:/我的论文/实验代码部分/spectralnet4k/src/pretrain_weights/aeVGG16inout4096_{}.json'.format(dset)
    weights_path = 'H:/我的论文/实验代码部分/spectralnet4k/src/pretrain_weights/aeVGG16inout4096_{}_weights.h5'.format(dset)

    with open(json_path) as f:
        pt_aeVGG16 = model_from_json(f.read())
    pt_aeVGG16.load_weights(weights_path)

    x = x.reshape(-1, np.prod(x.shape[1:]))

    get_embeddings = K.function([pt_aeVGG16.input], [pt_aeVGG16.get_layer('dense_2').output])

    get_reconstruction = K.function([pt_aeVGG16.get_layer('dense_3').input], [pt_aeVGG16.output])
    x_embedded = predict_with_K_fn(get_embeddings, x)[0]
    x_recon = predict_with_K_fn(get_reconstruction, x_embedded)[0]
    reconstruction_mse = np.mean(np.square(x - x_recon))
    print("using pretrained embeddings; sanity check, total reconstruction error:", np.mean(reconstruction_mse))

    del pt_aeVGG16

    return x_embedded

def predict_with_K_fn(K_fn, x, bs=1000):
    '''
    Convenience function: evaluates x by K_fn(x), where K_fn is
    a Keras function, by batches of size 1000.
    '''
    if not isinstance(x, list):
        x = [x]
    num_outs = len(K_fn.outputs)
    y = [np.empty((len(x[0]), output_.get_shape()[1])) for output_ in K_fn.outputs]
    recon_means = []
    for i in range(int(x[0].shape[0]/bs + 1)):
        x_batch = []
        for x_ in x:
            x_batch.append(x_[i*bs:(i+1)*bs])
        temp = K_fn(x_batch)
        for j in range(num_outs):
            y[j][i*bs:(i+1)*bs] = temp[j]

    return y

def split_data(x, y, split, permute=None):
    '''
    Splits arrays x and y, of dimensionality n x d1 and n x d2, into
    k pairs of arrays (x1, y1), (x2, y2), ..., (xk, yk), where both
    arrays in the ith pair is of shape split[i-1]*n x (d1, d2)

    x, y:       two matrices of shape n x d1 and n x d2
    split:      a list of floats of length k (e.g. [a1, a2,..., ak])
                where a, b > 0, a, b < 1, and a + b == 1
    permute:    a list or array of length n that can be used to
                shuffle x and y identically before splitting it

    returns:    a tuple of tuples, where the outer tuple is of length k
                and each of the k inner tuples are of length 3, of
                the format (x_i, y_i, p_i) for the corresponding elements
                from x, y, and the permutation used to shuffle them
                (in the case permute == None, p_i would simply be
                range(split[0]+...+split[i-1], split[0]+...+split[i]),
                i.e. a list of consecutive numbers corresponding to the
                indices of x_i, y_i in x, y respectively)
    '''
    n = len(x)
    if permute is not None:
        if not isinstance(permute, np.ndarray):
            raise ValueError("Provided permute array should be an np.ndarray, not {}!".format(type(permute)))
        if len(permute.shape) != 1:
            raise ValueError("Provided permute array should be of dimension 1, not {}".format(len(permute.shape)))
        if len(permute) != len(x):
            raise ValueError("Provided permute should be the same length as x! (len(permute) = {}, len(x) = {}".format(len(permute), len(x)))
    else:
        permute = np.arange(len(x))

    if np.sum(split) != 1:
        raise ValueError("Split elements must sum to 1!")

    ret_x_y_p = []
    prev_idx = 0
    for s in split:
        idx = prev_idx + np.round(s * n).astype(np.int)
        p_ = permute[prev_idx:idx]
        x_ = x[p_]
        y_ = y[p_]
        prev_idx = idx
        ret_x_y_p.append((x_, y_, p_))

    return tuple(ret_x_y_p)

def pre_process(x_train, x_test, standardize):
    '''
    Convenience function: uses the sklearn StandardScaler on x_train
    and x_test if standardize == True
    '''
    #if we are going to standardize
    if standardize:
        #standardize the train data set
        preprocessor = preprocessing.StandardScaler().fit(x_train)
        x_train = preprocessor.transform(x_train)
        #if we have test data
        if x_test.shape[0] > 0:
            #standardize the test data set
            preprocessor = preprocessing.StandardScaler().fit(x_test)
            x_test = preprocessor.transform(x_test)
    return x_train, x_test

