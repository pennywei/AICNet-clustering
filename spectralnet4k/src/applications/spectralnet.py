'''
spectralnet.py: contains run function for spectralnet
'''
import sys, os, pickle, shutil
import tensorflow as tf
import numpy as np
import traceback
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2,3'
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop

from core import train
from core import costs
from core import networks
from core.layer import stack_layers
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds, plot_confusion_matrix

from matplotlib import offsetbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def run_net(data, params):
    #
    # UNPACK DATA
    #

    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
    x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral']['train_unlabeled_and_labeled']
    x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

    if 'siamese' in params['affinity']:
        pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']

    x = np.concatenate((x_train, x_val, x_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)

    X = x
    n_samples, n_features = X.shape

    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], '.', color=plt.cm.Set1(1),fontdict={'weight': 'bold', 'size': 20})
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    from sklearn import manifold

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    start_time = time.time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne)
    plt.show()


    if len(x_train_labeled):
        y_train_labeled_onehot = OneHotEncoder().fit_transform(y_train_labeled.reshape(-1, 1)).toarray()
    else:
        y_train_labeled_onehot = np.empty((0, len(np.unique(y))))

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
            'Unlabeled': params['batch_size'],
            'Labeled': params['batch_size'],
            'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

    input_shape = x.shape[1:]

    # spectralnet has three inputs -- they are defined here
    inputs = {
            'Unlabeled': Input(shape=input_shape,name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape,name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape,name='OrthonormInput'),
            }

    #
    # DEFINE AND TRAIN SIAMESE NET
    #

    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true)

        history = siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
                params['siam_lr'], params['siam_drop'], params['siam_patience'],
                params['siam_ne'], params['siam_batch_size'])

    else:
        siamese_net = None

    spectral_net = networks.SpectralNet(inputs, params['arch'], params.get('spec_reg'), y_true, y_train_labeled_onehot, params['n_clusters'], params['affinity'], params['scale_nbr'], params['n_nbrs'], batch_sizes, siamese_net, x_train, len(x_train_labeled))

    spectral_net.train(x_train_unlabeled, x_train_labeled, x_val_unlabeled, params['spec_lr'], params['spec_drop'], params['spec_patience'], params['spec_ne'])

    print("finished training")
    #
    # EVALUATE
    #

    # get final embeddings
    x_spectralnet = spectral_net.predict(x)

    # get accuracy and nmi
    kmeans_assignments, km = get_cluster_sols(x_spectralnet, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init':10})
    y_spectralnet, _, _1 = get_y_preds(kmeans_assignments, y, params['n_clusters'])
    print_accuracy(kmeans_assignments, y, params['n_clusters'])

    from sklearn.metrics import normalized_mutual_info_score as nmi
    nmi_score = nmi(kmeans_assignments, y)
    print('NMI: ' + str(np.round(nmi_score, 3)))

    if params['generalization_metrics']:
        x_spectralnet_train = spectral_net.predict(x_train_unlabeled)
        x_spectralnet_test = spectral_net.predict(x_test)
        km_train = KMeans(n_clusters=params['n_clusters']).fit(x_spectralnet_train)
        from scipy.spatial.distance import cdist
        dist_mat = cdist(x_spectralnet_test, km_train.cluster_centers_)
        closest_cluster = np.argmin(dist_mat, axis=1)
        print_accuracy(closest_cluster, y_test, params['n_clusters'], ' generalization')
        nmi_score = nmi(closest_cluster, y_test)
        print('generalization NMI: ' + str(np.round(nmi_score, 3)))

    return x_spectralnet, y_spectralnet

