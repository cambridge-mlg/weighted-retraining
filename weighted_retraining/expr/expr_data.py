""" Code for loading and manipulating the arithmetic expression data """

import os
import h5py
import numpy as np
from pathlib import Path
from numpy import exp, sin
from tqdm import tqdm

from weighted_retraining.utils import print_flush


def load_data_str(data_dir):
    """ load the arithmetic expression data in string format """

    fname = 'equation2_15_dataset.txt'
    with open(data_dir / fname) as f:
        eqs = f.readlines()

    for i in range(len(eqs)):
        eqs[i] = eqs[i].strip().replace(' ', '')

    return eqs


def load_data_enc(data_dir):
    """ load the arithmetic expression dataset in one-hot encoded format """

    fname = 'eq2_grammar_dataset.h5'
    h5f = h5py.File(data_dir / fname, 'r')
    data = h5f['data'][:]
    h5f.close()

    return data


def get_initial_dataset_and_weights(data_dir, ignore_percentile, n_data):
    """ get the initial dataset (with corresponding scores) and the sample weights """

    # load equation dataset, both one-hot encoded and as plain strings, and compute corresponding scores
    data_str = load_data_str(data_dir)
    data_enc = load_data_enc(data_dir)
    data_scores = score_function(data_str)

    # subsample data based on the desired percentile and # of datapoints
    perc = np.percentile(data_scores, ignore_percentile)
    perc_idx = data_scores >= perc
    data_idx = np.random.choice(sum(perc_idx), min(n_data, sum(perc_idx)), replace=False)
    data_str = list(np.array(data_str)[perc_idx][data_idx])
    data_enc = data_enc[perc_idx][data_idx]
    data_scores = data_scores[perc_idx][data_idx]

    return data_str, data_enc, data_scores


def update_dataset_and_weights(new_inputs, new_scores, data_str, data_enc, data_scores, model):
    """ update the dataet and the sample weights """

    # discard invalid (None) inputs and their corresponding scores
    valid_idx = np.array(new_inputs) != None
    valid_inputs = list(new_inputs[valid_idx])
    valid_scores = new_scores[valid_idx]
    print_flush("\tDiscarding {}/{} new inputs that are invalid!".format(len(new_inputs) - len(valid_inputs), len(new_inputs)))

    # add new inputs and scores to dataset, both as plain string and one-hot vector
    print_flush("\tAppending new valid inputs to dataset...")
    data_str += valid_inputs
    new_inputs_one_hot = model.smiles_to_one_hot(valid_inputs)
    data_enc = np.append(data_enc, new_inputs_one_hot, axis=0)
    data_scores = np.append(data_scores, valid_scores)

    return data_str, data_enc, data_scores


def subsample_dataset(X, y, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand):
    """ subsample dataset for training the sparse GP """

    if use_test_set:
        X_train, y_train, X_test, y_test = split_dataset(X, y)
    else:
        X_train, y_train, X_test, y_test = X, y, None, None

    if len(y_train) < n_best + n_rand:
        n_best = int(n_best / (n_best + n_rand) * len(y_train))
        n_rand = int(n_rand / (n_best + n_rand) * len(y_train))

    if not use_full_data_for_gp:
        # pick n_best best points and n_rand random points
        best_idx = np.argsort(np.ravel(y_train))[:n_best]
        rand_idx = np.argsort(np.ravel(y_train))[np.random.choice(
            list(range(n_best, len(y_train))), n_rand, replace=False)]
        all_idx = np.concatenate([best_idx, rand_idx])
        X_train = X_train[all_idx, :]
        y_train = y_train[all_idx]

    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    save_data(X_train, y_train, X_test, y_test, X_mean, X_std, y_mean, y_std, data_file)

    return X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std


def split_dataset(X, y, split=0.9):
    """ split the data into a train and test set """

    n = X.shape[0]
    permutation = np.random.choice(n, n, replace=False)

    X_train = X[permutation, :][0: np.int(np.round(split * n)), :]
    y_train = y[permutation][0: np.int(np.round(split * n))]

    X_test = X[permutation, :][np.int(np.round(split * n)):, :]
    y_test = y[permutation][np.int(np.round(split * n)):]

    return X_train, y_train, X_test, y_test


def append_trainset(X_train, y_train, new_inputs, new_scores):
    """ add new inputs and scores to training set """

    if len(new_inputs) > 0:
        X_train = np.concatenate([X_train, new_inputs], 0)
        y_train = np.concatenate([y_train, new_scores[:, np.newaxis]], 0)
    return X_train, y_train


def save_data(X_train, y_train, X_test, y_test, X_mean, X_std, y_mean, y_std, data_file):
    """ save data """

    X_train_ = (X_train - X_mean) / X_std
    y_train_ = (y_train - y_mean) / y_std
    np.savez_compressed(
        data_file,
        X_train=np.float32(X_train_),
        y_train=np.float32(y_train_),
        X_test=np.float32(X_test),
        y_test=np.float32(y_test),
    )


def score_function(inputs, target_eq='1 / 3 + x + sin( x * x )', worst=7.0):
    """ compute equation scores of given inputs """

    # define inputs and outputs of ground truth target expression
    x = np.linspace(-10, 10, 1000)
    yT = np.array(eval(target_eq))

    scores = []
    for inp in inputs:
        try:
            y_pred = np.array(eval(inp))
            scores.append(np.minimum(worst, np.log(1 + np.mean((y_pred - yT)**2))))
        except:
            scores.append(worst)

    return np.array(scores)


def get_latent_encodings(use_test_set, use_full_data_for_gp, model, data_file, data_scores, data_str, n_best, n_rand, bs=5000):
    """ get latent encodings and split data into train and test data """

    print_flush("\tComputing latent training data encodings and corresponding scores...")
    n_batches = int(np.ceil(len(data_str) / bs))
    Xs = [model.encode(data_str[i * bs:(i + 1) * bs]) for i in tqdm(range(n_batches))]
    X = np.concatenate(Xs, axis=0)
    y = data_scores.reshape((-1, 1))

    return subsample_dataset(X, y, data_file, use_test_set, use_full_data_for_gp, n_best, n_rand)
