""" Code for various 1-off functions """
import os
import sys
import uuid
import gzip
import pickle
import functools
import argparse
import datetime
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
from pathlib import Path
import torch
import pytorch_lightning as pl


# Weighting functions
class DataWeighter:
    weight_types = ["uniform", "rank", "dbas", "fb", "rwr", "cem-pi"]

    def __init__(self, hparams):

        if hparams.weight_type in ["uniform", "fb"]:
            self.weighting_function = DataWeighter.uniform_weights
        elif hparams.weight_type == "rank":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights, k_val=hparams.rank_weight_k
            )

        # Most other implementations are from:
        # https://github.com/dhbrookes/CbAS/blob/master/src/optimization_algs.py
        elif hparams.weight_type == "dbas":
            self.weighting_function = functools.partial(
                DataWeighter.dbas_weights,
                quantile=hparams.weight_quantile,
                noise=hparams.dbas_noise,
            )
        elif hparams.weight_type == "rwr":
            self.weighting_function = functools.partial(
                DataWeighter.rwr_weights, alpha=hparams.rwr_alpha
            )
        elif hparams.weight_type == "cem-pi":
            self.weighting_function = functools.partial(
                DataWeighter.cem_pi_weights, quantile=hparams.weight_quantile
            )

        else:
            raise NotImplementedError

        self.weight_quantile = hparams.weight_quantile
        self.weight_type = hparams.weight_type

    @staticmethod
    def normalize_weights(weights: np.array):
        """ Normalizes the given weights """
        return weights / np.mean(weights)

    @staticmethod
    def reduce_weight_variance(weights: np.array, data: np.array):
        """ Reduces the variance of the given weights via data replication """

        weights_new = []
        data_new = []
        for w, d in zip(weights, data):
            if w == 0.0:
                continue
            while w > 1:
                weights_new.append(1.0)
                data_new.append(d)
                w -= 1
            weights_new.append(w)
            data_new.append(d)

        return np.array(weights_new), np.array(data_new)

    @staticmethod
    def uniform_weights(properties: np.array):
        return np.ones_like(properties)

    @staticmethod
    def rank_weights(properties: np.array, k_val: float):
        """
        Calculates rank weights assuming maximization.
        Weights are not normalized.
        """
        if np.isinf(k_val):
            return np.ones_like(properties)
        ranks = np.argsort(np.argsort(-1 * properties))
        weights = 1.0 / (k_val * len(properties) + ranks)
        return weights

    @staticmethod
    def dbas_weights(properties: np.array, quantile: float, noise: float):
        y_star = np.quantile(properties, quantile)
        if np.isclose(noise, 0):
            weights = (properties >= y_star).astype(float)
        else:
            weights = stats.norm.sf(y_star, loc=properties, scale=noise)
        return weights

    @staticmethod
    def cem_pi_weights(properties: np.array, quantile: float):

        # Find quantile cutoff
        cutoff = np.quantile(properties, quantile)
        weights = (properties >= cutoff).astype(float)
        return weights

    @staticmethod
    def rwr_weights(properties: np.array, alpha: float):

        # Subtract max property value for more stable calculation
        # It doesn't change the weights since they are normalized by the sum anyways
        prop_max = np.max(properties)
        weights = np.exp(alpha * (properties - prop_max))
        weights /= np.sum(weights)
        return weights

    @staticmethod
    def add_weight_args(parser: argparse.ArgumentParser):
        weight_group = parser.add_argument_group("weighting")
        weight_group.add_argument(
            "--weight_type",
            type=str,
            choices=DataWeighter.weight_types,
            default="uniform",
        )
        weight_group.add_argument(
            "--rank_weight_k",
            type=float,
            default=None,
            help="k parameter for rank weighting",
        )
        weight_group.add_argument(
            "--weight_quantile",
            type=float,
            default=None,
            help="quantile argument for dbas, cem-pi cutoffs",
        )
        weight_group.add_argument(
            "--dbas_noise",
            type=float,
            default=None,
            help="noise parameter for dbas (to induce non-binary weights)",
        )
        weight_group.add_argument(
            "--rwr_alpha", type=float, default=None, help="alpha value for rwr"
        )
        return parser


# Various pytorch functions
def _get_zero_grad_tensor(device):
    """ return a zero tensor that requires grad. """
    loss = torch.as_tensor(0.0, device=device)
    loss = loss.requires_grad_(True)
    return loss


def save_object(obj, filename):
    """ Function that saves an object to a file using pickle """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, "wb") as dest:
        dest.write(result)
    dest.close()


def print_flush(text):
    print(text)
    sys.stdout.flush()


def update_hparams(hparams, model):

    # Make the hyperparameters match
    for k in model.hparams.keys():
        try:
            if vars(hparams)[k] != model.hparams[k]:
                print(
                    f"Overriding hparam {k} from {model.hparams[k]} to {vars(hparams)[k]}"
                )
                model.hparams[k] = vars(hparams)[k]
        except KeyError:  # not all keys match, it's ok
            pass

    # Add any new hyperparameters
    for k in vars(hparams).keys():
        if k not in model.hparams.keys():
            print(f'Adding missing hparam {k} with value "{vars(hparams)[k]}".')
            model.hparams[k] = vars(hparams)[k]


def add_default_trainer_args(parser, default_root=None):
    pl_trainer_grp = parser.add_argument_group("pl trainer")
    pl_trainer_grp.add_argument("--gpu", action="store_true")
    pl_trainer_grp.add_argument("--seed", type=int, default=0)
    pl_trainer_grp.add_argument("--root_dir", type=str, default=default_root)
    pl_trainer_grp.add_argument("--load_from_checkpoint", type=str, default=None)
    pl_trainer_grp.add_argument("--max_epochs", type=int, default=1000)


class SubmissivePlProgressbar(pl.callbacks.ProgressBar):
    """ progress bar with tqdm set to leave """

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Retraining Progress",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar
