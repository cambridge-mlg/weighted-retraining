import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pytorch_lightning as pl

NUM_WORKERS = 3


class WeightedNumpyDataset(pl.LightningDataModule):
    """ Implements a weighted numpy dataset (used for shapes task) """

    def __init__(self, hparams, data_weighter):
        super().__init__()
        self.dataset_path = hparams.dataset_path
        self.val_frac = hparams.val_frac
        self.property_key = hparams.property_key
        self.batch_size = hparams.batch_size

        self.data_weighter = data_weighter

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument(
            "--dataset_path", type=str, required=True, help="path to npz file"
        )

        data_group.add_argument("--batch_size", type=int, default=32)
        data_group.add_argument(
            "--val_frac",
            type=float,
            default=0.05,
            help="Fraction of val data. Note that data is NOT shuffled!!!",
        )
        data_group.add_argument(
            "--property_key",
            type=str,
            required=True,
            help="Key in npz file to the object properties",
        )
        return parent_parser

    def prepare_data(self):
        pass

    @staticmethod
    def _get_tensor_dataset(data):
        data = torch.as_tensor(data, dtype=torch.float)
        data = torch.unsqueeze(data, 1)
        return TensorDataset(data)

    def setup(self, stage):

        with np.load(self.dataset_path) as npz:
            all_data = npz["data"]
            all_properties = npz[self.property_key]
        assert all_properties.shape[0] == all_data.shape[0]

        N_val = int(all_data.shape[0] * self.val_frac)
        self.data_val = all_data[:N_val]
        self.prop_val = all_properties[:N_val]
        self.data_train = all_data[N_val:]
        self.prop_train = all_properties[N_val:]

        # Make into tensor datasets
        self.train_dataset = WeightedNumpyDataset._get_tensor_dataset(self.data_train)
        self.val_dataset = WeightedNumpyDataset._get_tensor_dataset(self.data_val)
        self.set_weights()

    def set_weights(self):
        """ sets the weights from the weighted dataset """

        # Make train/val weights
        self.train_weights = self.data_weighter.weighting_function(self.prop_train)
        self.val_weights = self.data_weighter.weighting_function(self.prop_val)

        # Create samplers
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.val_sampler = WeightedRandomSampler(
            self.val_weights, num_samples=len(self.val_weights), replacement=True
        )

    def append_train_data(self, x_new, prop_new):

        # Special adjustment for fb-vae: only add the best points
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(prop_new, self.data_weighter.weight_quantile)
            indices_to_add = prop_new >= cutoff

            # Filter all but top quantile
            x_new = x_new[indices_to_add]
            prop_new = prop_new[indices_to_add]
            assert len(x_new) == len(prop_new)

            # Replace data (assuming that number of samples taken is less than the dataset size)
            self.data_train = np.concatenate(
                [self.data_train[len(x_new) :], x_new], axis=0
            )
            self.prop_train = np.concatenate(
                [self.prop_train[len(x_new) :], prop_new], axis=0
            )
        else:

            # Normal treatment: just concatenate the points
            self.data_train = np.concatenate([self.data_train, x_new], axis=0)
            self.prop_train = np.concatenate([self.prop_train, prop_new], axis=0)
        self.train_dataset = WeightedNumpyDataset._get_tensor_dataset(self.data_train)
        self.set_weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=self.val_sampler,
            drop_last=True,
        )
