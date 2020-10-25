""" Run weighted retraining for shapes with the optimal model """

import sys
import logging
import itertools
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl

# My imports
from weighted_retraining.shapes.shapes_data import WeightedNumpyDataset
from weighted_retraining.shapes.shapes_model import ShapesVAE
from weighted_retraining import utils
from weighted_retraining.opt_scripts import base as wr_base


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):

    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = utils.SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val",)

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=1 if gpu else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=checkpointer,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar],
    )

    # Fit model
    trainer.fit(model, datamodule)


def _batch_decode_z_and_props(model, z, args, filter_unique=True):
    """
    helper function to decode some latent vectors and calculate their properties
    """
    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j : j + batch_size])
            img = img.cpu().numpy()
            z_decode.append(img)
            del img

    # Concatentate all points and convert to numpy
    z_decode = np.concatenate(z_decode, axis=0)
    z_decode = np.around(z_decode)  # convert to int
    z_decode = z_decode[:, 0, ...]  # Correct slicing
    if filter_unique:
        z_decode, uniq_indices = np.unique(
            z_decode, axis=0, return_index=True
        )  # Unique elements only
        z = z.cpu().numpy()[uniq_indices]

    # Calculate objective function values and choose which points to keep
    if args.property_key == "areas":
        z_prop = np.sum(z_decode, axis=(-1, -2))
    else:
        raise ValueError(args.property)

    if filter_unique:
        return z_decode, z_prop, z
    else:
        return z_decode, z_prop


def latent_optimization(args, model, datamodule, num_queries_to_do):
    """ do latent space optimization with the optimal model (aka cheating) """

    unit_line = np.linspace(-args.opt_bounds, args.opt_bounds, args.opt_grid_len)
    latent_grid = list(itertools.product(unit_line, repeat=model.latent_dim))
    latent_grid = np.array(latent_grid, dtype=np.float32)
    z_latent_opt = torch.as_tensor(latent_grid, device=model.device)

    z_decode, z_prop, z = _batch_decode_z_and_props(model, z_latent_opt, args)

    z_prop_argsort = np.argsort(-1 * z_prop)  # assuming maximization of property

    # Choose new points
    new_points = z_decode[z_prop_argsort[:num_queries_to_do]]
    y_new = z_prop[z_prop_argsort[:num_queries_to_do]]
    z_query = z[z_prop_argsort[:num_queries_to_do]]

    return new_points, y_new, z_query


def latent_sampling(args, model, datamodule, num_queries_to_do, filter_unique=True):
    """ Draws samples from latent space and appends to the dataset """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    return _batch_decode_z_and_props(model, z_sample, args, filter_unique=filter_unique)


def main_loop(args):

    # Seeding
    pl.seed_everything(args.seed)

    # Make results directory
    result_dir = Path(args.result_root).resolve()
    result_dir.mkdir(parents=True)
    data_dir = result_dir / "data"
    data_dir.mkdir()

    # Load data
    datamodule = WeightedNumpyDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit")

    # Load model
    model = ShapesVAE.load_from_checkpoint(args.pretrained_model_file)
    model.beta = model.hparams.beta_final  # Override any beta annealing

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_latent_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
        latent_space_snapshots=[],
        latent_space_snapshot_version=[],
    )

    # Set up latent space snapshot!
    results["latent_space_grid"] = np.array(
        list(itertools.product(np.arange(-4, 4.01, 0.5), repeat=model.latent_dim)),
        dtype=np.float32,
    )

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain, best=-float("inf"), n_train=len(datamodule.data_train)
    )

    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx

            # Optionally do retraining
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = result_dir / "retraining"
                version = f"retrain_{samples_so_far}"
                retrain_model(
                    model, datamodule, retrain_dir, version, num_epochs, args.gpu
                )

            # Draw samples for logs!
            if args.samples_per_model > 0:
                pbar.set_description("sampling")
                sample_x, sample_y = latent_sampling(
                    args, model, datamodule, args.samples_per_model, filter_unique=False
                )

                # Append to results dict
                results["sample_points"].append(sample_x)
                results["sample_properties"].append(sample_y)
                results["sample_versions"].append(ret_idx)

            # Take latent snapshot
            latent_snapshot = _batch_decode_z_and_props(
                model,
                torch.as_tensor(results["latent_space_grid"], device=model.device),
                args,
                filter_unique=False,
            )[0]
            results["latent_space_snapshots"].append(latent_snapshot)
            results["latent_space_snapshot_version"].append(ret_idx)

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)
            pbar.set_description("querying")

            # Do querying!
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            if args.lso_strategy == "opt":
                x_new, y_new, z_query = latent_optimization(
                    args, model, datamodule, num_queries_to_do
                )
            elif args.lso_strategy == "sample":
                x_new, y_new, z_query = latent_sampling(
                    args, model, datamodule, num_queries_to_do
                )
            else:
                raise NotImplementedError(args.lso_strategy)

            # Append new points to dataset
            datamodule.append_train_data(x_new, y_new)

            # Save a new dataset
            new_data_file = (
                data_dir / f"train_data_iter{samples_so_far + num_queries_to_do}.npz"
            )
            np.savez_compressed(
                str(new_data_file),
                data=datamodule.data_train,
                **{args.property_key: datamodule.prop_train},
            )

            # Save results
            results["opt_latent_points"] += [z for z in z_query]
            results["opt_points"] += [x for x in x_new]
            results["opt_point_properties"] += [y for y in y_new]
            results["opt_model_version"] += [ret_idx] * len(x_new)
            np.savez_compressed(str(result_dir / "results.npz"), **results)

            # Final update of progress bar
            postfix["best"] = max(postfix["best"], float(y_new.max()))
            postfix["n_train"] = len(datamodule.data_train)
            pbar.set_postfix(postfix)
            pbar.update(n=num_queries_to_do)


if __name__ == "__main__":

    # arguments and argument checking
    parser = argparse.ArgumentParser()
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)

    # Optimal model arguments
    opt_group = parser.add_argument_group(title="opt-model")
    opt_group.add_argument("--opt_bounds", type=float, default=4)
    opt_group.add_argument("--opt_grid_len", type=float, default=50)

    args = parser.parse_args()
    main_loop(args)
