""" Train a Grammar VAE on arithmetic expression data """

import os
import h5py
import argparse
import itertools
import numpy as np
from pathlib import Path
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from weighted_retraining.expr.expr_model import EquationVAE
from weighted_retraining.expr.equation_vae import EquationGrammarModel
from weighted_retraining.expr.expr_data import load_data_str, load_data_enc, score_function
from weighted_retraining.utils import print_flush, DataWeighter


def main():
    """ Train model from scratch """

    # parse arguments
    parser = argparse.ArgumentParser()
    parser = DataWeighter.add_weight_args(parser)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=25,
        help="dimensionality of latent space",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="assets/data/expr"
        help="directory of datasets",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="directory of model",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--ignore_percentile",
        type=int,
        default=50,
        help="percentile of scores to ignore"
    )
    weight_group.add_argument(
        "--k",
        type=str,
        default="inf",
        help="k parameter for rank weighting",
    )
    args = parser.parse_args()
    args.weight_type = "rank"
    args.rank_weight_k = float(args.k)

    # print python command run
    print_flush(' '.join(sys.argv[1:]))

    # set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # run tensorflow in eager mode
    tf.config.experimental_run_functions_eagerly(True)

    # load and subsample equation dataset and compute corresponding scores
    data_str = load_data_str(Path(args.data_dir))
    data_enc = load_data_enc(Path(args.data_dir))
    data_scores = score_function(data_str)
    perc = np.percentile(data_scores, args.ignore_percentile)
    perc_idx = data_scores >= perc
    data = data_enc[perc_idx]
    scores = -data_scores[perc_idx]

    # compute sample weights (multiply scores by -1 as our goal is _minimization_)
    data_weighter = DataWeighter(args)
    sample_weights = DataWeighter.normalize_weights(data_weighter.weighting_function(scores))
    sample_weights, data = DataWeighter.reduce_weight_variance(sample_weights, data)

    # train model
    model_dir = Path(args.root_dir) / f"expr-k_{args.k}.hdf5"
    train_model(True, args.latent_dim, args.n_epochs, data, model_dir, sample_weights=sample_weights, batch_size=args.batch_size)


def train_model(retrain_from_scratch, latent_dim, epochs, data, new_weights_dir, prev_weights_dir=None, sample_weights=None, batch_size=600):
    """ train model """

    if retrain_from_scratch:
        MAX_LEN = 15
        import weighted_retraining.expr.eq_grammar as G
        data_info = G.gram.split('\n')
        vae = EquationVAE().create(data_info, max_length=MAX_LEN, latent_rep_size=latent_dim)
    else:
        vae = EquationGrammarModel(prev_weights_dir, latent_rep_size=latent_dim).vae

    checkpointer = ModelCheckpoint(filepath=new_weights_dir, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)

    if sample_weights is None:
        sample_weights = np.ones(len(data))

    vae.autoencoder.fit(
        x=data,
        y=data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpointer, reduce_lr],
        validation_split=0.1,
        shuffle=True,
        sample_weight=sample_weights,
    )


def get_model(pretrained_model_file, latent_dim, n_init_retrain_epochs, n_retrain_epochs, retrain_from_scratch, ite, save_dir, data_enc, data_scores, data_weighter):
    """ load or train the model """

    if ite == 1:
        print_flush("Loading pre-trained model...")
        new_weights_dir = pretrained_model_file

    else:
        print_flush("\t(Re-)training model...")

        # compute sample weights (multiply scores by -1 as our goal is _minimization_)
        sample_weights = data_weighter.weighting_function(-1 * data_scores)
        if data_weighter.weight_type == "rank":
            # for rank-based weighting, normalize the weights and reduce their variance
            sample_weights = DataWeighter.normalize_weights(sample_weights)
            sample_weights, data = DataWeighter.reduce_weight_variance(sample_weights, data_enc)
        else:
            data = data_enc

        # train model
        new_weights_dir = str(save_dir / 'expr.hdf5')
        prev_weights_dir = pretrained_model_file if ite == 2 else new_weights_dir.replace(f"opt{ite}", f"opt{ite-1}")
        n_epochs = int(np.ceil(n_init_retrain_epochs if ite == 1 and n_init_retrain_epochs else n_retrain_epochs))
        train_model(retrain_from_scratch, latent_dim, n_epochs, data, new_weights_dir, prev_weights_dir, sample_weights)

    # load trained model
    model = EquationGrammarModel(new_weights_dir, latent_rep_size=latent_dim)

    return model


if __name__ == "__main__":
    main()
