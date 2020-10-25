import time
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from weighted_retraining.opt_scripts.base import add_common_args, add_gp_args
from weighted_retraining.expr import expr_data
from weighted_retraining.train_scripts import train_expr
from weighted_retraining.utils import save_object, print_flush, DataWeighter
from weighted_retraining.gp_train import gp_train
from weighted_retraining.gp_opt import gp_opt


def main():
    """ main """

    # parse arguments
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser = add_gp_args(parser)
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
        default="/home/ead54/rds/hpc-work/project_data/data",
        help="directory of datasets",
    )
    parser.add_argument(
        '--retrain_from_scratch',
        dest="retrain_from_scratch",
        action="store_true",
        help="flag to retrain the generative model from scratch in every iteration"
    )
    parser.add_argument(
        "--n_data",
        type=int,
        default=100000,
        help="number of datapoints to use",
    )
    parser.add_argument(
        "--ignore_percentile",
        type=int,
        default=0,
        help="percentile of scores to ignore"
    )
    parser.add_argument(
        '--use_test_set',
        dest="use_test_set",
        action="store_true",
        help="flag to use a test set for evaluating the sparse GP"
    )
    parser.add_argument(
        '--use_full_data_for_gp',
        dest="use_full_data_for_gp",
        action="store_true",
        help="flag to use the full dataset for training the GP"
    )
    parser.add_argument(
        "--n_decode_attempts",
        type=int,
        default=100,
        help="number of decoding attempts",
    )

    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # run tensorflow in eager mode
    tf.config.experimental_run_functions_eagerly(True)

    # create result directory
    directory = Path(args.result_root)
    directory.mkdir(parents=True, exist_ok=True)

    # define DataWeighter
    data_weighter = DataWeighter(args)

    # get initial dataset
    data_str, data_enc, data_scores = expr_data.get_initial_dataset_and_weights(
        Path(args.data_dir), args.ignore_percentile, args.n_data)

    # print python command run
    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")

    # set up results dictionary
    results = dict(
        params=cmd,
        opt_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
    )

    start_time = time.time()
    n_opt_iters = int(np.ceil(args.query_budget / args.retraining_frequency))
    n_bo_iters = args.query_budget // n_opt_iters
    for ite in range(1, n_opt_iters + 1):
        print_flush("OPTIMIZATION ITERATION {}/{} ({:.3f}s)".format(ite, n_opt_iters, time.time() - start_time))
        opt_dir = directory / "opt{}".format(ite)
        opt_dir.mkdir(exist_ok=True)

        # load/update model
        model = train_expr.get_model(args.pretrained_model_file, args.latent_dim, args.n_init_retrain_epochs,
                                     args.n_retrain_epochs, args.retrain_from_scratch, ite, opt_dir, data_enc, data_scores, data_weighter)

        # draw and store samples from model's latent space
        if args.samples_per_model > 0:
            sample_x, sample_y = latent_sampling(args.samples_per_model, model, args.latent_dim, args.n_decode_attempts)
            results["sample_points"].append(sample_x)
            results["sample_properties"].append(sample_y)
            results["sample_versions"].append(ite-1)

        # select new inputs via optimization or sampling
        if args.lso_strategy == "opt":
            new_inputs, new_scores = latent_optimization(n_bo_iters, ite, model, args.seed, args.n_inducing_points, opt_dir, start_time, args.n_decode_attempts,
                args.use_test_set, args.use_full_data_for_gp, data_scores, data_str, args.n_best_points, args.n_rand_points)
        elif args.lso_strategy == "sample":
            new_inputs, new_scores = latent_sampling(n_bo_iters, model, args.latent_dim, args.n_decode_attempts)
        else:
            raise NotImplementedError(args.lso_strategy)

        # update dataset and weights
        data_str, data_enc, data_scores = expr_data.update_dataset_and_weights(
            new_inputs, new_scores, data_str, data_enc, data_scores, model)

        # add new results
        results["opt_points"] += list(new_inputs)
        results["opt_point_properties"] += list(new_scores)
        results["opt_model_version"] += [ite-1] * len(new_inputs)

        # save results
        np.savez_compressed(str(directory / "results.npz"), **results, allow_pickle=True)

    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))


def latent_optimization(n_bo_iters, opt_iter, model, seed, n_inducing_points, directory, start_time, n_decode_attempts, use_test_set, use_full_data_for_gp, data_scores, data_str, n_best, n_rand):
    """ run Bayesian optimization loop """

    # compute latent encodings and corresponding scores to fit the GP on
    data_file = directory / "data.npz"
    X_train, y_train, X_test, y_test, X_mean, y_mean, X_std, y_std = expr_data.get_latent_encodings(
        use_test_set, use_full_data_for_gp, model, data_file, data_scores, data_str, n_best, n_rand)

    gp_file = None
    new_inputs = np.array([])
    new_scores = np.array([])
    for ite in range(n_bo_iters):
        print_flush("\tBO ITERATION {}/{} ({:.3f}s)".format(ite + 1, n_bo_iters, time.time() - start_time))

        # set random seed
        iter_seed = seed * ((opt_iter - 1) * n_bo_iters + ite)
        np.random.seed(iter_seed)
        tf.random.set_seed(iter_seed)

        # fit the GP model (using gpflow)
        print_flush("\t\tFitting predictive model...")
        init = kmean_init = gp_file == None
        new_gp_file = directory / "gp_iter.npz"
        gp_train(nZ=n_inducing_points, data_file=data_file, logfile=directory / "gp_train.log", save_file=new_gp_file,
                 n_perf_measure=1, use_test_set=X_test is not None, init=init, kmeans_init=kmean_init, gp_file=gp_file)
        gp_file = new_gp_file

        # identify the best inputs (using gpflow)
        print_flush("\t\tPicking new inputs via optimization...")
        new_latents = gp_opt(gp_file, data_file, directory / "gp_opt_res.npy", 1, directory / "gp_opt.log")
        new_latents = new_latents * X_std + X_mean

        # compute and save new inputs and corresponding scores
        new_inputs = np.append(new_inputs, model.decode_from_latent_space(zs=new_latents, n_decode_attempts=n_decode_attempts))
        new_scores = np.append(new_scores, expr_data.score_function([new_inputs[-1]]))

        # add new inputs and scores to training set and get new dataset filename
        X_train, y_train = expr_data.append_trainset(X_train, y_train, new_latents, np.array([new_scores[-1]]))
        expr_data.save_data(X_train, y_train, X_test, y_test, X_mean, X_std, y_mean, y_std, data_file)

    return new_inputs, new_scores


def latent_sampling(n_samples, model, latent_dim, n_decode_attempts):
    """ Draws samples from latent space and appends to the dataset """

    print_flush("\t\tPicking new inputs via sampling...")
    new_latents = np.random.randn(n_samples, latent_dim)
    new_inputs = model.decode_from_latent_space(zs=new_latents, n_decode_attempts=n_decode_attempts)
    new_scores = expr_data.score_function(new_inputs)

    return new_inputs, new_scores


if __name__ == "__main__":
    main()
