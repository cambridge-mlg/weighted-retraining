""" Code to do Bayesian Optimization with GP """

import argparse
import logging
import functools
import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow

# configs
gpflow.config.set_default_float(np.float32)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--logfile",
    type=str,
    help="file to log to",
    default="gp_opt.log"
)
parser.add_argument(
    "--seed",
    type=int,
    required=True
)
parser.add_argument(
    "--gp_file",
    type=str,
    required=True,
    help="file to load GP hyperparameters from",
)
parser.add_argument(
    "--data_file",
    type=str,
    help="file to load data from",
    required=True
)
parser.add_argument(
    "--save_file",
    type=str,
    required=True,
    help="File to save results to"
)
parser.add_argument(
    "--n_out",
    type=int,
    default=1,
    help="Number of optimization points to return"
)
parser.add_argument(
    "--n_starts",
    type=int,
    default=20,
    help="Number of optimization starts to use"
)
parser.add_argument(
    "--no_early_stopping",
    dest="early_stopping",
    action="store_false",
    help="Flag to turn off early stopping"
)


# Functions to calculate expected improvement
# =============================================================================
def _ei_tensor(x):
    """ convert arguments to tensor for ei calcs """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def neg_ei(x, gp, fmin, check_type=True):
    if check_type:
        x = _ei_tensor(x)

    std_normal = tfp.distributions.Normal(loc=0., scale=1.)
    mu, var = gp.predict_f(x)
    sigma = tf.sqrt(var)
    z = (fmin - mu) / sigma

    ei = ((fmin - mu) * std_normal.cdf(z) +
          sigma * std_normal.prob(z))
    return -ei


def neg_ei_and_grad(x, gp, fmin, numpy=True):
    x = _ei_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        val = neg_ei(x, gp, fmin, check_type=False)
    grad = tape.gradient(val, x)
    if numpy:
        return val.numpy(), grad.numpy()
    else:
        return val, grad


def robust_multi_restart_optimizer(
        func_with_grad,
        X_train, y_train,
        num_pts_to_return=1,
        num_random_starts=5,
        num_good_starts=5,
        good_point_cutoff=0.0,
        use_tqdm=False,
        bounds_abs=4.,
        return_res=False,
        logger=None,
        early_stop=True,
        shuffle_points=True):
    """
    Wrapper that calls scipy's optimize function at many different start points.
    It uses a mix of random starting points, and good points in the dataset.
    """

    # wrapper for tensorflow functions, that handles array flattening and dtype changing
    def objective1d(v):
        return tuple([arr.ravel().astype(np.float64) for arr in func_with_grad(v)])

    # Set up points to optimize in
    rand_points = [np.random.randn(X_train.shape[1]).astype(np.float32)
                   for _ in range(num_random_starts)]
    top_point_idxs = np.arange(len(y_train))[(
        y_train <= good_point_cutoff).ravel()]
    chosen_top_point_indices = np.random.choice(
        top_point_idxs, size=num_good_starts, replace=False)
    top_points = [X_train[i].ravel().copy() for i in chosen_top_point_indices]
    all_points = rand_points + top_points
    point_sources = ["rand"] * len(rand_points) + ["good"] * len(top_points)

    # Optionally shuffle points (early stopping means order can matter)
    if shuffle_points:
        _list_together = list(zip(all_points, point_sources))
        np.random.shuffle(_list_together)
        all_points, point_sources = list(zip(*_list_together))
        del _list_together

    # Main optimization loop
    start_time = time.time()
    num_good_results = 0
    if use_tqdm:
        all_points = tqdm(all_points)
    opt_results = []
    for i, (x, src) in enumerate(zip(all_points, point_sources)):
        res = minimize(
            fun=objective1d, x0=x,
            jac=True,
            bounds=[(-bounds_abs, bounds_abs) for _ in range(X_train.shape[1])])

        opt_results.append(res)

        if logger is not None:
            logger.info(
                f"Iter#{i} t={time.time()-start_time:.1f}s: val={sum(res.fun):.2e}, "
                f"init={src}, success={res.success}, msg={str(res.message.decode())}")

        # Potentially do early stopping
        # Good results succeed, and stop due to convergences, not low gradients
        result_is_good = res.success and ("REL_REDUCTION_OF_F_<=_FACTR*EPSMCH" in res.message.decode())
        if result_is_good:
            num_good_results += 1
            if (num_good_results >= num_pts_to_return) and early_stop:
                logger.info(f"Early stopping since {num_good_results} good points found.")
                break

    # Potentially directly return optimization results
    if return_res:
        return opt_results

    # Find the best successful results
    successful_results = [res for res in opt_results if res.success]
    sorted_results = sorted(successful_results, key=lambda r: r.fun.sum())
    x_out = [res.x for res in sorted_results[:num_pts_to_return]]
    opt_vals_out = [res.fun.sum()
                    for res in sorted_results[:num_pts_to_return]]
    return np.array(x_out), opt_vals_out


def gp_opt(gp_file, data_file, save_file, n_out, logfile, n_starts=20, early_stopping=True):
    """ Do optimization via GPFlow"""

    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(logfile))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        X_test = npz['X_test']
        y_train = npz['y_train']
        y_test = npz['y_test']

    # Initialize the GP
    with np.load(gp_file, allow_pickle=True) as npz:
        Z = npz['Z']
        kernel_lengthscales = npz['kernel_lengthscales']
        kernel_variance = npz['kernel_variance']
        likelihood_variance = npz['likelihood_variance']

    # Make the GP
    gp = gpflow.models.SGPR(
        data=(X_train, y_train),
        inducing_variable=Z,
        kernel=gpflow.kernels.SquaredExponential(
            lengthscales=kernel_lengthscales,
            variance=kernel_variance
        )
    )
    gp.likelihood.variance.assign(likelihood_variance)

    """ 
    Choose a value for fmin.
    In pratice, it seems that for a very small value, the EI gradients
    are very small, so the optimization doesn't converge.
    Choosing a low-ish percentile seems to be a good comprimise.
    """
    fmin = np.percentile(y_train, 10)
    LOGGER.info(f"Using fmin={fmin:.2f}")

    # Choose other bounds/cutoffs
    good_point_cutoff = np.percentile(y_train, 20)
    LOGGER.info(f"Using good point cutoff={good_point_cutoff:.2f}")
    data_bounds = np.percentile(np.abs(X_train), 99.9)  # To account for outliers
    LOGGER.info(f"Data bound of {data_bounds} found")
    data_bounds *= 1.1
    LOGGER.info(f"Using data bound of {data_bounds}")

    # Run the optimization, with a mix of random and good points
    LOGGER.info("\n### Starting optimization ### \n")
    latent_pred, ei_vals = robust_multi_restart_optimizer(
        functools.partial(neg_ei_and_grad, gp=gp, fmin=fmin),
        X_train, y_train,
        num_pts_to_return=n_out,
        num_random_starts=n_starts // 2,
        num_good_starts=n_starts - n_starts // 2,
        good_point_cutoff=good_point_cutoff,
        bounds_abs=data_bounds,
        logger=LOGGER,
        early_stop=early_stopping
    )
    LOGGER.info(f"Done optimization! {len(latent_pred)} results found\n\n.")

    # Save results
    latent_pred = np.array(latent_pred, dtype=np.float32)
    np.save(save_file, latent_pred)

    # Make some gp predictions in the log file
    LOGGER.info("EI results:")
    LOGGER.info(ei_vals)

    mu, var = gp.predict_f(latent_pred)
    LOGGER.info("mu at points:")
    LOGGER.info(list(mu.numpy().ravel()))
    LOGGER.info("var at points:")
    LOGGER.info(list(var.numpy().ravel()))

    LOGGER.info("\n\nEND OF SCRIPT!")

    return latent_pred


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    gp_opt(args.gp_file, args.data_file, args.save_file, args.n_out, args.logfile, args.n_starts, args.early_stopping)
