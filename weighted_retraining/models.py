""" code for base VAE model """

import argparse
import torch
import pytorch_lightning as pl


class BaseVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.latent_dim = hparams.latent_dim

        # Register buffers for prior
        self.register_buffer("prior_mu", torch.zeros([self.latent_dim]))
        self.register_buffer("prior_sigma", torch.ones([self.latent_dim]))

        # Create beta
        self.beta = hparams.beta_final
        self.beta_annealing = False
        if hparams.beta_start is not None:
            self.beta_annealing = True
            self.beta = hparams.beta_start
            assert (
                hparams.beta_step is not None
                and hparams.beta_step_freq is not None
                and hparams.beta_warmup is not None
            )

        self.logging_prefix = None
        self.log_progress_bar = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        vae_group = parser.add_argument_group("VAE")
        vae_group.add_argument("--latent_dim", type=int, required=True)
        vae_group.add_argument("--lr", type=float, default=1e-3)
        vae_group.add_argument("--beta_final", type=float, default=1.0)
        vae_group.add_argument(
            "--beta_start",
            type=float,
            default=None,
            help="starting beta value; if None then no beta annealing is used",
        )
        vae_group.add_argument(
            "--beta_step",
            type=float,
            default=None,
            help="multiplicative step size for beta, if using beta annealing",
        )
        vae_group.add_argument(
            "--beta_step_freq",
            type=int,
            default=None,
            help="frequency for beta step, if taking a step for beta",
        )
        vae_group.add_argument(
            "--beta_warmup",
            type=int,
            default=None,
            help="number of iterations of warmup before beta starts increasing",
        )
        return parser

    def forward(self, x):
        """ calculate the VAE ELBO """
        mu, logstd = self.encode_to_params(x)
        encoder_distribution = torch.distributions.Normal(
            loc=mu, scale=torch.exp(logstd)
        )
        z_sample = encoder_distribution.rsample()
        reconstruction_loss = self.decoder_loss(z_sample, x)

        # Manual formula for kl divergence (more numerically stable!)
        kl_div = 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1.0 - 2 * logstd)
        kl_loss = kl_div.sum() / z_sample.shape[0]

        # Final loss
        loss = reconstruction_loss + self.beta * kl_loss

        # Logging
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                reconstruction_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"kl/{self.logging_prefix}", kl_loss, prog_bar=self.log_progress_bar
            )
            self.log(f"loss/{self.logging_prefix}", loss)
        return loss

    def sample_prior(self, n_samples):
        return torch.distributions.Normal(self.prior_mu, self.prior_sigma).sample(
            torch.Size([n_samples])
        )

    def _increment_beta(self):

        if not self.beta_annealing:
            return

        # Check if the warmup is over and if it's the right step to increment beta
        if (
            self.global_step > self.hparams.beta_warmup
            and self.global_step % self.hparams.beta_step_freq == 0
        ):
            # Multiply beta to get beta proposal
            self.beta = min(self.hparams.beta_final, self.beta * self.hparams.beta_step)

    # Methods to overwrite (ones that differ between specific VAE implementations)
    def encode_to_params(self, x):
        """ encode a batch to it's distributional parameters """
        raise NotImplementedError

    def decoder_loss(self, z: torch.Tensor, x_orig) -> torch.Tensor:
        """ Get the loss of the decoder given a batch of z values to decode """
        raise NotImplementedError

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self._increment_beta()
        self.log("beta", self.beta, prog_bar=True)

        self.logging_prefix = "train"
        loss = self(batch[0])
        self.logging_prefix = None
        return loss

    def validation_step(self, batch, batch_idx):
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(batch[0])
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class UnFlatten(torch.nn.Module):
    """ unflattening layer """

    def __init__(self, filters=1, size=28):
        super().__init__()
        self.filters = filters
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.filters, self.size, self.size)
