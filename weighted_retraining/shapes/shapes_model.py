""" Contains code for the shapes model """

import itertools
import numpy as np
import torch
from torch import nn, distributions
from torchvision.utils import make_grid

# My imports
from weighted_retraining.models import BaseVAE, UnFlatten


class ShapesVAE(BaseVAE):
    """ Convolutional VAE for encoding/decoding 64x64 images """

    def __init__(self, hparams):
        super().__init__(hparams)

        # Set up encoder and decoder
        self.encoder = nn.Sequential(
            # Many convolutions
            nn.Conv2d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            # Flatten and FC layers
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * self.latent_dim),
        )

        self.decoder = nn.Sequential(
            # FC layers
            nn.Linear(in_features=self.latent_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=256),
            nn.ReLU(),
            # Unflatten
            UnFlatten(16, 4),
            # Conv transpose layers
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=5,
                padding=2,
                stride=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
        )

    def encode_to_params(self, x):
        enc_output = self.encoder(x)
        mu, logstd = enc_output[:, : self.latent_dim], enc_output[:, self.latent_dim :]
        return mu, logstd

    def decoder_loss(self, z, x_orig):
        """ return negative Bernoulli log prob """
        logits = self.decoder(z)
        dist = distributions.Bernoulli(logits=logits)
        return -dist.log_prob(x_orig).sum() / z.shape[0]
    
    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        return torch.sigmoid(logits)

    def validation_step(self, *args, **kwargs):
        super().validation_step(*args, **kwargs)

        # Visualize latent space
        self.visualize_latent_space(20)

    def visualize_latent_space(self, nrow: int) -> torch.Tensor:

        # Currently only support 2D manifold visualization
        if self.latent_dim == 2:

            # Create latent manifold
            unit_line = np.linspace(-4, 4, nrow)
            latent_grid = list(itertools.product(unit_line, repeat=2))
            latent_grid = np.array(latent_grid, dtype=np.float32)
            z_manifold = torch.as_tensor(latent_grid, device=self.device)

            # Decode latent manifold
            with torch.no_grad():
                img = self.decode_deterministic(z_manifold).detach().cpu()
            img = torch.clamp(img, 0.0, 1.0)

            # Make grid
            img = make_grid(img, nrow=nrow, padding=5, pad_value=0.5)

            # Log image
            self.logger.experiment.add_image(
                "latent manifold", img, global_step=self.global_step
            )
