""" Trains a convnet for the shapes task """
import argparse
import pytorch_lightning as pl

# My imports
from weighted_retraining.shapes.shapes_model import ShapesVAE
from weighted_retraining.shapes.shapes_data import WeightedNumpyDataset
from weighted_retraining import utils

if __name__ == "__main__":

    # Create arg parser
    parser = argparse.ArgumentParser()
    parser = ShapesVAE.add_model_specific_args(parser)
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    utils.add_default_trainer_args(parser, default_root="logs/train/ShapesVAE") 

    # Parse arguments
    hparams = parser.parse_args()
    pl.seed_everything(hparams.seed)

    # Create data
    datamodule = WeightedNumpyDataset(hparams, utils.DataWeighter(hparams))

    # Load model
    model = ShapesVAE(hparams)
    if hparams.load_from_checkpoint is not None:
        model = ShapesVAE.load_from_checkpoint(hparams.load_from_checkpoint)
        utils.update_hparams(hparams, model)

    # Main trainer
    trainer = pl.Trainer(
        gpus=1 if hparams.gpu else 0,
        default_root_dir=hparams.root_dir,
        max_epochs=hparams.max_epochs,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            period=10, monitor="loss/val", save_top_k=-1,
            save_last=True
        ),
        terminate_on_nan=True
    )

    # Fit
    trainer.fit(model, datamodule=datamodule)
    print("Training finished; end of script")
