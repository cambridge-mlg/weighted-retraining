# Script to train all models for the paper

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"

# Train shapes VAE
python weighted_retraining/train_scripts/train_shapes.py \
    --root_dir="$root_dir/shapes" \
    --seed="$seed" $gpu \
    --latent_dim=2 \
    --dataset_path=data/shapes/squares_G64_S1-20_seed0_R10_mnc32_mxc33.npz \
    --property_key=areas \
    --max_epochs=20 \
    --beta_final=10.0 --beta_start=1e-6 \
    --beta_warmup=1000 --beta_step=1.1 --beta_step_freq=10 \
    --batch_size=16

# Train chem VAE
python weighted_retraining/train_scripts/train_chem.py \
    --root_dir="$root_dir/chem" \
    --seed="$seed" $gpu \
    --beta_final=0.005 --lr=0.0007 --latent_dim=56 \
    --max_epochs=30 --batch_size=32 \
    --train_path=data/chem/zinc/orig_model/tensors_train \
    --val_path=data/chem/zinc/orig_model/tensors_val \
    --vocab_file=data/chem/zinc/orig_model/vocab.txt \
    --property_file=data/chem/zinc/orig_model/pen_logP_all.pkl

# Train expr VAE
for k in "1e-3" "1e-1" "inf"; do
	python weighted_retraining/train_scripts/train_expr.py \
	    --root_dir="${root_dir}/expr" \
	    --seed="$seed" $gpu \
	    --k="$k"
done