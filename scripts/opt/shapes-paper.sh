# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed_array=( 1 2 3 4 5 )
root_dir="logs/opt/shapes"
start_model="assets/pretrained_models/shapes.ckpt"
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_bounds=3

# Experiment 1: weighted retraining with different parameters
# ==================================================
k_high=1e-1
k_low=1e-3
r_high=50
r_low=5
r_inf="1000000"  # Set to essentially be infinite (since "inf" is not supported)
weight_type="rank"
lso_strategy="opt"

# Set specific experiments to do:
# normal LSO, our setting, retrain only, weight only, high weight/low retrain, high retrain/low weight
k_expt=(  "inf"    "$k_low" "inf"    "$k_low" "$k_low"  "$k_high" )
r_expt=(  "$r_inf" "$r_low" "$r_low" "$r_inf" "$r_high" "$r_low" )

expt_index=0  # Track experiments
for seed in "${seed_array[@]}"; do
    for ((i=0;i<${#k_expt[@]};++i)); do

        # Increment experiment index
        expt_index=$((expt_index+1))

        # Break loop if using slurm and it's not the right task
        if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]
        then
            continue
        fi

        # Echo info
        r="${r_expt[$i]}"
        k="${k_expt[$i]}"
        echo "r=${r} k=${k} seed=${seed}"

        # Run the command
        python weighted_retraining/opt_scripts/opt_shapes.py \
            --seed="$seed" $gpu \
            --dataset_path=data/shapes/squares_G64_S1-20_seed0_R10_mnc32_mxc33.npz \
            --property_key=areas \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
            --pretrained_model_file="$start_model" \
            --weight_type="$weight_type" \
            --rank_weight_k="$k" \
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --opt_bounds="$opt_bounds" \
            --lso_strategy="$lso_strategy"
    done
done

# Experiment 2: various sampling baselines
# ==================================================
seed_array=( 1 2 3 )  # fewer seeds for these??
query_budget=5000  # way more samples
n_retrain_epochs=1  # Increase this due to less frequent training
n_init_retrain_epochs=1
lso_strategy="sample"

# List of hyperparameters to test!
r_list=( 200 500 )
quantile_list=( 0.8 0.95 )
dbas_noise_list=( 10 )  # based on max of ~100, max noise is 10%
rwr_alpha_list=( "1e-3" "1e-2" )  # Max is alpha * value ~= 1

# Base command
base_command="python weighted_retraining/opt_scripts/opt_shapes.py "\
"$gpu "\
"--dataset_path=data/shapes/squares_G64_S1-20_seed0_R10_mnc32_mxc33.npz "\
"--property_key=areas "\
"--query_budget=$query_budget "\
"--pretrained_model_file=$start_model "\
"--n_retrain_epochs=$n_retrain_epochs "\
"--n_init_retrain_epochs=$n_init_retrain_epochs "\
"--opt_bounds=$opt_bounds "\
"--lso_strategy=$lso_strategy "\
"--samples_per_model=50 "

# Big loop to gather all the experiments to run!
baseline_array=( )
for seed in "${seed_array[@]}"; do

    # Iterate over retraining frequencies
    for r in "${r_list[@]}"; do

        # Iterate over quantiles for those points that require it
        for quantile in "${quantile_list[@]}"; do

            # ####################
            # fb-vae strategy
            # ####################
            weight_type="fb"
            cmd="$base_command --seed=${seed} "\
"--weight_type=$weight_type --weight_quantile=$quantile "\
"--retraining_frequency=$r "\
"--result_root=${root_dir}/${weight_type}/q_${quantile}/r_${r}/seed${seed}"
            baseline_array+=("$cmd")

            # ####################
            # DbAS strategy
            # ####################
            weight_type="dbas"
            for dbas_noise in "${dbas_noise_list[@]}"; do

                cmd="$base_command --seed=${seed} "\
"--dbas_noise=$dbas_noise "\
"--weight_type=$weight_type --weight_quantile=$quantile "\
"--retraining_frequency=$r "\
"--result_root=${root_dir}/${weight_type}/q_${quantile}/r_${r}/n_${dbas_noise}/seed${seed}"
                baseline_array+=("$cmd")

            done


            # ####################
            # CEM-PI
            # ####################
            weight_type="cem-pi"
            cmd="$base_command --seed=${seed} "\
"--weight_type=$weight_type --weight_quantile=$quantile "\
"--retraining_frequency=$r "\
"--result_root=${root_dir}/${weight_type}/q_${quantile}/r_${r}/seed${seed}"
            baseline_array+=("$cmd")

        done


        # ####################
        # RWR strategy (doesn't use quantiles)
        # ####################
        weight_type="rwr"
        for alpha in "${rwr_alpha_list[@]}"; do
            cmd="$base_command --seed=${seed} "\
"--weight_type=$weight_type --rwr_alpha=$alpha "\
"--retraining_frequency=$r "\
"--result_root=${root_dir}/${weight_type}/r_${r}/alpha_${alpha}/seed${seed}"
            baseline_array+=("$cmd")
        done
    done
done

# Run all the various commands (possibly skipping using slurm)
for cmd in "${baseline_array[@]}"; do

    # Increment experiment index
    expt_index=$((expt_index+1))

    # Break loop if using slurm and it's not the right task
    if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]
    then
        continue
    fi

    # Run (or echo) command
    eval $cmd
    echo ""
done

echo "Final expt_index: $expt_index"