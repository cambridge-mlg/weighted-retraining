# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed_array=( 1 2 3 4 5 )
root_dir="logs/opt/chem"
start_model="assets/pretrained_models/chem.ckpt"
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1

# Experiment 1: weighted retraining with different parameters
# ==================================================
k_high="1e-1"
k_low="1e-3"
r_high=100
r_low=50
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


        # Echo info of task to be executed
        r="${r_expt[$i]}"
        k="${k_expt[$i]}"
        echo "r=${r} k=${k} seed=${seed}"

        # Run command
        python weighted_retraining/opt_scripts/opt_chem.py \
            --seed="$seed" $gpu \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
            --pretrained_model_file="$start_model" \
            --lso_strategy="$lso_strategy" \
            --train_path=data/chem/zinc/orig_model/tensors_train \
            --val_path=data/chem/zinc/orig_model/tensors_val \
            --vocab_file=data/chem/zinc/orig_model/vocab.txt \
            --property_file=data/chem/zinc/orig_model/pen_logP_all.pkl \
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --n_best_points=2000 --n_rand_points=8000 \
            --n_inducing_points=500 \
            --weight_type="$weight_type" --rank_weight_k="$k"

    done
done

# Experiment 2: baselines
# ==================================================
seed_array=( 1 2 3 )  # fewer seeds for these??
query_budget=5000  # way more samples
n_retrain_epochs=0.1
n_init_retrain_epochs=0  # no need to retrain initially
lso_strategy="sample"

# List of hyperparameters to test!
r_list=( 200 500 )
quantile_list=( 0.8 0.95 )
dbas_noise_list=( 0.1 )  # based on max of ~100, max noise is 10%
rwr_alpha_list=( "1e-1" "1.0" )  # Max is alpha * value ~= 1

# Base command
base_command="python weighted_retraining/opt_scripts/opt_chem.py "\
"$gpu "\
"--train_path=data/chem/zinc/orig_model/tensors_train "\
"--val_path=data/chem/zinc/orig_model/tensors_val "\
"--vocab_file=data/chem/zinc/orig_model/vocab.txt "\
"--property_file=data/chem/zinc/orig_model/pen_logP_all.pkl "\
"--query_budget=$query_budget "\
"--pretrained_model_file=$start_model "\
"--n_retrain_epochs=$n_retrain_epochs "\
"--n_init_retrain_epochs=$n_init_retrain_epochs "\
"--lso_strategy=$lso_strategy "\
"--n_best_points=2000 --n_rand_points=8000 "\
"--n_inducing_points=500 "\
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