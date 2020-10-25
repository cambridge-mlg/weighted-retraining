# Experiment 1: weighted retraining with different parameters
# ==================================================
gpu="--gpu"  # change to "" if no GPU is to be used
root_dir="logs/opt/expr"
data_dir="assets/data/expr"
n_init_retrain_epochs=1
weight_type="rank"
lso_strategy="opt"

seed_array=( 1 2 3 )
n_data=5000
n_inducing_points=500
n_best_points=1000
n_rand_points=4000
n_decode_attempts=5
ignore_percentile=50
samples_per_model=1000
n_retrain_epochs=1
query_budget=500
test_dir=""

# Set specific experiments to do:
k_high="1e-1"
k_low="1e-3"
r_high=100
r_low=50
r_inf="1000000"  # Set to essentially be infinite (since "inf" is not supported)

# normal LSO, our setting, retrain only, weight only, high weight/low retrain, high retrain/low weight
k_expt=(  "inf"    "$k_low" "inf"    "$k_low" "$k_low"  "$k_high" )
r_expt=(  "$r_inf" "$r_low" "$r_low" "$r_inf" "$r_high" "$r_low" )

expt_index=0  # Track experiments
for seed in "${seed_array[@]}"; do
    for ((i=0;i<${#k_expt[@]};++i)); do

	# Increment experiment index
	expt_index=$((expt_index+1))

	# Break loop if using slurm and it's not the right task
	if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]; then
	    continue
	fi

	# Echo info of task to be executed
	r="${r_expt[$i]}"
	k="${k_expt[$i]}"
	echo "r=${r} k=${k} seed=${seed}"

	# Set pretrained model path
	if [[ "$k" ==  "inf" ]]; then
		start_model="assets/pretrained_models/expr/expr-k_inf.hdf5"
	elif [[ "$k" ==  "$k_high" ]]; then
		start_model="assets/pretrained_models/expr/expr-k_1e-1.hdf5"
	else
		start_model="assets/pretrained_models/expr/expr-k_1e-3.hdf5"
	fi

	# Run command
	python weighted_retraining/opt_scripts/opt_expr.py \
	    --seed="$seed" $gpu \
	    --query_budget="$query_budget" \
	    --retraining_frequency="$r" \
	    --result_root="${root_dir}${test_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
	    --data_dir="$data_dir" \
	    --pretrained_model_file="$start_model" \
	    --lso_strategy="$lso_strategy" \
	    --n_retrain_epochs="$n_retrain_epochs" \
	    --n_init_retrain_epochs="$n_init_retrain_epochs" \
	    --n_data="$n_data" \
	    --n_best_points="$n_best_points" --n_rand_points="$n_rand_points" \
	    --n_inducing_points="$n_inducing_points" \
	    --weight_type="$weight_type" --rank_weight_k="$k" \
	    --samples_per_model="$samples_per_model" \
	    --n_decode_attempts="$n_decode_attempts" \
	    --ignore_percentile="$ignore_percentile"
    done
done


# Experiment 2: baselines
# ==================================================
seed_array=( 1 2 3 )
query_budget=5000
n_retrain_epochs=1
n_init_retrain_epochs=0  # no need to retrain initially
lso_strategy="sample"
start_model="assets/pretrained_models/expr/expr-k_inf.hdf5"
n_data=5000

# List of hyperparameters to test!
r_list=( 200 500 )
quantile_list=( 0.8 0.95 )
dbas_noise_list=( 0.1 )
rwr_alpha_list=( "1e-1" "1.0" )

# Base command
base_command="python weighted_retraining/opt_scripts/opt_expr.py "\
"$gpu "\
"--query_budget=$query_budget "\
"--pretrained_model_file=$start_model "\
"--data_dir=$data_dir "\
"--n_retrain_epochs=$n_retrain_epochs "\
"--n_init_retrain_epochs=$n_init_retrain_epochs "\
"--lso_strategy=$lso_strategy "\
"--n_data=$n_data "\
"--n_best_points=2000 --n_rand_points=8000 "\
"--n_inducing_points=500 "\
"--samples_per_model=50 "\
"--n_decode_attempts=5 "\
"--ignore_percentile=50 "

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
    if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]; then
	continue
    fi

    # Run (or echo) command
    eval $cmd
    echo ""
done
