#!/bin/bash

#SBATCH --job-name=gen_RNN
#SBATCH --mail-type=END
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k 
# SBATCH --partition=ckpt
#SBATCH --nodes=3
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

#SBATCH --chdir=.

TEST_ONLY=false
TEST_ONLY=true
RUN="DP"
# RUN="MNIST"
# Your programs to run.

declare -a run_ids=("2" "1" "0")
declare -a models=("SO_SC" "SO_FR" "SR")

if [ "$RUN" = "DP" ]; then
    echo "DP experiment"
    hid_dim=1000
    nepochs=1000
elif [ "$RUN" = "MNIST" ]; then
    echo "MNIST experiment"
    hid_dim=20000
    nepochs=1000
fi

if [ "$TEST_ONLY" = true ]; then
    echo "TEST_ONLY"
    for i in ${!run_ids[@]}; do
        srun --ntasks=1 --gres=gpu:1 python main.py --runner $RUN --run_id ${run_ids[$i]} --hid_dim $hid_dim --nepochs $nepochs --model ${models[$i]} --test &
    done
else
    echo "TRAINING AND TESTING"
    for i in ${!run_ids[@]}; do
        srun --ntasks=1 --gres=gpu:1 python main.py --runner $RUN --run_id ${run_ids[$i]} --hid_dim $hid_dim --nepochs $nepochs --model ${models[$i]} &
    done
    wait
    for i in ${!run_ids[@]}; do
        srun --ntasks=1 --gres=gpu:1 python main.py --runner $RUN --run_id ${run_ids[$i]} --hid_dim $hid_dim --nepochs $nepochs --model ${models[$i]} --test &
    done
fi
# wait for all programs to finish
wait
