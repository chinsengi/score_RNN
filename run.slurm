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
#SBATCH --output=./slurm_out/slurm-%j.out
TEST_ONLY=false
# TEST_ONLY=true
RUN="DP"
# RUN="MNIST"
# Your programs to run.

RUNID=3
declare -a models=("SR" "SO_SC" "SO_FR")
# declare -a run_ids=("4")
# declare -a models=("SR")

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
    for i in ${!models[@]}; do
        srun --ntasks=1 --gres=gpu:1 python main.py --runner $RUN --run_id $RUNID --test &
        ((RUNID++))
    done

else
    echo "TRAINING AND TESTING"
    for i in ${!models[@]}; do
        srun --ntasks=1 --gres=gpu:1 python main.py --runner $RUN --run_id $RUNID --hid_dim $hid_dim --nepochs $nepochs --model ${models[$i]} --nonlin tanh &
        ((RUNID++))
    done
    wait
    RUNID=$((RUNID-${#models[@]}))
    for i in ${!models[@]}; do
        srun --ntasks=1 --gres=gpu:1 python main.py --runner $RUN --run_id $RUNID --test &
        ((RUNID++))
    done
fi
# wait for all programs to finish
wait
