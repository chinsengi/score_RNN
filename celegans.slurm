#!/bin/bash

#SBATCH --job-name=celegans
#SBATCH --mail-type=END
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
# SBATCH --partition=gpu-rtx6k 
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm_out/slurm-%j.out

# TEST_ONLY=true
RUN="Celegans"
RUN_ID="2"
N_EPOCHS="50000"
# Your programs to run.
python main.py --runner $RUN --run_id $RUN_ID --nepochs $N_EPOCHS --nonlin relu --disable_impute
python main.py --runner $RUN --run_id $RUN_ID --test 
# wait for all programs to finish
wait
