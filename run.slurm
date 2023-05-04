#!/bin/bash

#SBATCH --job-name=gen_RNN
#SBATCH --mail-type=END
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
# SBATCH --partition=gpu-rtx6k 
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1

#SBATCH --chdir=.

TEST_ONLY=false
TEST_ONLY=true
RUN="MNIST"
# RUN="DP"
# Your programs to run.
if [ "$RUN" = "DP" ]; then
    echo "DP experiment"
    if [ "$TEST_ONLY" = true ]; then
        echo "TEST_ONLY"
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 1 --hid_dim 500 --model SO_SC --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 1 --hid_dim 500 --model SO_FR --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 0 --hid_dim 500 --model SR --test &
    else
        echo "TRAINING AND TESTING"
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 1 --hid_dim 500 --nepochs 400 --model SO_SC &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 1 --hid_dim 500 --nepochs 400 --model SO_FR &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 0 --hid_dim 500 --nepochs 400 --model SR &
        wait
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 1 --hid_dim 500 --model SO_FR --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 1 --hid_dim 500 --model SO_SC --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner DP --run_id 0 --hid_dim 500 --model SR --test &
    fi
elif [ "$RUN" = "MNIST" ]; then
    echo "MNIST experiment"
    if [ "$TEST_ONLY" = true ]; then
        echo "TEST_ONLY"
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 1 --hid_dim 20000  --model SO_FR --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 1 --hid_dim 20000  --model SO_SC --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 0 --hid_dim 20000  --model SR --test &
    else
        echo "TRAINING AND TESTING"
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 1 --hid_dim 20000 --nepochs 1000 --model SO_FR &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 1 --hid_dim 20000 --nepochs 1000 --model SO_SC &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 0 --hid_dim 20000 --nepochs 1000 --model SR &
        wait
        # define dependencies for program 2 and 4
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 1 --hid_dim 20000 --model SO_FR --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 1 --hid_dim 20000 --model SO_SC --test &
        srun --ntasks=1 --gres=gpu:1 python main.py --runner MNIST --run_id 0 --hid_dim 20000 --model SR --test &

    fi
fi
# wait for all programs to finish
wait
