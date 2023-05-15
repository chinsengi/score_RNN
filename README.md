# score-based generative model with RNN

Run different experiments with command

`python main.py --runner MNIST --run_id 0 --hid_dim 20000 --nepochs 1000 --model SR`
`python main.py --runner DP --run_id 3 --hid_dim 1000 --nepochs 1000 --model SR`
`python main.py --runner DP --run_id 2 --hid_dim 1000 --nepochs 1000 --model SO_SC --test`
`python main.py --runner DP --run_id 1 --hid_dim 1000 --nepochs 1000 --model SO_FR --test`
`python main.py --runner DP --run_id 0 --hid_dim 1000 --nepochs 1000 --model SR --test`

## Hyak commands
`salloc -A amath -p gpu-rtx6k -N 1 -c 10 --mem=40G  --time=24:00:00 --gpus=1`
`squeue -u <net_id>`

