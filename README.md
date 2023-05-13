# score-based generative model with RNN

Run different experiments with command

`python main.py --runner MNIST --run_id 0 --hid_dim 10000 --nepochs 100 --model SR`

## Hyak commands
`salloc -A amath -p gpu-rtx6k -N 1 -c 10 --mem=40G  --time=24:00:00 --gpus=1`
`squeue -u <net_id>`

