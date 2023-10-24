# Score-based generative model with RNN

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
conda env create -f environment.yml
```

### Usage
Run the following command to train the generative RNN:

```bash
python main.py --runner MNIST --hid_dim 20000 --nepochs 1000 --model SR
```

To test the planner, run the following command:

```bash
python main.py --runner MNIST --hid_dim 20000 --nepochs 1000 --model SR --test
```

For full help, run `python main.py -h`.
## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  chen2023expressive,
  title={Expressive probabilistic sampling in recurrent neural networks},
  author={Chen, Shirui and Jiang, Linxin Preston and Rao, Rajesh PN and Shea-Brown, Eric},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=ch1buUOGa3}
}
```

