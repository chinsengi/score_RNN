import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import numpy as np
from utility import *
from runners import *
import json


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--runner', type=str, default='Celegans', help='the experiment runner to execute')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--hid_dim', type=int, default=5000, help='number of hidden units used by the model')
    parser.add_argument('--out_dim', type=int, default=300, help='dimension of output sample by the model')
    parser.add_argument('--resume', action='store_true', help='whether to train from the last checkpoint')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--run_id', type=str, default='0', help='id used to identify different runs')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--nepochs', type=int, default=400)
    parser.add_argument('--filter', type=str, default='pca', help='Different filters for MNIST runner: pca | sparse | ae | none')
    parser.add_argument('--sparse-weight-path', type=str, default='data/MNIST/sparse_weights/sparse_net.pth', \
                        help='path to sparse filter weights trained on MNIST')
    parser.add_argument('--ae-weight-path', type=str, default='data/MNIST/ae_weights/ae.pth', help='path to sparse filter weights trained on MNIST')
    parser.add_argument("--model", type=str, default="SR", help="model type: SR (Reservoir-sampler) |\
                         SO_FR (Sampler-only with firing rate dynamics) | SO_SC (Sampler-only with synaptic current dynamics)")
    parser.add_argument("--noise_level", type=int, default=10, help="number of noise steps")
    parser.add_argument("--nonlin", type=str, default="tanh", help="nonlinearity used")
    parser.add_argument('--test', action='store_true', help='specify to enable testing')
    parser.add_argument('--disable_impute', action='store_true', help='specify whether to impute missing data')
    parser.add_argument('--impute_freq', type=int, default=20, help='the frequency of imputing missing data')
    args = parser.parse_args()
    args.log = os.path.join(args.run, args.runner, 'logs', args.run_id)
    args.device = use_gpu()

    # specify logging configuration
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    if not args.resume and not args.test:
        if os.path.exists(args.log):
            shutil.rmtree(args.log)
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
 
    return args

def main():
    args = parse_args_and_config()
    print(f"Writing log file to {args.log}")
    logging.info(f"Exp instance id = {os.getpid()}")
    logging.info(f"Exp comment = {args.comment}")

    # set random seed
    if args.seed != 0:
        set_seed(args.seed)

    try:
        if not args.test:
            runner = eval(args.runner)(args)
            # print out the runner file   
            with open(os.path.join('runners', args.runner+'_runner.py'), 'r') as f:
                logging.info(f.read())
            # save the config file
            with open(os.path.join(args.log, 'config.yaml'), 'w') as f:
                yaml.dump(vars(args), f)
            logging.info(json.dumps(vars(args), indent=2))
            runner.train()
        else:
            # make sure that the config matches
            with open(f"run/{args.runner}/logs/{args.run_id}/config.yaml") as f:
                args = yaml.load(f, Loader=yaml.FullLoader)
                args = argparse.Namespace(**args)
            runner = eval(args.runner)(args)
            args.test = True
            runner.test()
    except:
        logging.error(traceback.format_exc())

if __name__=="__main__":
    sys.exit(main())

