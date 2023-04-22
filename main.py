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
    parser.add_argument('--test', action='store_true', help='specify to enable testing')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--nepochs', type=int, default=400)
    parser.add_argument('--filter', type=str, default='pca', help='Different filters for MNIST runner: pca | sparse | none')
    parser.add_argument('--sparse-weight-path', type=str, default='data/MNIST/sparse_weights/sparse_net.pth', help='path to sparse filter weights trained on MNIST')
    args = parser.parse_args()
    args.log = os.path.join(args.run, args.runner, 'logs', args.run_id)
    args.device = use_gpu()

    # specify logging configuration
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    if not args.test:
        if not args.resume:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler2.setFormatter(formatter)
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        # logger.addHandler(h
        logger.addHandler(handler2)
        logger.setLevel(level)
    else:
        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)
 
    return args

def main():
    args = parse_args_and_config()
    print(f"Writing log file to {args.log}")
    logging.info(f"Exp instance id = {os.getpid()}")
    logging.info(f"Exp comment = {args.comment}")
    logging.info(args)

    # set random seed
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    try:
        runner = eval(args.runner)(args)
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        logging.error(traceback.format_exc())

if __name__=="__main__":
    sys.exit(main())

