import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
from utility import *
from runners import *


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--runner', type=str, default='Celegans', help='the experiment runner to execute')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')

    args = parser.parse_args()
    args.log = os.path.join(args.run, 'logs')
    args.device = use_gpu()
    return args

def main():
    args = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

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

