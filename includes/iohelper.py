import time
import os
import re
import argparse
from includes.dataset_helper import ConvLayer
import json

def setup_run(args, tag_name ='sparc'):
    '''
    Either load config from existing result directory if --load or create new directory for results.
    :param args: script input args
    :param tag_name: tag of current model
    :return:
    '''
    if args.load:
        run_tag = args.load
        print("Using data from {} run".format(run_tag))
    else:
        run_tag = str(round(time.time()))
        print("Starting new run: {} for {}".format(run_tag, tag_name))
    run_dir = 'result/' + tag_name + '/' + run_tag + '/'
    if not args.load:
        #os.mkdir(run_dir)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
    return run_tag, run_dir

def parse_args():
    '''
    Parse input arguments.
    :return: object with arguments
    '''
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--load", type=str, help="load vectors from previous run")
    arg_parser.add_argument("--task", type=str, help="load run config from task/ dir")
    args = arg_parser.parse_args()
    return args

def set_kfold(task):
    kfold_reg = re.search(r"run(\d+)$", task)
    if kfold_reg:
        k_fold = int(kfold_reg.group(1))
    else:
        k_fold = 0
    return k_fold


def process_config(run_tag, run_dir, args, rconf):
    '''
    Augument input config
    :param run_tag: tag for current run
    :param run_dir: dir for results and logs (timestamp)
    :param args: input arguments for script
    :param rconf: config to edit
    :return: void
    '''
    rconf['run_args']['run_config'] = [run_tag, run_dir]
    # back compatibility
    rconf['run_args']['tag_name'] = rconf['tag_name']
    rconf['run_args']['dataset_path'] = rconf['test_file_name']
    if 'inference_file_name' in rconf:
        # to preserve it in config.json with meaningful name
        rconf['run_args']['dataSetInference'] = rconf['inference_file_name']
    rconf['run_args']['task'] = args.task

    #convert conv layers
    if 'conv_layers' in rconf['run_args']:
        layers_struct = []
        for conv_layer in rconf['run_args']['conv_layers']:
            layers_struct.append(ConvLayer(wSize=conv_layer[0], ch=conv_layer[1], strd=conv_layer[2], mxpl=conv_layer[3]))
        rconf['run_args']['conv_layers'] = layers_struct

    rconf['run_args']['k_fold'] = set_kfold(args.task)

def load_config(args):
    '''
    Load config from jSON file
    :param args: input arguments object
    :return: json-object
    '''
    if args.task is None:
        raise Exception("Task is not set!")
    return json.load(open("tasks/{}.task".format(args.task)))

