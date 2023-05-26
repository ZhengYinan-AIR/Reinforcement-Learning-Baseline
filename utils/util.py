import argparse 
import os
import os.path as osp
import time
import torch
import pickle
import numpy as np
import json
from PIL import Image

DEFAULT_DATA_DIR = osp.join(osp.dirname(__file__),'result')

def evaluate(agent, env, environment, num_evaluation=10, max_steps=None):
    episode_rewards = []
    episode_costs = []
    if max_steps is None and environment == "safety-gym":
        max_steps = 1000
    assert max_steps != None

    o, r, d, ep_r, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0

    while n < num_evaluation:

        a = agent.step((np.array(o)).astype(np.float32))
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        info_keys = info.keys()
        goal_met = ('goal_met' in info_keys)
        d = d or goal_met # collision not done, reach goal done and terminal done
        violation = info['violation']
        
        ep_r += r
        ep_cost += violation
        ep_len += 1

        if ep_len == max_steps:
            episode_rewards.append(ep_r)
            episode_costs.append(ep_cost)
            if n < num_evaluation - 1:
                o, r, d, ep_r, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    return np.mean(episode_rewards), np.mean(episode_costs)

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=True):

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False

class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='result_logs.json', exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.output_fname = output_fname
        self.output_dir = output_dir

        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        
        print(colorize("Logging data to %s"%self.output_dir, 'green', bold=True))

        self.exp_name = exp_name
    
    def save_result_logs(self, result_logs):
        result = convert_json(result_logs)
        output = json.dumps(result, indent=4)
        with open(osp.join(self.output_dir, self.output_fname), 'w') as out:
            out.write(output)
        # f_save = open(osp.join(self.output_dir, self.output_fname), 'wb')
        # pickle.dump(result_logs, f_save)
        # f_save.close()

        print(colorize("Logging results to %s"%out.name, 'green', bold=True))

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name

        # output = json.dumps(config_json, separators=(',',':\t'), indent=4)
        output = json.dumps(config_json, indent=4)
        print(colorize('Saving config:', color='cyan', bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)
        print(colorize("Logging results to %s"%out.name, 'green', bold=True))

    def get_dir(self):
        return self.output_dir

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))
        
if __name__=='__main__':
    print(DEFAULT_DATA_DIR)
