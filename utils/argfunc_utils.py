import torch.nn as nn
import datetime
import yaml
from collections import OrderedDict

def opt_parser(args):
    with open(f'my_model/specs/{args.task}/{args.model_name}.yml', mode='r') as f:
        Loader = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    for k,v in opt.items():
        if isinstance(v, str) and v.startswith('nn'):
            opt[k] = str2nn_module(v)
        if k=='mv_ver' and v==1:
            opt['exp_factor'] = None
            opt['expand_groups'] = None
    if 'gray' in args.target_mode:
        opt['in_chans'] = 1
        
    return opt

def ordered_yaml():
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader

def right_type(value, type_name):
    if 'nn.modules' in value:
        return str2nn_module(value)
    elif 'str' in type_name:
        return value
    elif 'int' in type_name:
        return int(value)
    elif 'float' in type_name:
        return float(value)
    elif 'tuple' in type_name:
        return str2tuple(value)
    elif 'bool' in type_name:
        return str2bool(value)
    elif 'None' in type_name:
        return None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2tuple(s):
    s = s.replace("(", "").replace(")", "")
    s = s.split(',')
    if len(s) >= 2:
        s = s[:-1] if s[1]=='' else s
    mapped_int = map(int, s)
    return tuple(mapped_int)

def str2nn_module(s):
    s = s.lower()
    if 'gelu' in s: return nn.GELU
    elif 'leaky' in s: return nn.LeakyReLU
    elif 'relu6' in s: return nn.ReLU6
    elif 'relu' in s: return nn.ReLU
    elif 'layernorm' in s: return nn.LayerNorm
    elif 'silu' in s: return nn.SiLU
    else: return None
    
def record_args_before_build(args):
    if args.load_model:
        assert args.model_time is not None, "'model_time' should be set!!"
        assert args.checkpoint_epoch != 0, "'checkpoint_epoch' should not be zero!!"
        with open(f'./args/args_{args.model_time}.txt', 'r') as f: 
            lines = f.readlines()
        for line in lines:
            if 'Opts' in line: break
            if line.startswith(('***', 'restart', '[changed]')): continue
            k,v,t = line.strip().split('===')
            if k not in args.__dict__: continue
            if args.__dict__[k] != right_type(v,t):
                while True:
                    changing = input(f"args '{k}' is not same. {v}(previous) {args.__dict__[k]}(current) change to previous? [y/n] ").lower()
                    if changing in ['y', 'n']: break
                    else: print("only enter 'y' or 'n'")
                if changing == 'y':
                    args.__dict__[k] = right_type(v,t)
                else:
                    with open(f'./args/args_{args.model_time}.txt', 'a') as f:
                        f.writelines(f'[changed] {k}==={args.__dict__[k]}==={type(v)}\n')
        retime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'./args/args_{args.model_time}.txt', 'a') as f:
            f.writelines(f'restart time: {retime}\n')
        with open(f'./logs/{args.model_time}_train.txt', 'a') as f:
            f.writelines(f'restart time: {retime}\n')
        with open(f'./logs/{args.model_time}_test.txt', 'a') as f:
            f.writelines(f'restart time: {retime}\n')
        print('==================================================================')
        print(f'train info already recorded ./args/args_{args.model_time}.txt')
        print(f'restart at {retime}')
        print('==================================================================')
    return args
        
def record_args_after_build(args, opts):
    if not args.load_model:
        assert args.model_time is None, "'model_time' should be None, if you train from scratch!!"
        assert args.checkpoint_epoch == 0, "'checkpoint_epoch' should be zero, if you train from scratch!!"
        args.model_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'./args/args_{args.model_time}.txt', 'w') as f:
            f.writelines('*******************Args*******************\n')
            for k, v in args.__dict__.items():
                f.writelines(f'{k}==={v}==={type(v)}\n')
            f.writelines('*******************Opts*******************\n')
            for k, v in opts.items():
                f.writelines(f'{k}==={v}==={type(v)}\n')
        print('==================================================================')
        print(f'train info record complete ./args/args_{args.model_time}.txt')
        print('==================================================================')
    return args, opts

def record_whole_finetune_time(args):
    wf_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'./logs/{args.model_time}_train.txt', 'a') as f:
        f.writelines(f'whole fine-tuning start: {wf_time}\n')
    with open(f'./logs/{args.model_time}_test.txt', 'a') as f:
        f.writelines(f'whole fine-tuning start: {wf_time}\n')