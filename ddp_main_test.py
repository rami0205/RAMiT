import argparse
from utils.argfunc_utils import *
from ddp_builder import *
import os
from ddp_engine_train import train_one_epoch
from ddp_engine_test import test_one_epoch
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle

def get_args_parser():
    parser = argparse.ArgumentParser('Reciprocal Attention Mixing Transformer for Lightweight Image Restoration', description='Reciprocal Attention Mixing Transformer for Lightweight Image Restoration', add_help=True)
    # ddp
    parser.add_argument('--total_nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus_per_node', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--node_rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--ip_address', type=str, required=True, help='ip address of the host node')
    parser.add_argument('--backend', default='gloo', type=str, help='nccl or gloo')
    
    parser.add_argument('--model_time', type=str, help='automatically set when build model')
    parser.add_argument('--model_name', type=str, default='RAMiT', help='model name to use. RAMiT, RAMiT-1, RAMiT-slimSR, RAMiT-LLE')
    parser.add_argument('--pretrain_path', type=str, help='pretrained model .pth file path')
    parser.add_argument('--task', type=str, default='lightweight_sr', help='lightweight_sr, lightweight_dn')
    parser.add_argument('--target_mode', type=str, default='light_x2', help='light_x2, light_x3, light_x4, light_dn, light_dn, light_graydn, light_lle, light_dr')
    parser.add_argument('--scale', type=int, help="upscale factor corresponding to 'target_mode'. it is automatically set.")
    parser.add_argument('--sigma', type=str2tuple, help='only for denosing tasks')
    parser.add_argument('--result_image_save', action='store_true', default=True, help='save reconstructed image at test')
    parser.add_argument('--img_norm', action='store_true', help="image normalization before input")
    
    return parser

def main(gpu, args, opts):
    
    rank = args.node_rank * args.gpus_per_node + gpu
    dist.init_process_group(
        backend=args.backend,
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
      
    model = build_model_test(gpu, args, opts)
    test_one_epoch(rank, gpu, model, 0, args, opts)
    
if __name__ == '__main__':
    os.makedirs(f'./logs/', exist_ok=True)
    parser = get_args_parser()
    args = parser.parse_args()
    opts = opt_parser(args)
    
    args.model_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.scale = int(args.target_mode[-1]) if args.target_mode[-1].isdigit() else 1
    
    args.world_size = args.gpus_per_node * args.total_nodes
    os.environ['MASTER_ADDR'] = args.ip_address
    os.environ['MASTER_PORT'] = '8888' if args.backend=='nccl' else '8989'
    mp.spawn(main, nprocs=args.gpus_per_node, args=(args, opts))