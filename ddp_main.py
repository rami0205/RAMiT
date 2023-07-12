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
    parser.add_argument('--gpus_per_node', default=2, type=int, help='number of gpus per node')
    parser.add_argument('--node_rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--ip_address', type=str, required=True, help='ip address of the host node')
    parser.add_argument('--backend', default='gloo', type=str, help='nccl or gloo')
    # fine-tuning
    parser.add_argument('--finetune', action='store_true', help='whether finetune or not (SR only)')
    parser.add_argument('--pretrain_path', type=str, help='pretrained model .pth file path')
    parser.add_argument('--warm_start', action='store_true', help='do or not warm-start when fine-tuning')
    parser.add_argument('--warm_start_epoch', type=int, help='how much warm-start before whole fine-tuning')
    # etc
    parser.add_argument('--model_time', type=str, help='automatically set when build model or manually set when load_model is True')
    parser.add_argument('--load_model', action='store_true', help='use checkpoint epoch or not')
    parser.add_argument('--checkpoint_epoch', type=int, default=0, help='restart train checkpoint')
    parser.add_argument('--seed', type=int, default=9725, help='random seed')
    parser.add_argument('--patch_load', action='store_true', help='random cropped patch load from pre-defined file')
    parser.add_argument('--task', type=str, default='lightweight_sr', help='lightweight_sr, lightweight_dn')
    # training data
    parser.add_argument('--data_name', type=str, default='DIV2K', help='training dataset. DIV2K, DFBW, LLE, DR, DF2K')
    parser.add_argument('--training_patch_size', type=str2tuple, default=(64,), help='LQ image patch size. model input patch size only for training. For DN, LLE, DR use (64,96,128)')
    # model type
    parser.add_argument('--model_name', type=str, default='RAMiT', help='model name to use. RAMiT, RAMiT-1, RAMiT-slimSR, RAMiT-LLE')
    parser.add_argument('--target_mode', type=str, default='light_x2', help='light_x2, light_x3, light_x4, light_dn, light_dn, light_graydn, light_lle, light_dr')
    parser.add_argument('--scale', type=int, help="upscale factor corresponding to 'target_mode'. it is automatically set.")
    parser.add_argument('--sigma', type=str2tuple, help='only for denosing tasks')
    # train / test spec
    parser.add_argument('--total_epochs', type=int, default=500, help='number of total epochs')
    parser.add_argument('--autocast', action='store_true', help='mixed precision with auto-casting')
    parser.add_argument('--test_only', action='store_true', help='only evaluate model. not train')
    parser.add_argument('--test_epoch', type=int, default=20, help='each epoch to run model evaluation')
    parser.add_argument('--result_image_save', action='store_true', default=True, help='save reconstructed image at test')
    # dataset / dataloader spec
    parser.add_argument('--img_norm', action='store_true', help="image normalization before input")
    parser.add_argument('--batch_size', type=str2tuple, default=(32,), help="mini-batch size per gpu. For DN, LLE, DR use (32,16,8)")
    parser.add_argument('--progressive_epoch', type=str2tuple, default=(0,), help="epochs for progressive learning. For DN, LLE, DR use (0,100,200)")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers in dataloader")
    parser.add_argument('--pin_memory', action='store_false', help="turn off pin-memory in dataloader")
    parser.add_argument('--record_iter', type=int, default=100, help='iteration to record history while training')
    # optimizer optim criterion spec
    parser.add_argument('--lrd', type=str, default='half', help="learning rate decay schedule")
    parser.add_argument('--min_lr', type=float, help='minimum learning rate only for cosine-decay')
    parser.add_argument('--warmup_epoch', type=int, default=20, help='warmup epoch')
    parser.add_argument('--accum_iter', type=int, default=1, help='full-batch size divided by mini-batch size')
    parser.add_argument('--lr_cycle', type=int, help='one frequency of cosine learning rate decay. it is automatically set.')
    parser.add_argument('--half_list', type=str2tuple, default=(200,300,400,425,450,475), help='epochs at which learning rate is reduced in half, if lrd is half. For SRx3x4 use (50,100,150,175,200,225). For DN, LLE, DR use (200,300,350,375)')
    parser.add_argument('--criterion', type=str, default='L1', help='loss function')
    # optimzier regularizer spec -> negative effects -> no use
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float, default=1.0, help='layer-wise learning rate decay. 1.0 means no lwd')
    parser.add_argument('--max_norm', type=float, help='clip grad max norm')
        
    return parser

def main(gpu, args, opts):
    
    rank = args.node_rank * args.gpus_per_node + gpu
    dist.init_process_group(
        backend=args.backend,
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    
    
    torch.manual_seed(args.seed)
    model, optimizer, loss_scaler = build_model_optimizer_scaler(gpu, args, opts)
    criterion = build_loss_func(args)
    
    if rank==0:
        args, opts = record_args_after_build(args, opts)
    if args.model_time is None:
        import time
        time.sleep(15)
        args.model_time = sorted(os.listdir('args'))[-1][5:-4]
    for epoch in range(args.total_epochs):
        if (epoch+1) <= args.checkpoint_epoch: continue
        if epoch in args.progressive_epoch or epoch+1 == args.checkpoint_epoch+1:
            args, train_loader, model, optimizer, loss_scaler = rebuild_progressive(gpu, rank, args, opts, epoch+1)
        if args.finetune and args.warm_start and (epoch+1)==args.warm_start_epoch+1:
            model, optimizer, loss_scaler = rebuild_after_warm_start(gpu, args, model)
            if rank==0: record_whole_finetune_time(args)
        train_loader.sampler.set_epoch(epoch)
        if not args.test_only:
            if args.patch_load:
                if args.task=='lightweight_sr':
                    if args.data_name == 'DIV2K':
                        plp = f'SR_patch_load/{str(epoch).zfill(3)}.pickle'
                    elif args.data_name == 'DF2K':
                        plp = f'SR_DF_patch/{str(epoch).zfill(3)}.pickle'
                elif args.target_mode=='light_dn':
                    plp = f'DN_patch_load/{str(epoch).zfill(3)}.pickle'
                elif args.target_mode=='light_graydn':
                    plp = f'GDN_patch_load/{str(epoch).zfill(3)}.pickle'
                elif args.target_mode=='light_lle':
                    plp = f'LLE_patch_load/{str(epoch).zfill(3)}.pickle'
                elif args.target_mode=='light_dr':
                    plp = f'DR_patch_load/{str(epoch).zfill(3)}.pickle'
                with open(plp, 'rb') as f:
                    train_loader.dataset.epoch_dict = pickle.load(f)
            train_one_epoch(gpu, rank, model, train_loader, optimizer, loss_scaler, criterion, epoch, args)
        if (epoch+1)%args.test_epoch==0 or (epoch+1)==args.total_epochs:
            test_one_epoch(rank, gpu, model, epoch, args, opts)
            
if __name__ == '__main__':
    os.makedirs(f'./args/', exist_ok=True)
    os.makedirs(f'./logs/', exist_ok=True)
    parser = get_args_parser()
    args = parser.parse_args()
    opts = opt_parser(args)
    
    args.scale = int(args.target_mode[-1]) if args.target_mode[-1].isdigit() else 1
    args.lr_cycle = args.warm_start_epoch if args.finetune and args.warm_start else args.lr_cycle
    
    args.world_size = args.gpus_per_node * args.total_nodes
    os.environ['MASTER_ADDR'] = args.ip_address
    os.environ['MASTER_PORT'] = '8888' if args.backend=='nccl' else '8989'
    if args.node_rank==0:
        args = record_args_before_build(args)
    mp.spawn(main, nprocs=args.gpus_per_node, args=(args, opts))