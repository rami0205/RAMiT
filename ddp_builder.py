from dataset import DIV2KDatasetRandomCrop, DF2KDatasetRandomCrop, DFBWDatasetRandomCrop, LLEDatasetRandomCrop, DRDatasetRandomCrop
from torch.utils.data import DataLoader, DistributedSampler
from importlib import import_module
from utils.train_utils import param_groups_lrd, NativeScalerWithGradNormCount, CharbonnierLoss
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from timm.models.layers import _assert

def build_dataset(rank, ps, bs, args):
    if args.target_mode[-1].isdigit():
        if args.data_name == 'DIV2K':
            train_data = DIV2KDatasetRandomCrop('../DIV2K/DIV2K_train_HR',
                                                f'../DIV2K/DIV2K_train_LR_bicubic/X{args.scale}/',
                                                ps, args.model_time, args.patch_load)
        elif args.data_name == 'DF2K':
            train_data = DF2KDatasetRandomCrop('../DIV2K/DIV2K_train_HR', f'../DIV2K/DIV2K_train_LR_bicubic/X{args.scale}/',
                                               '../Flickr2K/Flickr2K_HR', f'../Flickr2K/Flickr2K_LR_bicubic/X{args.scale}',
                                               ps, args.model_time, args.patch_load)
    else:
        gray = False if 'gray' not in args.target_mode else True
        if args.data_name == 'DFBW':
            train_data = DFBWDatasetRandomCrop('../DIV2K/DIV2K_train_HR', '../Flickr2K/Flickr2K_HR',
                                               '../BSDS500/HQ', '../WED/HQ', ps, args.sigma, gray, args.model_time, args.patch_load)
        elif args.data_name == 'LLE':
            train_data = LLEDatasetRandomCrop('../LOL/HQ', '../VELOL/HQ', '../LOL/LQ', '../VELOL/LQ', 
                                              ps, args.model_time, args.patch_load)
        elif args.data_name == 'DR':
            train_data = DRDatasetRandomCrop('../Rain13K/HQ', '../Rain13K/LQ', ps, args.model_time, args.patch_load)
            
    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=bs, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
        
    return train_loader

def build_model_optimizer_scaler(gpu, args, opts):
    _assert(args.model_name in ['RAMiT', 'RAMiT-1', 'RAMiT-slimSR', 'RAMiT-slimLLE'],
            "'model_name' should be RAMiT, RAMiT-1, RAMiT-slimSR, RAMiT-slimLLE")
    if not args.finetune:
        _assert(args.pretrain_path is None and not args.warm_start and args.warm_start_epoch is None, 
                "Some arguments that must not be decided are assigned.")
    else:
        _assert(args.pretrain_path is not None, "--pretrain_path should be set for fine-tuning.")
    
    if '1' not in args.model_name:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')}")
    else:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')[:-2]}")
    model = module.make_model(args, opts, 0)
    
    # for warm-start and fine-tuning
    if args.finetune:
        if not args.load_model:
            sd = torch.load(args.pretrain_path, map_location='cpu')
            new_sd = OrderedDict()
            for n,p in sd.items():
                new_sd[n] = p if 'to_target' not in n else model.state_dict()[n]
            for n,p in model.state_dict().items():
                new_sd[n] = p if n not in new_sd else new_sd[n]
            print(model.load_state_dict(new_sd, strict=False))
        if args.warm_start and args.checkpoint_epoch <= args.warm_start_epoch:
            for n,p in model.named_parameters():
                p.requires_grad = False if 'to_target' not in n else True
            
    if args.load_model:
        sd = torch.load(f'./models/{args.model_time}/model_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        new_sd = OrderedDict([(n,p) for n,p in sd.items() if 'attn_mask' not in n])
        print(model.load_state_dict(new_sd, strict=False))
        
    model = model.to(gpu)
    model_no_ddp = model
    param_groups = param_groups_lrd(model_no_ddp, weight_decay=args.weight_decay,
                                    no_weight_decay_list=model_no_ddp.no_weight_decay(),
                                    layer_decay=args.layer_decay, model_name=args.model_name)
    model = DDP(model, device_ids=[gpu])
    
    args.init_lr = 0.0004 / 64 * args.batch_size[0]*args.world_size # any lr is okay. anyway newly defined at main function
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay) # lr anyway will be newly defined at lr decay function
    loss_scaler = NativeScalerWithGradNormCount() if args.autocast else None
    
    if args.load_model:
        osd = torch.load(f'./optims/{args.model_time}/optim_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        optimizer.load_state_dict(osd)
        if loss_scaler is not None:
            ssd = torch.load(f'./scalers/{args.model_time}/scaler_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
            loss_scaler.load_state_dict(ssd)
    
    return model, optimizer, loss_scaler

def rebuild_progressive(gpu, rank, args, opts, epoch):
    try: # pei means progressive-epoch-index
        pei = list(args.progressive_epoch).index(epoch-1)
    except:
        for pei, temp in enumerate(args.progressive_epoch+(args.total_epochs,)):
            if (epoch) < temp: break
        pei -= 1
    bs = args.batch_size[pei]
    ps = args.training_patch_size[pei]
    args.init_lr = 0.0004 / 64 * bs*args.world_size
    train_loader = build_dataset(rank, ps, bs, args)
    
    if '1' not in args.model_name:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')}")
    else:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')[:-2]}")
    model = module.make_model(args, opts, pei)
    
    # for warm-start and fine-tuning
    if args.finetune:
        if not args.load_model:
            sd = torch.load(args.pretrain_path, map_location='cpu')
            new_sd = OrderedDict()
            for n,p in sd.items():
                new_sd[n] = p if 'to_target' not in n else model.state_dict()[n]
            for n,p in model.state_dict().items():
                new_sd[n] = p if n not in new_sd else new_sd[n]
            print(model.load_state_dict(new_sd, strict=False))
    
    if epoch!=1:
        sd = torch.load(f'./models/{args.model_time}/model_{str(epoch-1).zfill(3)}.pth', map_location='cpu')
        new_sd = OrderedDict([(n,p) for n,p in sd.items() if 'attn_mask' not in n])
        model.load_state_dict(new_sd, strict=False)
        
    if args.finetune and args.warm_start and args.checkpoint_epoch <= args.warm_start_epoch:
        for n,p in model.named_parameters():
            p.requires_grad = False if 'to_target' not in n else True
    
    model = model.to(gpu)
    model_no_ddp = model
    param_groups = param_groups_lrd(model_no_ddp, weight_decay=args.weight_decay,
                                    no_weight_decay_list=model_no_ddp.no_weight_decay(),
                                    layer_decay=args.layer_decay, model_name=args.model_name)
    model = DDP(model, device_ids=[gpu])
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay) # lr anyway will be newly defined at lr decay function
    loss_scaler = NativeScalerWithGradNormCount() if args.autocast else None
    if epoch!=1:
        osd = torch.load(f'./optims/{args.model_time}/optim_{str(epoch-1).zfill(3)}.pth', map_location='cpu')
        optimizer.load_state_dict(osd)
        if loss_scaler is not None:
            ssd = torch.load(f'./scalers/{args.model_time}/scaler_{str(epoch-1).zfill(3)}.pth', map_location='cpu')
            loss_scaler.load_state_dict(ssd)
            
    print(f'progressive learning...epoch: {epoch}...patch-size: {ps}...batch-size: {bs*args.world_size}...init-lr:{args.init_lr}')
    return args, train_loader, model, optimizer, loss_scaler

def rebuild_after_warm_start(gpu, args, model):
    model = model.module.to(gpu)
    for n,p in model.named_parameters():
        p.requires_grad = True
    model_no_ddp = model
    param_groups = param_groups_lrd(model_no_ddp, weight_decay=args.weight_decay,
                                    no_weight_decay_list=model_no_ddp.no_weight_decay(),
                                    layer_decay=args.layer_decay, model_name=args.model_name)
    model = DDP(model, device_ids=[gpu])
    
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScalerWithGradNormCount() if args.autocast else None
    
    return model, optimizer, loss_scaler

def build_loss_func(args):
    _assert(args.criterion in ['L1', 'MSE', 'Charbonnier'], "'criterion' should be in [L1, MSE, Charbonnier]")
    criterion_dict = {'L1': nn.L1Loss(),
                      'MSE': nn.MSELoss(),
                      'Charbonnier': CharbonnierLoss()}
    criterion = criterion_dict[args.criterion]
    
    return criterion

def build_model_test(gpu, args, opts):
    _assert(args.model_name in ['RAMiT', 'RAMiT-1', 'RAMiT-slimSR', 'RAMiT-slimLLE'],
            "'model_name' should be RAMiT, RAMiT-1, RAMiT-slimSR, RAMiT-slimLLE")
    if '1' not in args.model_name:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')}")
    else:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')[:-2]}")
    model = module.make_model(args, opts, 0)
    sd = torch.load(args.pretrain_path, map_location='cpu')
    print(model.load_state_dict(sd, strict=False))
        
    model = model.to(gpu)
    model = DDP(model, device_ids=[gpu])
    
    return model