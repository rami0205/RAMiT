import numpy as np
import torch

import os
import datetime
from tqdm import tqdm
import pickle

from utils.train_utils import cosine_learning_rate, half_learning_rate, keep_num_files

from collections import OrderedDict

def train_one_epoch(gpu, rank, model, data_loader, optimizer, loss_scaler, criterion, epoch, args):
    
    if rank == 0:
        with open(f'{args.model_time}_dataload.txt', 'a') as f:
            f.writelines(f'[[{str(epoch).zfill(3)}]]\n')
    epoch_zfill = len(str(args.total_epochs))
    iter_zfill = len(str(len(data_loader)))
    
    model.train()
    total_loss = 0
    inf_loss_count = 0
    if args.seed != 100:
        torch.random.manual_seed(epoch+args.seed)
        np.random.seed(epoch+args.seed)
    else:
        torch.random.manual_seed(epoch)
        np.random.seed(epoch)
    for data_iter, (img_hr, img_lr) in enumerate(tqdm(data_loader)):
        if data_iter%args.accum_iter==0:
            if args.lrd=='half':
                half_learning_rate(optimizer, epoch + data_iter/len(data_loader), args)
        
        img_hr = img_hr.to(gpu)
        img_lr = img_lr.to(gpu)
        
        if not args.autocast:
            img_sr = model(img_lr)
            loss = criterion(img_sr, img_hr)
        else:
            with torch.cuda.amp.autocast():
                img_sr = model(img_lr)
                loss = criterion(img_sr, img_hr)
                
        loss /= args.accum_iter
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter+1)%args.accum_iter==0)
        else:
            loss.backward()
            if (data_iter+1)%args.accum_iter==0:
                optimizer.step()
        
        if (data_iter+1)%args.accum_iter==0:
            optimizer.zero_grad()
            
        total_loss += loss.item()*args.accum_iter
        
        if rank==0 and (data_iter+1)%args.record_iter==0 and (data_iter+1)!=len(data_loader):
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(f'./logs/{args.model_time}_train.txt', 'a') as f:
                f.writelines(f'epoch: [{str(epoch+1).zfill(epoch_zfill)}/{args.total_epochs}], ')
                f.writelines(f'iter: [{str(data_iter+1).zfill(iter_zfill)}/{len(data_loader)}], ')
                f.writelines(f'loss: {total_loss/((data_iter+1-inf_loss_count)):.8f} {now}\n')
    
    # end of train one epoch
    if rank==0:
        avg_loss = total_loss/(len(data_loader)-inf_loss_count)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(f'./logs/{args.model_time}_train.txt', 'a') as f:
            f.writelines(f'epoch: [{str(epoch+1).zfill(epoch_zfill)}/{args.total_epochs}], ')
            f.writelines(f'iter: [{str(data_iter+1).zfill(iter_zfill)}/{len(data_loader)}], ')
            f.writelines(f'loss: {avg_loss:.8f} {now}\n')

        # model, optimizer, scaler state_dict SAVE
        sd_save_list = ['models', 'optims', 'scalers']
        for sd_save in sd_save_list: os.makedirs(f'./{sd_save}/{args.model_time}', exist_ok=True)
        sd = model.module.state_dict() if 'module' in model.__dir__() else model.state_dict()
        new_sd = OrderedDict([(n,p) for n,p in sd.items() if 'attn_mask' not in n])
        torch.save(new_sd, f'./models/{args.model_time}/model_{str(epoch+1).zfill(epoch_zfill)}.pth')
        torch.save(optimizer.state_dict(), f'./optims/{args.model_time}/optim_{str(epoch+1).zfill(epoch_zfill)}.pth')
        if loss_scaler is not None:
            torch.save(loss_scaler.state_dict(), f'./scalers/{args.model_time}/scaler_{str(epoch+1).zfill(epoch_zfill)}.pth')
        for sd_save in sd_save_list: keep_num_files(f'./{sd_save}/{args.model_time}', 'pth', 500)