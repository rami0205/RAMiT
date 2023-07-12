import torch
from torchvision.transforms import functional as TF

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from utils import acc_utils
from dataset import make_some_noise

@torch.no_grad()
def test_one_epoch(rank, gpu, model, epoch, args, opts):
    model.eval()
    if args.task == 'lightweight_sr':
        default_dataset_list = ['Set5', 'Set14', 'BSDS100', 'urban100', 'manga109']
        if args.world_size==4:
            dataset_list = [default_dataset_list[rank+1]]
            if rank==0:
                dataset_list.insert(0, default_dataset_list[0])

        elif args.world_size==2:
            if rank==0:
                dataset_list = default_dataset_list[:4]
            else:
                dataset_list = default_dataset_list[4:]
    elif args.target_mode == 'light_dn':
        default_dataset_list = ['CBSD68', 'Kodak24', 'McMaster', 'urban100']
        if args.world_size==4:
            dataset_list = [default_dataset_list[rank]]

        elif args.world_size==2:
            if rank==0:
                dataset_list = default_dataset_list[:3]
            else:
                dataset_list = default_dataset_list[3:]
                
    elif args.target_mode == 'light_graydn':
        default_dataset_list = ['Set12', 'CBSD68', 'urban100']
        if args.world_size==4:
            if rank != 0:
                dataset_list = [default_dataset_list[rank-1]]
            else:
                dataset_list = ['Set12']

        elif args.world_size==2:
            if rank==0:
                dataset_list = default_dataset_list[:2]
            else:
                dataset_list = default_dataset_list[2:]
                
    elif args.target_mode == 'light_lle':
        default_dataset_list = ['LOL', 'VELOL-cap']
        if args.world_size==4:
            if rank < 2:
                dataset_list = default_dataset_list[:1]
            else:
                dataset_list = default_dataset_list[1:]

        elif args.world_size==2:
            if rank==0:
                dataset_list = default_dataset_list[:1]
            else:
                dataset_list = default_dataset_list[1:]
                
    elif args.target_mode == 'light_dr':
        default_dataset_list = ['Rain100H', 'Test100']
        if args.world_size==4:
            if rank < 2:
                dataset_list = default_dataset_list[:1]
            else:
                dataset_list = default_dataset_list[1:]

        elif args.world_size==2:
            if rank==0:
                dataset_list = default_dataset_list[:1]
            else:
                dataset_list = default_dataset_list[1:]
        
    if 'RAMiT' in args.model_name:
        min_multiple = (4*opts['window_size'], 4*opts['window_size'])
    dataset_list = default_dataset_list if args.world_size==1 else dataset_list
    
    for dd, dataset in enumerate(dataset_list):
        test_results = OrderedDict()
        if args.task=='lightweight_sr':
            folder_lq = f'../testsets/{dataset}/LR_bicubic/X{args.scale}/'
            folder_hq = f'../testsets/{dataset}/HR/'
            test_degrade = ('',)
            save_dir = f'results/{args.model_name}_{args.task}_x{args.scale}/{args.model_time}'
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []
            psnr_y, ssim_y = 0, 0
        elif args.task == 'lightweight_dn':
            folder_hq = f'../testsets/{dataset}/HQ/'
            test_degrade = (15,25,50)
            save_dir = f'results/{args.model_name}_{args.task}/{args.model_time}'
            test_results['psnr'] = []
            test_results['ssim'] = []
            psnr, ssim = 0, 0
        elif args.task == 'lightweight_lle':
            folder_lq = f'../testsets/{dataset}/LQ/'
            folder_hq = f'../testsets/{dataset}/HQ/'
            test_degrade = ('',)
            save_dir = f'results/{args.model_name}_{args.task}/{args.model_time}'
            test_results['psnr'] = []
            test_results['ssim'] = []
            psnr, ssim = 0, 0
        elif args.task == 'lightweight_dr':
            folder_lq = f'../testsets/{dataset}/LQ/'
            folder_hq = f'../testsets/{dataset}/HQ/'
            test_degrade = ('',)
            save_dir = f'results/{args.model_name}_{args.task}/{args.model_time}'
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []
            psnr_y, ssim_y = 0, 0
        border = args.scale if args.task == 'lightweight_sr' else 0
        
        # setup result dictionary
        os.makedirs(save_dir, exist_ok=True)
        
        path_list = sorted(glob.glob(os.path.join(folder_hq, '*.npy')))
        imgname_maxlen = max([len(os.path.splitext(os.path.basename(p))[0]) for p in path_list])
        
        for degrade in test_degrade:
            for data_iter, path in enumerate(tqdm(path_list)):
                # read image
                if 'gray' not in args.target_mode:
                    img_hq = torch.from_numpy(np.load(path))/255
                else:
                    img_hq = np.transpose(np.load(path), (1,2,0))
                    img_hq = cv2.cvtColor(img_hq, cv2.COLOR_RGB2GRAY)
                    img_hq = torch.from_numpy(img_hq).unsqueeze(0)/255
                imgname, imgext = os.path.splitext(os.path.basename(path))
                if args.task=='lightweight_sr':
                    path_lq = os.path.join(folder_lq, f'{imgname}x{args.scale}{imgext}')
                    img_lq = torch.from_numpy(np.load(path_lq)).unsqueeze(0)/255
                elif args.task=='lightweight_dn':
                    img_lq = torch.clone(img_hq)
                    img_lq = make_some_noise(img_lq, (degrade,), seed=0).unsqueeze(0)
                elif args.task in ['lightweight_lle', 'lightweight_dr']:
                    path_lq = os.path.join(folder_lq, f'{imgname}{imgext}')
                    img_lq = torch.from_numpy(np.load(path_lq)).unsqueeze(0)/255

                # inference
                # pad input image to be a multiple of window_size X final_patch_size
                _, _, lqh, lqw = img_lq.size()
                padw = min_multiple[1] - (lqw%min_multiple[1]) if lqw%min_multiple[1]!=0 else 0
                padh = min_multiple[0] - (lqh%min_multiple[0]) if lqh%min_multiple[0]!=0 else 0
                img_lq = TF.pad(img_lq, (0,0,padw,padh), padding_mode='symmetric')
                
                img_rc = model(img_lq.to(gpu))
                img_rc = img_rc[..., :lqh*args.scale, :lqw*args.scale]
                
                # save image
                img_rc = img_rc[0].detach().cpu().clamp(0,1).numpy()
                if 'gray' not in args.target_mode:
                    img_rc = np.transpose(img_rc[[2, 1, 0],:,:], (1, 2, 0)) if img_rc.ndim == 3 else img_rc # CHW-RGB to HWC-BGR
                else:
                    img_rc = np.transpose(img_rc, (1, 2, 0)) if img_rc.ndim == 3 else img_rc # CHW-RGB to HWC-BGR
                img_rc = (img_rc * 255.0).round().astype(np.uint8)  # float32 to uint8
                if args.result_image_save:
                    if args.task=='lightweight_sr':
                        cv2.imwrite(f'{save_dir}/{dataset}_{imgname}_x{args.scale}_{args.model_name}.png', img_rc)
                    elif args.task=='lightweight_dn':
                        cv2.imwrite(f'{save_dir}/{dataset}_{imgname}_sigma{degrade}_{args.model_name}.png', img_rc)
                    elif args.task in ['lightweight_lle', 'lightweight_dr']:
                        cv2.imwrite(f'{save_dir}/{dataset}_{imgname}_{args.model_name}.png', img_rc)

                # evaluate psnr/ssim
                if 'gray' not in args.target_mode:
                    img_hq = img_hq.permute(1,2,0)[:,:,[2,1,0]].numpy() # CHW-RGB to HWC-BGR
                else:
                    img_hq = img_hq.permute(1,2,0).numpy() # CHW-Gr to HWC-Gr (Gr: Gray)
                img_hq = (img_hq * 255.0).round().astype(np.uint8) # float32 to uint8
                img_hq = img_hq[:lqh*args.scale,:lqw*args.scale,:]  # crop HQ
                img_hq = np.squeeze(img_hq) if 'gray' not in args.target_mode else img_hq

                if args.task in ['lightweight_sr', 'lightweight_dr']:
                    psnr_y = acc_utils.calculate_psnr(img_rc, img_hq, crop_border=border, test_y_channel=True)
                    ssim_y = acc_utils.calculate_ssim(img_rc, img_hq, crop_border=border, test_y_channel=True)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                elif args.task in ['lightweight_dn', 'lightweight_lle']:
                    psnr = acc_utils.calculate_psnr(img_rc, img_hq, crop_border=border, test_y_channel=False)
                    ssim = acc_utils.calculate_ssim(img_rc, img_hq, crop_border=border, test_y_channel=False)
                    test_results['psnr'].append(psnr)
                    test_results['ssim'].append(ssim)
                deg = f'_{degrade}' if degrade!='' else ''
                with open(f'./logs/{args.model_time}_test_{dataset}{deg}.txt', 'a') as f:
                    if data_iter==0: f.writelines(f'[[{epoch+1}]]\n')
                    for _ in range(imgname_maxlen-len(imgname)): imgname+=' '
                    f.writelines(f'{dataset:10s} {data_iter+1:3d} {imgname} - ')
                    if args.task in ['lightweight_sr', 'lightweight_dr']:
                        f.writelines(f'PSNR_Y: {psnr_y:.2f}, SSIM_Y: {ssim_y:.4f}\n')
                    elif args.task in ['lightweight_dn', 'lightweight_lle']:
                        f.writelines(f'PSNR: {psnr:.2f}, SSIM: {ssim:.4f}\n')
                    if data_iter+1 == len(path_list): f.writelines('\n')

            # summarize psnr/ssim
            with open(f'./logs/{args.model_time}_test{deg}.txt', 'a') as f:
                if (rank==0 and dd==0) or (rank==1 and dd==0 and args.target_mode in ['light_graydn', 'light_lle', 'light_dr'] and args.world_size==4):
                    f.writelines(f'[[{epoch+1}]]\n')
                if args.task in ['lightweight_sr', 'lightweight_dr']:
                    avg_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    avg_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                    f.writelines(f'{dataset} {args.target_mode} - PSNR_Y/SSIM_Y: {avg_psnr_y:.2f}/{avg_ssim_y:.4f}\n')
                elif args.task in ['lightweight_dn', 'lightweight_lle']:
                    avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                    f.writelines(f'{dataset} {args.target_mode} - PSNR/SSIM: {avg_psnr:.2f}/{avg_ssim:.4f}\n')
                if rank==args.world_size-1 and dd+1==len(dataset_list):
                    f.writelines('\n')
            for k in test_results.keys():
                test_results[k] = []