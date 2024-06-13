# RAMiT

## Reciprocal Attention Mixing Transformer for Lightweight Image Restoration (CVPR 2024 Workshop NTIRE)
Haram Choi<sup>*</sup>, Cheolwoong Na, Jihyeon Oh, Seungjae Lee, Jinseop Kim, Subeen Choe, Jeongmin Lee, Taehoon Kim, and Jihoon Yang<sup>+</sup>

<sup>*</sup>: This work has been done during Master Course in Sogang University.

<sup>+</sup>: Corresponding author.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-00E600)](https://arxiv.org/abs/2305.11474)
[![paper](https://img.shields.io/badge/CVF-Paper-6196CA)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Choi_Reciprocal_Attention_Mixing_Transformer_for_Lightweight_Image_Restoration_CVPRW_2024_paper.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-E5B095)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/supplemental/Choi_Reciprocal_Attention_Mixing_CVPRW_2024_supplemental.pdf)
[![visual](https://img.shields.io/badge/Visual-Results-FF5050)](https://1drv.ms/f/s!AoUesdU_BVZri5xOqem2eCHKNxQ88Q)
[![poster](https://img.shields.io/badge/Presentation-Poster-B762C1)](https://1drv.ms/b/s!AoUesdU_BVZrjJwR8xL7GTv2--O3vg)

- Proposes RAMiT which employs Dimensional Reciprocal Attention Mixing Transformer (D-RAMiT) and Hierarchical Reciprocal Attention Mixer (H-RAMi)
- D-RAMiT: computing bi-dimensional self-attention in parallel to capture both local and global dependencies
- H-RAMi: using multi-scale attention for considering where and how much attention to pay semantically and globally
- Achieves state-of-the-art results on five lightweight image restoration tasks: Super-Resolution, Color Denoising, Grayscale Denoising, Low-Light Enhancement, Deraining

### News
-**June 07, 2024:** Presentation poster available.

-**April 17, 2024:** Accepted at CVPR 2024 Workshop [NTIRE](https://cvlai.net/ntire/2024/) (New Trends in Image Restoration and Enhancement)

-**July 12, 2023:** Codes released publicly

-**May 19, 2023:** Pre-printed at arXiv

### Model Architecture

<details>
<summary>Click</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/2da7a0a5-3935-41ad-afad-b88ffd09f1f9)

</details>

### Dimensional Reciprocal Self-Attentions

<details>
<summary>Click</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/5a9fd4ac-4610-443b-9633-5817e98ae5c5)

</details>

### Lightweight Image Restoration Results

<details>
<summary>Super-Resolution (SR)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/891646e4-2dd2-46b4-8a66-fcf9d5e32108)

</details>

<details>
<summary>slimSR</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/22a2c7fb-a9ec-4b46-8c8b-479e98a1d238)

</details>

<details>
<summary>SR trade-off</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/7a5584a5-e6ac-4a28-9f49-3105d5a28bb6)

</details>

<details>
<summary>Color Denoising (CDN)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/c857c150-6c7e-47ee-b316-ad42a490f346)

</details>

<details>
<summary>Grayscale Denoising (GDN)</summary>
  
![image](https://github.com/rami0205/RAMiT/assets/69415453/a5a94a5d-deee-4906-8fcb-0b10c5ea4623)

</details>

<details>
<summary>Low-Light Enhancement (LLE)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/57b04c41-b630-44e1-bf13-8b542666aef8)

</details>

<details>
<summary>Deraining (DR)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/d78eaec8-5eac-4024-875a-de7b614c0e0d)

</details>

### Visual Results

#### * The visual results on the other images can be downloaded in my [drive](https://1drv.ms/f/s!AoUesdU_BVZri5xOqem2eCHKNxQ88Q).

<details>
<summary>Super-Resolution (SR)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/71cf0ce5-2dad-49d7-af3d-c5a58ead2260)

</details>

<details>
<summary>Color Denoising (CDN)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/1f5eca0b-d7f3-4a77-9585-21e3678badbf)

</details>

<details>
<summary>Low-Light Enhancement (LLE)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/909a3084-ca9b-481e-b417-de530ec9f002)

</details>

<details>
<summary>Deraining (DR)</summary>

![image](https://github.com/rami0205/RAMiT/assets/69415453/1e605152-3ac5-4e70-998a-fbd561c73f2c)

</details>

## Testing Instructions (with pre-trained models)

Please properly edit the first five arguments to work on your devices.

#### RAMiT SR
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_X2.pth --task lightweight_sr --target_mode light_x2 --result_image_save --img_norm

python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_X3.pth --task lightweight_sr --target_mode light_x3 --result_image_save --img_norm

python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_X4.pth --task lightweight_sr --target_mode light_x4 --result_image_save --img_norm
```

#### RAMiT-1 SR
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-1 --pretrain_path ./pretrained/RAMiT-1_X2.pth --task lightweight_sr --target_mode light_x2 --result_image_save --img_norm

python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-1 --pretrain_path ./pretrained/RAMiT-1_X3.pth --task lightweight_sr --target_mode light_x3 --result_image_save --img_norm

python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-1 --pretrain_path ./pretrained/RAMiT-1_X4.pth --task lightweight_sr --target_mode light_x4 --result_image_save --img_norm
```

#### RAMiT-slimSR SR
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-slimSR --pretrain_path ./pretrained/RAMiT-slimSR_X2.pth --task lightweight_sr --target_mode light_x2 --result_image_save --img_norm

python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-slimSR --pretrain_path ./pretrained/RAMiT-slimSR_X3.pth --task lightweight_sr --target_mode light_x3 --result_image_save --img_norm

python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-slimSR --pretrain_path ./pretrained/RAMiT-slimSR_X4.pth --task lightweight_sr --target_mode light_x4 --result_image_save --img_norm
```

#### RAMiT CDN
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_CDN.pth --task lightweight_dn --target_mode light_dn --result_image_save --img_norm
```

#### RAMiT-1 CDN
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-1 --pretrain_path ./pretrained/RAMiT-1_CDN.pth --task lightweight_dn --target_mode light_dn --result_image_save --img_norm
```

#### RAMiT GDN
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_GDN.pth --task lightweight_dn --target_mode light_graydn --result_image_save --img_norm
```

#### RAMiT LLE
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_LLE.pth --task lightweight_lle --target_mode light_lle --result_image_save --img_norm
```

#### RAMiT-slimLLE LLE
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-slimLLE --pretrain_path ./pretrained/RAMiT-slimLLE_LLE.pth --task lightweight_lle --target_mode light_lle --result_image_save --img_norm
```

#### RAMiT DR
```
python3 ddp_main_test.py --total_nodes 1 --gpus_per_node 1 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --pretrain_path ./pretrained/RAMiT_DR.pth --task lightweight_dr --target_mode light_dr --result_image_save --img_norm
```

## Training Instructions

Please properly edit the first five arguments to work on your devices.

##### RAMiT SR
```
(x2) from scratch
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --target_mode light_x2 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 500 --half_list 200,300,400,425,450,475 --img_norm

(x3) warm-start
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --finetune --pretrain_path [pretrain PATH] --warm_start --warm_start_epoch 50 --model_name RAMiT --target_mode light_x3 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 300 --warmup_epoch 10 --half_list 50,100,150,175,200,225 --img_norm

(x4) warm-start
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --finetune --pretrain_path [pretrain PATH] --warm_start --warm_start_epoch 50 --model_name RAMiT --target_mode light_x4 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 300 --warmup_epoch 10 --half_list 50,100,150,175,200,225 --img_norm
```

#### RAMiT-1 SR
```
(x2) from scratch
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-1 --target_mode light_x2 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 500 --half_list 200,300,400,425,450,475 --img_norm

(x3) warm-start
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --finetune --pretrain_path [pretrain PATH] --warm_start --warm_start_epoch 50 --model_name RAMiT-1 --target_mode light_x3 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 300 --warmup_epoch 10 --half_list 50,100,150,175,200,225 --img_norm

(x4) warm-start
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --finetune --pretrain_path [pretrain PATH] --warm_start --warm_start_epoch 50 --model_name RAMiT-1 --target_mode light_x4 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 300 --warmup_epoch 10 --half_list 50,100,150,175,200,225 --img_norm
```

#### RAMiT-slimSR SR
```
(x2) from scratch
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-slimSR --target_mode light_x2 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 500 --half_list 200,300,400,425,450,475 --img_norm

(x3) warm-start
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --finetune --pretrain_path [pretrain PATH] --warm_start --warm_start_epoch 50 --model_name RAMiT-slimSR --target_mode light_x3 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 300 --warmup_epoch 10 --half_list 50,100,150,175,200,225 --img_norm

(x4) warm-start
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --finetune --pretrain_path [pretrain PATH] --warm_start --warm_start_epoch 50 --model_name RAMiT-slimSR --target_mode light_x4 --task lightweight_sr --training_patch_size 64 --batch_size 32 --progressive_epoch 0 --data_name DIV2K --total_epochs 300 --warmup_epoch 10 --half_list 50,100,150,175,200,225 --img_norm
```

#### RAMiT CDN (blind noise level)
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --target_mode light_dn --task lightweight_dn --sigma 0,50 --training_patch_size 64,96,128 --batch_size 32,16,8 --progressive_epoch 0,100,200 --data_name DFBW --total_epochs 400 --half_list 200,300,350,375 --img_norm
```

#### RAMiT-1 CDN (blind noise level)
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-1 --target_mode light_dn --task lightweight_dn --sigma 0,50 --training_patch_size 64,96,128 --batch_size 32,16,8 --progressive_epoch 0,100,200 --data_name DFBW --total_epochs 400 --half_list 200,300,350,375 --img_norm
```

#### RAMiT GDN (blind noise level)
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --target_mode light_graydn --task lightweight_dn --sigma 0,50 --training_patch_size 64,96,128 --batch_size 32,16,8 --progressive_epoch 0,100,200 --data_name DFBW --total_epochs 400 --half_list 200,300,350,375 --img_norm
```

#### RAMiT LLE
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --target_mode light_lle --task lightweight_lle --training_patch_size 64,96,128 --batch_size 32,16,8 --progressive_epoch 0,100,200 --data_name LLE --total_epochs 400 --half_list 200,300,350,375 --img_norm
```

#### RAMiT-slimLLE LLE
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT-slimLLE --target_mode light_lle --task lightweight_lle --training_patch_size 64,96,128 --batch_size 32,16,8 --progressive_epoch 0,100,200 --data_name LLE --total_epochs 400 --half_list 200,300,350,375 --img_norm
```

#### RAMiT DR
```
python3 ddp_main.py --total_nodes 1 --gpus_per_node 2 --node_rank 0 --ip_address [ip address XXX.XXX.XXX.XXX] --backend gloo --model_name RAMiT --target_mode light_dr --task lightweight_dr --training_patch_size 64,96,128 --batch_size 32,16,8 --progressive_epoch 0,100,200 --data_name DR --total_epochs 400 --half_list 200,300,350,375 --img_norm
```

### Citation
```
(preferred)
@inproceedings{choi2024reciprocal,
  title={Reciprocal Attention Mixing Transformer for Lightweight Image Restoration},
  author={Choi, Haram and Na, Cheolwoong and Oh, Jihyeon and Lee, Seungjae and Kim, Jinseop and Choe, Subeen and Lee, Jeongmin and Kim, Taehoon and Yang, Jihoon},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5992--6002}
  year={2024}
}

@article{choi2023reciprocal,
  title={Reciprocal Attention Mixing Transformer for Lightweight Image Restoration},
  author={Choi, Haram and Na, Cheolwoong and Oh, Jihyeon and Lee, Seungjae and Kim, Jinseop and Choe, Subeen and Lee, Jeongmin and Kim, Taehoon and Yang, Jihoon},
  journal={arXiv preprint arXiv:2305.11474},
  year={2023}
}
```

## My Related Works
- N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution, CVPR 2023. [![proceedings](https://img.shields.io/badge/Proceedings-FFB181)](https://openaccess.thecvf.com/content/CVPR2023/html/Choi_N-Gram_in_Swin_Transformers_for_Efficient_Lightweight_Image_Super-Resolution_CVPR_2023_paper.html) [![arXiv](https://img.shields.io/badge/arXiv-00E600)](https://arxiv.org/abs/2211.11436) [![code](https://img.shields.io/badge/Code-000000)](https://github.com/rami0205/NGramSwin)
- Exploration of Lightweight Single Image Denoising with Transformers and Truly Fair Training, ICMR 2023. [![proceedings](https://img.shields.io/badge/Proceedings-FFB181)](https://dl.acm.org/doi/abs/10.1145/3591106.3592265?casa_token=9RpLMzgnrZ0AAAAA:SJ0_WItZIKzxeXWEnwUkLesyK4hDQLAmybJgRIGqVYYpnjuxteT5W48Ega-8s-olD13jBz9G_UEQ0g) [![arXiv](https://img.shields.io/badge/arXiv-00E600)](https://arxiv.org/abs/2304.01805) [![code](https://img.shields.io/badge/Code-000000)](https://github.com/rami0205/LWDN)
