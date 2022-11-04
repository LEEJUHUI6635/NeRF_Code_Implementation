from JH_data_loader import LLFF, Rays_DATALOADER
from JH_solver import Solver

import torch
from torch.backends import cudnn
import numpy as np
import argparse
import os

# Random seed 고정
torch.manual_seed(1234)
cudnn.deterministic = True
cudnn.benchmark = True

# Paser
parser = argparse.ArgumentParser(description='NeRF Implementation by JH')

# LLFF
parser.add_argument('--base_dir', type=str, default='./data/nerf_llff_data/fern')
parser.add_argument('--factor', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--ndc_space', type=bool, default=True)

# train
parser.add_argument('--resume_iters', type=int, default=None)
parser.add_argument('--mode', type=str, default='Train') # list 형식으로 만들기
parser.add_argument('--nb_epochs', type=int, default=200000) # 500000
parser.add_argument('--near', type=int, default=0)
parser.add_argument('--far', type=int, default=1)
parser.add_argument('--coarse_num', type=int, default=64)
parser.add_argument('--fine_num', type=int, default=64)
parser.add_argument('--L_pts', type=int, default=10)
parser.add_argument('--L_dirs', type=int, default=4)
parser.add_argument('--learning_rate', type=int, default=5e-4)
parser.add_argument('--sample_num', type=int, default=64)

# save path
parser.add_argument('--save_results_path', type=str, default='results/')
parser.add_argument('--save_train_path', type=str, default='results/train/')
parser.add_argument('--save_test_path', type=str, default='results/test/')
parser.add_argument('--save_model_path', type=str, default='models/')
parser.add_argument('--save_coarse_path', type=str, default='models/coarse/')
parser.add_argument('--save_fine_path', type=str, default='models/fine/')
config = parser.parse_args()

# make directories
if not os.path.exists(config.save_results_path):
    os.makedirs(config.save_results_path)
if not os.path.exists(config.save_train_path):
    os.makedirs(config.save_train_path)
if not os.path.exists(config.save_test_path):
    os.makedirs(config.save_test_path)
if not os.path.exists(config.save_model_path):
    os.makedirs(config.save_model_path)
if not os.path.exists(config.save_coarse_path):
    os.makedirs(config.save_coarse_path)
if not os.path.exists(config.save_fine_path):
    os.makedirs(config.save_fine_path)
    
# Dataset preprocessing
images, poses, bds, render_poses, i_val = LLFF(config.base_dir, config.factor).outputs()

# 아래의 것들은 JH_data_loader에 넣어버린다.
height = images.shape[1]
width = images.shape[2]
factor = 8
focal = 3260.526333 / factor
# print(focal) # 407.0
intrinsic = np.array([
            [focal, 0, 0.5*width], # 0.5*W = x축 방향의 주점
            [0, focal, 0.5*height], # 0.5*H = y축 방향의 주점
            [0, 0, 1]])
near = 1.
data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images, near, config.ndc_space, False, True).data_loader() # Train
val_data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images, near, config.ndc_space, False, False).data_loader() # Validation
# test_data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, render_poses, None, None, near, config.ndc_space, True, False).data_loader() # Test

# Train or Test
# config.mode를 나눌 필요 x -> Train 후의 Test는 진행되어야 하기 때문
if config.mode == 'Train': # Train + Validation
    Solver(data_loader, val_data_loader, config, i_val, height, width).train()