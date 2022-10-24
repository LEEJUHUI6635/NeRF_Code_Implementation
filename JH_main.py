from JH_data_loader import LLFF, Rays_DATASET, Rays_DATALOADER
import numpy as np
import argparse
import os
import torch
from JH_solver import Solver
import random
from torch.backends import cudnn
torch.manual_seed(1234)
cudnn.deterministic = True
cudnn.benchmark = True
np.random.seed(1234)
random.seed(1234)

# Paser
parser = argparse.ArgumentParser(description='NeRF Implementation by JH')

# LLFF
parser.add_argument('--base_dir', type=str, default='./data/nerf_llff_data/fern')
parser.add_argument('--factor', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1024)

# train
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
parser.add_argument('--save_model', type=str, default='./models')
config = parser.parse_args()

# make directories
if not os.path.exists(config.save_model):
    os.makedirs(config.save_model)

# LLFF -> images, poses, bds, render_poses, i_val = LLFF(base_dir, factor).outputs()
images, poses, bds, render_poses, i_val = LLFF(config.base_dir, config.factor).outputs()

# print(images.shape) # [20, 378, 504, 3]
height = images.shape[1]
width = images.shape[2]
# print(height, width) # 378, 504
# print(poses.shape) # [20, 3, 5]
# focal = poses[0,2,-1]
# print(focal) # 407.0
factor = 8
focal = 3260.526333 / factor
# print(focal) # 407.0
intrinsic = np.array([
            [focal, 0, 0.5*width], # 0.5*W = x축 방향의 주점
            [0, focal, 0.5*height], # 0.5*H = y축 방향의 주점
            [0, 0, 1]])
near = 1.
ndc_space = True

data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images, near, ndc_space, False, True).data_loader()

# Train or Test
# config.mode를 나눌 필요 x -> Train 후의 Test는 진행되어야 하기 때문
if config.mode == 'Train':
    Solver(data_loader, None, config, i_val).train()
# elif config.mode == 'Test':
#     solver.test()