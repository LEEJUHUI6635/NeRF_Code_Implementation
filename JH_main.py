from JH_data_loader import LLFF, Rays_DATASET, Rays_DATALOADER
import numpy as np
import argparse

from JH_solver import solver

# Paser
parser = argparse.ArgumentParser(description='NeRF Implementation by JH')

# LLFF
parser.add_argument('--base_dir', type=str, default='./data/nerf_llff_data/fern')
parser.add_argument('--factor', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1024)

# train
parser.add_argument('--mode', type=str, default='Train') # list 형식으로 만들기
parser.add_argument('--nb_epochs', type=int, default=500000) # 500000
parser.add_argument('--near', type=int, default=0)
parser.add_argument('--far', type=int, default=1)
parser.add_argument('--sample_num', type=int, default=64)
parser.add_argument('--L_pts', type=int, default=10)
parser.add_argument('--L_dirs', type=int, default=4)

config = parser.parse_args()

# LLFF -> images, poses, bds, render_poses, i_val = LLFF(base_dir, factor).outputs()
images, poses, bds, render_poses, i_val = LLFF(config.base_dir, config.factor).outputs()

# print(images.shape) # [20, 378, 504, 3]
height = images.shape[1]
width = images.shape[2]
# print(height, width) # 378, 504
# print(poses.shape) # [20, 3, 5]
focal = poses[0,2,-1]
# print(focal) # 407.0
# focal = 3260.526333 // factor
# print(focal) # 407.0
intrinsic = np.array([
            [focal, 0, 0.5*width], # 0.5*W = x축 방향의 주점
            [0, focal, 0.5*height], # 0.5*H = y축 방향의 주점
            [0, 0, 1]])

# Dataloader -> rays
data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images).data_loader()

# Train or Test
if config.mode == 'Train':
    solver(data_loader, config).train()
# elif config.mode == 'Test':
#     solver.test()