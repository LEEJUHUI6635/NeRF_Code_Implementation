import numpy as np
import torch
import torch.nn as nn
from JH_data_loader import Rays_DATALOADER, Rays_DATASET, LLFF
# NeRF 네트워크 + Positional Encoding + Stratified sampling + Hierarchical sampling

base_dir = './data/nerf_llff_data/fern'
factor = 8
images, poses, bds, render_poses, i_val = LLFF(base_dir, factor).outputs()
# print(i_val) # 12
pts_channel = 63 # position
output_channel = 4 # rgb + density
dir_channel = 27 # viewing_dirs
batch_size = 1024
sample_num = 64 # stratified sampling
near = 0.
far = 1.
L_pts = 10
L_dirs = 4

height = 3024 // factor
width = 4032 // factor
focal = 3260.526333 // factor
# K = [3, 3] -> K[0][2] = 0.5*W, K[0][0] = focal, K[1][2] = 0.5*H, K[1][1] = focal
intrinsic = np.array([
            [focal, 0, 0.5*width], # 0.5*W = x축 방향의 주점
            [0, focal, 0.5*height], # 0.5*H = y축 방향의 주점
            [0, 0, 1]])

samples = Rays_DATASET(height, width, intrinsic, poses, i_val, images, near=1.0, ndc_space=True, train=True)

data_loader = Rays_DATALOADER(batch_size, height, width, intrinsic, poses, i_val, images).data_loader()

rays = None
for idx, samples in enumerate(data_loader):
    rays = samples
    break

z_vals = torch.ones([1024, 64])
weights = torch.rand(size=[1024, 64])
# print(z_vals.shape, weights.shape) # [1024, 64], [1024, 64]

# Stratified sampling
# in Dataloader -> rays = rays_o + rays_d + rays_rgb -> [1024, 3, 3] = [1024, 1, 3] + [1024, 2, 3] + [1024, 3, 3]
# pts = rays_o + rays_d * z_vals = [1024, 64, 3]
# near = 0, far = 1 -> [1024, 1]
# t_vals -> 0~1을 sample_num 개수만큼 동일한 간격으로 나눈다.
# *****Stratified sampling 수식 이해*****
class Stratified_Sampling(object):
    def __init__(self, rays, batch_size, sample_num, near, far):
        self.rays = rays # Rays_Dataloader에서 받아온 rays -> [1024, 3, 3]
        self.rays_o = self.rays[:,0:1,:]
        self.rays_d = self.rays[:,1:2,:]
        self.rays_rgb = self.rays[:,2:3,:]
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.near = torch.tensor(near) # 0 -> scalar
        self.far = torch.tensor(far) # 1 -> scalar
        self.z_sampling()
        # self.outputs()
        
        # *****NDC가 아닌 경우 -> z_sampling을 다르게 만들기(optional)*****
        # return z_vals
    def z_sampling(self): # pts = rays_o + rays_d + z_vals
        near = self.near.expand([self.batch_size, 1]) # 특정 크기의 array로 확장
        far = self.far.expand([self.batch_size, 1]) # 특정 크기의 array로 확장
        t_vals = torch.linspace(start=0., end=1., steps=self.sample_num) # 간격
        # print(t_vals.shape) # [64]
        z_vals = near + (far - near) * t_vals
        # print(z_vals.shape) # [1024, 64]
        mids = 0.5 * (z_vals[:,1:] + z_vals[:,:-1])
        # print(mids.shape) # [1024, 63]
        upper = torch.cat([mids, z_vals[:,-1:]], dim=-1)
        lower = torch.cat([z_vals[:,:1], mids], dim=-1)
        # print(upper.shape, lower.shape) # [1024, 64]
        t_vals = torch.rand(z_vals.shape)
        self.z_vals = lower + (upper - lower) * t_vals
        # print(self.z_vals.shape) # [1024, 64]
        
    def outputs(self):
        pts = self.rays_o + self.rays_d * self.z_vals[:,:,None]
        pts = pts.reshape(-1, 3)
        z_vals = self.z_vals
        return pts, z_vals # [1024, 64, 3]

# print(rays.shape) # [1024, 3, 3]

pts, z_vals = Stratified_Sampling(rays, batch_size, sample_num, near, far).outputs()
# print(pts.shape) # [1024, 64, 3] = [65536, 3]

def viewing_directions(rays):
    rays_d = rays[:,1,:] # [1024, 1, 3]
    dirs = rays_d / torch.norm(input=rays_d, dim=-1, keepdim=True)
    # print(dirs.shape)
    return dirs

dirs = viewing_directions(rays)
# print(dirs.shape) # [1024, 3]

# Positional Encoding
class Positional_Encoding(object): # shuffle x 
    def __init__(self, L):
        self.L = L # pts : L = 10 / dirs : L = 4
    def outputs(self, x):
        freq_bands = torch.linspace(start=0, end=self.L - 1, steps=self.L)
        # print(freq_bands) # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        freq = 2 ** freq_bands 
        # print(freq) # tensor([1., 2., 4., 8., 16., 32., 64., 128., 256., 512.])
        freq_list = []
        for i in freq:
            freq_list.append(torch.sin(x * i))
            freq_list.append(torch.cos(x * i))
        freq_arr = x
        for i in freq_list:
            freq_arr = torch.cat([freq_arr, i], dim=-1)
        # print(freq_arr.shape) # [65536, 63]
        return freq_arr
    
pts = Positional_Encoding(10).outputs(pts)

# Hierarchical sampling -> sample_num = 64
# *****Hierarchical sampling 과정 이해*****
# input -> Stratified sampling을 통해 얻은 z_vals와 weights
class Hierarchical_Sampling(object):
    def __init__(self, rays, z_vals, weights, batch_size, sample_num): # z_vals -> [1024, 64], weights -> [1024, 64]
        self.rays = rays # [1024, 3, 3] = [1024, 1, 3](rays_o) + [1024, 2, 3](rays_d) + [1024, 3, 3](rays_rgb)
        self.rays_o = self.rays[:,0:1,:]
        self.rays_d = self.rays[:,1:2,:]
        self.rays_rgb = self.rays[:,2:3,:]
        self.z_vals = z_vals
        self.weights = weights
        self.batch_size = batch_size
        self.sample_num = sample_num # fine sampling -> 64
        self.z_fine_sampling()
    def z_fine_sampling(self):
        # input -> Stratified sampling을 통해 추출한 z_vals와 weights
        # z_vals -> mids_z_vals = 0.5 * ([...,1:] + [...,:-1]) -> [1024, 63]
        # weights -> [1024, 64] -> weights[...,1:-1] -> [1024, 62]
        # weights = weights + 1e-5 -> 추후에 0으로 나누는 것을 방지
        # pdf -> weights를 정규화, 0 < pdf < 1
        # cdf -> torch.cumsum(pdf)
        # cdf = [0] + cdf -> [1024, 63]
        # random matrix u = [1024, 64] ~ [0, 1] uniform distribution
        mids = 0.5 * (self.z_vals[:,1:] + self.z_vals[:,:-1])
        # print(mids.shape) # [1024, 63]
        weights = self.weights[:,1:-1]
        # print(weights.shape) # [1024, 62]
        weights = weights + 1e-5 # 추후에 0으로 나뉘는 것을 방지
        pdf = weights / torch.norm(weights, dim=-1, keepdim=True)
        # print(pdf.shape) # [1024, 62]
        cdf = torch.cumsum(pdf, dim=-1)
        # print(cdf.shape) # [1024, 62]
        cdf = torch.cat([torch.zeros_like(cdf[:,:1]), cdf], dim=-1)
        # print(cdf.shape) # [1024, 63]
        # random matrix u
        u = torch.rand(size=[self.batch_size, self.sample_num])
        # print(u.shape) # [1024, 64]
        
        # inds -> a[i-1] < v < a[i] -> u의 값에 근사하는 값을 cdf matrix에서 찾아 index를 구한 matrix -> [1024, 64]
        # below -> inds - 1 = [1024, 64]
        # above -> inds = [1024, 64]

        # cdf_g -> inds에 대하여 indexing
        # bins_g -> inds에 대하여 indexing
        # indexing
        idx = torch.searchsorted(sorted_sequence=cdf, input=u, right=True)
        # print(idx.shape) # [1024, 64]
        below = torch.max(torch.zeros_like(idx-1), idx-1) # below = idx - 1
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(idx), idx) # above = idx
        # print(below.shape, above.shape) # [1024, 64]
        idx_ab = torch.stack([below, above], dim=-1) # index_above_below
        # print(idx_ab.shape) # [1024, 64, 2]
        
        mat_size = [idx_ab.shape[0], idx_ab.shape[1], cdf.shape[-1]]
        # torch.gather(input, dim, index)
        cdf_idx = torch.gather(input=cdf.unsqueeze(1).expand(mat_size), dim=-1, index=idx_ab)
        mids_idx = torch.gather(input=mids.unsqueeze(1).expand(mat_size), dim=-1, index=idx_ab)
        # print(cdf_idx.shape) # [1024, 64, 2]
        # print(mids_idx.shape) # [1024, 64, 2]
        # denorm = cdf_g(above) - cdf_g(below)
        # t = (u - cdf_g(below)) / denom
        # samples = bins_g(below) + t x (bins_g(above)-bins_g(below))
        # samples -> [1024, 64]
        denorm = cdf_idx[...,1] - cdf_idx[...,0]
        denorm = torch.where(denorm<1e-5, torch.ones_like(denorm), denorm)
        t = (u - cdf_idx[...,0]) / denorm
        z_fine_vals = mids_idx[...,0] + t * (mids_idx[...,1] - mids_idx[...,0])
        self.z_fine_vals = z_fine_vals.squeeze()
        # print(self.z_fine_vals.shape) # [1024, 64]
    def outputs(self):
        z_vals = torch.cat([self.z_vals, self.z_fine_vals], dim=-1) # [1024, 128]
        z_vals, _ = torch.sort(z_vals, dim=-1)
        fine_pts = self.rays_o + self.rays_d * z_vals[:,:,None]
        return fine_pts # [1024, 64+64, 3]

fine_pts = Hierarchical_Sampling(rays, z_vals, weights, batch_size, sample_num).outputs()
# print(fine_pts.shape) # [1024, 128, 3]

# input -> [65536, 90] = [1024 x 64, 20 x 3 + 3(position) + 8 x 3 + 3(direction)]
# pts -> [1024x64, 3], viewing_direction -> [1024x64, 3]
# positional encoding : pts -> [1024x64, 3 + 20 x 3], viewing_direction -> [1024x64, 3 + 8 x 3]
# output -> [1024, 64, 4] = [1024, 64, 3](rgb) + [1024, 64, 1](density)

# input_channel = 3 / output_channel = 4
# *****Viewing direction -> optional하게 만들기*****
class NeRF(nn.Module):
    def __init__(self, pts_channel, output_channel, dir_channel, batch_size, sample_num):
        super(NeRF, self).__init__()
        self.pts_channel = pts_channel # [x, y, z] points
        self.output_channel = output_channel
        self.hidden_channel = 256
        self.hidden2_channel = 128        
        self.dir_channel = dir_channel # viewing direction
        
        self.batch_size = batch_size # 1024
        self.sample_num = sample_num # 64
        
        # forward에서 쓰일 함수들
        self.density_outputs = nn.Linear(self.hidden_channel, 1)
        self.feature_outputs = nn.Linear(self.hidden_channel, self.hidden_channel)
        self.rgb_outputs = nn.Linear(self.hidden2_channel, 3)
        
        # forward에서 쓰일 block들 
        self.residual()
        self.density()
        self.rgb()
        
    def residual(self):
        self.residual_list = []
        self.residual_list.append(nn.Linear(in_features=self.pts_channel, out_features=self.hidden_channel))
        
        for i in range(3):
            self.residual_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel))
        # residual learning
        # [3, 256] -> [256, 256] -> [256, 256] -> [256, 256]
        self.residual_block = nn.Sequential(*self.residual_list)
        
    def density(self): # output -> density + 256 차원의 feature vector
        self.density_list = []
        self.density_list.append(nn.Linear(in_features=self.pts_channel+self.hidden_channel, out_features=self.hidden_channel))
        for i in range(3):
            self.density_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel))
            
        # density 출력하는 network
        # [256 + 3, 256] -> [256, 256] -> [256, 256] -> [256, 256]

        self.density_block = nn.Sequential(*self.density_list)
        # output -> 256 channel의 feature space + channel 1의 density
        
    def rgb(self):
        self.rgb_list = []
        self.rgb_list.append(nn.Linear(in_features=self.dir_channel+self.hidden_channel, out_features=self.hidden_channel))
        self.rgb_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden2_channel))

        # rgb 출력하는 network
        # [256 + 3, 256] -> [256, 128]
        # output -> 3 channel의 rgb
        self.rgb_block = nn.Sequential(*self.rgb_list)
        
    def forward(self, x): # x -> [1024x64, 90]
        # positional encoding -> [1024x64, 90] = [1024x64, 63](pts) + [1024x64, 27](viewing_dirs)
        x = torch.Tensor(x)
        # pts, dirs -> input x를 split한다.
        pts = x[:,:self.pts_channel]
        dirs = x[:,self.pts_channel:self.pts_channel+self.dir_channel]
        # print(pts.shape) # [65536, 63]
        # print(dirs.shape) # [65536, 27]
        # output -> channel 1의 density + channel 3의 rgb

        # Tensor 형식으로 -> numpy 대신 torch
        # residual_block을 거쳐 256 channel의 feature vector를 추출한다.
        feature = self.residual_block(pts) # linear
        # 256 channel의 feature vector와 3 channel의 pts를 concatenate한다.
        # print(feature.shape) # [65536, 256]
        feature = torch.cat([feature, pts], dim=1)
        # print(feature.shape) # [65536, 319]
        feature2 = self.density_block(feature)
        # print(feature2.shape) # [65536, 256]
        # feature2에서 하나의 layer를 더 거쳐서 feature vector와 1 channel의 density를 얻어내야 한다.
        density_outputs = self.density_outputs(feature2)
        feature_outputs = self.feature_outputs(feature2)
        # print(density_outputs.shape) # [65536, 1]

        # feature_outputs과 dirs를 concatenate한다. -> rgb layer에 집어 넣는다.
        feature3 = torch.cat([feature_outputs, dirs], dim=1)
        # print(feature3.shape) # [65536, 283]
        feature4 = self.rgb_block(feature3)
        # print(feature4.shape) # [65536, 128]
        rgb_outputs = self.rgb_outputs(feature4)
        # print(rgb_outputs.shape) # [65536, 3]
        # density_outputs과 rgb_outputs을 concatenate
        
        outputs = torch.cat([rgb_outputs, density_outputs], dim=1)
        # print(outputs.shape) # [65536, 4]
        outputs = outputs.reshape([self.batch_size, self.sample_num, -1])
        return outputs
    
model = NeRF(pts_channel, output_channel, dir_channel, batch_size, sample_num)
x = np.ones((65536, 90))
# print(x.shape)
outputs = model(x)
# print(outputs.shape) # [1024, 64, 4]