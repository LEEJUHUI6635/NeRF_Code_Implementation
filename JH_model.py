import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Stratified_Sampling(object):
    def __init__(self, rays_o, view_dirs, batch_size, sample_num, near, far, device):
        self.rays_o = rays_o
        self.rays_d = view_dirs
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.near = torch.tensor(near) # 0 -> scalar
        self.far = torch.tensor(far) # 1 -> scalar
        self.device = device
        self.z_sampling()
        # self.outputs()
        
        # *****NDC가 아닌 경우 -> z_sampling을 다르게 만들기(optional)*****
        # return z_vals
    def z_sampling(self): # pts = rays_o + rays_d + z_vals
        near = self.near.expand([self.batch_size, 1]) # 특정 크기의 array로 확장
        far = self.far.expand([self.batch_size, 1]) # 특정 크기의 array로 확장
        near = near.to(self.device)
        far = far.to(self.device)
        t_vals = torch.linspace(start=0., end=1., steps=self.sample_num) # 간격
        t_vals = t_vals.to(self.device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.to(self.device)
        mids = 0.5 * (z_vals[:,1:] + z_vals[:,:-1])
        mids = mids.to(self.device)
        upper = torch.cat([mids, z_vals[:,-1:]], dim=-1)
        upper = upper.to(self.device)
        lower = torch.cat([z_vals[:,:1], mids], dim=-1)
        lower = lower.to(self.device)
        t_vals = torch.rand(z_vals.shape).to(self.device)
        self.z_vals = lower + (upper - lower) * t_vals
        self.z_vals = self.z_vals.to(self.device)
        
    def outputs(self):
        self.rays_o = self.rays_o.to(self.device)
        self.rays_d = self.rays_d.to(self.device)
        self.z_vals = self.z_vals.to(self.device)
        
        # pts = self.rays_o + self.rays_d * self.z_vals[:,:,None]
        pts = self.rays_o[...,None,:] + self.rays_d[...,None,:] * self.z_vals[...,:,None] # [1024, 1, 3] + [1024, 1, 3] * [1024, 64, 1]
        pts = pts.reshape(-1, 3)
        z_vals = self.z_vals
        return pts, z_vals # [1024, 64, 3]

def viewing_directions(rays): # rays = [1024, 1, 3]
    rays_d = rays[:,0,:] 
    dirs = rays_d / torch.norm(input=rays_d, dim=-1, keepdim=True)
    return dirs

# Positional Encoding
class Positional_Encoding(object): # shuffle x 
    def __init__(self, L):
        self.L = L # pts : L = 10 / dirs : L = 4
    def outputs(self, x):
        freq_bands = torch.linspace(start=0, end=self.L - 1, steps=self.L)
        freq = 2 ** freq_bands 
        freq_list = []
        for i in freq:
            freq_list.append(torch.sin(x * i))
            freq_list.append(torch.cos(x * i))
        freq_arr = x
        for i in freq_list:
            freq_arr = torch.cat([freq_arr, i], dim=-1)
        return freq_arr

class Hierarchical_Sampling(object):
    def __init__(self, rays, z_vals, weights, batch_size, sample_num, device): # z_vals -> [1024, 64], weights -> [1024, 64]
        self.rays = rays # [1024, 3, 3] = [1024, 1, 3](rays_o) + [1024, 2, 3](rays_d) + [1024, 3, 3](rays_rgb)
        self.rays_o = self.rays[:,0:1,:]
        self.rays_d = self.rays[:,1:2,:]
        self.rays_rgb = self.rays[:,2:3,:]
        self.z_vals = z_vals
        self.weights = weights
        self.batch_size = batch_size
        self.sample_num = sample_num # fine sampling -> 64
        self.device = device
        self.z_fine_sampling()
    def z_fine_sampling(self):
        self.z_vals = self.z_vals.to(self.device)
        mids = 0.5 * (self.z_vals[:,1:] + self.z_vals[:,:-1])
        weights = self.weights[:,1:-1].to(self.device)
        weights = weights + 1e-5 # 추후에 0으로 나뉘는 것을 방지
        pdf = weights / torch.norm(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:,:1]), cdf], dim=-1)
        u = torch.rand(size=[self.batch_size, self.sample_num])
        u = torch.Tensor(u)
        u = u.to(self.device)
        cdf = cdf.to(self.device)
        idx = torch.searchsorted(sorted_sequence=cdf, input=u, right=True)
        below = torch.max(torch.zeros_like(idx-1), idx-1) # below = idx - 1
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(idx), idx) # above = idx
        idx_ab = torch.stack([below, above], dim=-1) # index_above_below
        
        mat_size = [idx_ab.shape[0], idx_ab.shape[1], cdf.shape[-1]]
        cdf_idx = torch.gather(input=cdf.unsqueeze(1).expand(mat_size), dim=-1, index=idx_ab)
        mids_idx = torch.gather(input=mids.unsqueeze(1).expand(mat_size), dim=-1, index=idx_ab)
        denorm = cdf_idx[...,1] - cdf_idx[...,0]
        denorm = torch.where(denorm<1e-5, torch.ones_like(denorm), denorm)
        t = (u - cdf_idx[...,0]) / denorm
        z_fine_vals = mids_idx[...,0] + t * (mids_idx[...,1] - mids_idx[...,0])
        self.z_fine_vals = z_fine_vals.squeeze()
    def outputs(self):
        z_vals = torch.cat([self.z_vals, self.z_fine_vals], dim=-1) # [1024, 128]
        z_vals, _ = torch.sort(z_vals, dim=-1) # sorting
        self.rays_o = self.rays_o.to(self.device)
        self.rays_d = self.rays_d.to(self.device)
        z_vals = z_vals.to(self.device)
        fine_pts = self.rays_o + self.rays_d * z_vals[:,:,None]
        fine_z_vals = z_vals
        return fine_pts, fine_z_vals # [1024, 64+64, 3]

# Debugging -> activation function을 추가해야 한다.
# input_channel = 3 / output_channel = 4
# *****Viewing direction -> optional하게 만들기*****
# model -> sample_num 따로 빼기 or forward() -> Coarse or Fine option
class NeRF(nn.Module):
    def __init__(self, pts_channel, output_channel, dir_channel, batch_size, sample_num, device):
        super(NeRF, self).__init__()
        self.pts_channel = pts_channel # [x, y, z] points -> 63
        self.output_channel = output_channel # rgb + density
        self.hidden_channel = 256 
        self.hidden2_channel = 128      
        self.dir_channel = dir_channel # viewing direction -> 27
        
        self.batch_size = batch_size # 1024
        self.sample_num = sample_num # 64
        self.device = device
        
        # forward에서 쓰일 함수들
        self.density_outputs = nn.Linear(self.hidden_channel, 1) # [256, 1]
        self.feature_outputs = nn.Linear(self.hidden_channel, self.hidden_channel) # [256, 256]
        self.rgb_outputs = nn.Linear(self.hidden2_channel, 3) # [128, 3]
        
        # forward에서 쓰일 block들 
        self.residual()
        self.density()
        self.rgb()
        
    def residual(self):
        self.residual_list = []
        self.residual_list.append(nn.Linear(in_features=self.pts_channel, out_features=self.hidden_channel)) # [63, 256]
        self.residual_list.append(nn.ReLU())
        
        for i in range(4):
            self.residual_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel)) # [256, 256]
            self.residual_list.append(nn.ReLU())
            
        # residual learning
        # [3, 256] -> [256, 256] -> [256, 256] -> [256, 256]
        self.residual_block = nn.Sequential(*self.residual_list)
        
    def density(self): # output -> density + 256 차원의 feature vector
        self.density_list = []
        self.density_list.append(nn.Linear(in_features=self.pts_channel+self.hidden_channel, out_features=self.hidden_channel)) # [63+256, 256]
        self.density_list.append(nn.ReLU())
        for i in range(3): # Q. 하나의 layer를 더 추가해야 하나?
            self.density_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel)) # [256, 256]
            self.density_list.append(nn.ReLU())
            
        # density 출력하는 network
        # [256 + 3, 256] -> [256, 256] -> [256, 256] -> [256, 256]

        self.density_block = nn.Sequential(*self.density_list)
        # output -> 256 channel의 feature space + channel 1의 density
        
    def rgb(self):
        self.rgb_list = []
        self.rgb_list.append(nn.Linear(in_features=self.dir_channel+self.hidden_channel, out_features=self.hidden_channel)) # [27+256, 256]
        self.rgb_list.append(nn.ReLU())
        self.rgb_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden2_channel)) # [256, 128]

        # rgb 출력하는 network
        # [256 + 3, 256] -> [256, 128]
        # output -> 3 channel의 rgb
        self.rgb_block = nn.Sequential(*self.rgb_list)
    
    def forward(self, x, sampling): # forward() : gradient의 학습을 결정 -> coarse와 fine을 한 개로 통일해야 한다.
        # coarse -> [65536, 90] / fine -> [131072, 90]
        if sampling.lower() == 'coarse':
            sample_num = 64 # 변수로 치환
        elif sampling.lower() == 'fine':
            sample_num = 128 # 변수로 치환
            
        pts = x[:,:self.pts_channel]
        dirs = x[:,self.pts_channel:self.pts_channel+self.dir_channel]
        feature = self.residual_block(pts)
        feature = torch.cat([pts, feature], dim=1)
        feature2 = self.density_block(feature)
        density_outputs = self.density_outputs(feature2)
        feature_outputs = self.feature_outputs(feature2)
        feature3 = torch.cat([feature_outputs, dirs], dim=1)
        feature4 = self.rgb_block(feature3)
        rgb_outputs = self.rgb_outputs(feature4)
        outputs = torch.cat([rgb_outputs, density_outputs], dim=1)
        outputs = outputs.reshape([x.shape[0] // sample_num, sample_num, self.output_channel]) # [1024, 64, 4]
        
        return outputs

# # Model
# class NeRF(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True): # use_viewdirs = True
#         """
#         """
#         super(NeRF, self).__init__()
#         self.D = D # 8 -> density를 뽑기 전 block의 개수
#         self.W = W # 256 -> node의 개수
#         self.input_ch = input_ch # 3 -> position [x, y, z]
#         self.input_ch_views = input_ch_views # 3 -> viewing direction [X, Y, Z] Cartesian unit vector
#         self.skips = skips # [4] for residual term
#         self.use_viewdirs = use_viewdirs # 실제 train -> True
        
#         # nn.ModuleList : nn.Sequential과 비슷하게, nn.Module의 list를 input으로 받는다. 하지만 nn.Sequential과 다르게 forward() method가 없다.

#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]) # 7
#             # i = 0, 1, 2, ..., 6 만일 i = 4 -> nn.Linear(256 + 3, 256)
#             # [3, 256] + [256, 256] + [256, 256] + [256, 256] + [256, 256] + [256 + 3, 256](skip connection) + [256, 256] + [256, 256] 

#         ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

#         self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)]) # [256 + 3, 128]
        
#         # [3 + 256, 128] -> direction + feature space

#         ### Implementation according to the paper
#         # self.views_linears = nn.ModuleList(
#         #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
#         if use_viewdirs: # viewing direction을 중간에 투입
#             self.feature_linear = nn.Linear(W, W) # [256, 256]
#             self.alpha_linear = nn.Linear(W, 1) # [256, 1] -> density 출력
#             self.rgb_linear = nn.Linear(W//2, 3) # [128, 3] -> RGB 출력
#         else:
#             self.output_linear = nn.Linear(W, output_ch) # viewing direction을 쓰지 않는다면, [256, 1+3] 한 번에 density와 RGB 출력

#     def forward(self, x):
#         # print(x.shape) # [65536, 90]
#         input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1) # tensor를 [self.input_ch, input_ch_views] size로 나누기
#         # input을 position과 viewing direction으로 나누기
#         h = input_pts # position
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h) # [256, 256]
#             h = F.relu(h)
#             if i in self.skips: # i = 4
#                 h = torch.cat([input_pts, h], -1) # skip connection -> dimension 합침

#         if self.use_viewdirs: # viewing direction = True
#             alpha = self.alpha_linear(h) # density 추출
#             feature = self.feature_linear(h) # feature vector 추출
#             h = torch.cat([feature, input_views], -1) # feature + viewing direction
        
#             for i, l in enumerate(self.views_linears):
#                 h = self.views_linears[i](h)
#                 h = F.relu(h)

#             rgb = self.rgb_linear(h)
#             outputs = torch.cat([rgb, alpha], -1) # outputs = rgb color + density
#         else:
#             outputs = self.output_linear(h)
#         # print(outputs.shape)
#         return outputs

#     # self.use_viewdirs = True
#     def load_weights_from_keras(self, weights):
#         assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
#         # assert [조건], [오류메시지] -> 조건 = True : 그대로 코드 진행, 조건 = False : 오류메시지 발생
#         # Load pts_linears
#         for i in range(self.D): # 8
#             idx_pts_linears = 2 * i # 0, 2, 4, 6, 8
#             self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
#             self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
#         # Load feature_linear
#         idx_feature_linear = 2 * self.D # 2 * 8
#         self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
#         self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

#         # Load views_linears
#         idx_views_linears = 2 * self.D + 2
#         self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
#         self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

#         # Load rgb_linear
#         idx_rbg_linear = 2 * self.D + 4
#         self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
#         self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

#         # Load alpha_linear
#         idx_alpha_linear = 2 * self.D + 6
#         self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
#         self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))