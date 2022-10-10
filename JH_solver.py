from JH_model import Stratified_Sampling, Positional_Encoding, Hierarchical_Sampling, NeRF, viewing_directions 
import torch
import torch.nn as nn
import torch.nn.functional as F
# train, validation, test, Classic Volume Rendering 함수 or class + Data_loader 설정
import numpy as np

class solver(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader # rays dataloader
        self.nb_epochs = config.nb_epochs
        self.batch_size = config.batch_size
        self.sample_num = config.sample_num 
        self.near = config.near
        self.far = config.far
        self.L_pts = config.L_pts
        self.L_dirs = config.L_dirs
        
        self.classic_volume_rendering()
        self.train()
        # self.test()
    def classic_volume_rendering(self): # input -> / output -> 
        
        classic = 0
    def train(self):
        for epoch in range(self.nb_epochs):
            for idx, rays in enumerate(self.data_loader):
                # print(rays.shape) # [1024, 3, 3]
                
                # Stratified sampling + viewing_directions
                pts, z_vals = Stratified_Sampling(rays, self.batch_size, self.sample_num, self.near, self.far).outputs()
                # print(z_vals.shape) # [1024, 64]
                # print(pts.shape) # [65536, 3]
                dirs = viewing_directions(rays)
                # print(dirs.shape) # [1024, 3]
                # Positional Encoding + viewing_directions
                pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                dirs = Positional_Encoding(self.L_dirs).outputs(dirs) # viewing direction
                ###########################################################################
                ###########################################################################
                # debugging
                dirs = dirs[:,None,:]
                dirs = dirs.expand([self.batch_size, self.sample_num, dirs.shape[-1]])
                dirs = dirs.reshape(-1, dirs.shape[-1])
                ###########################################################################
                ###########################################################################
                # print(dirs.shape) # [65536, 27]
                # print(dirs.shape) # [1024, 27]
                # print(pts.shape) # [65536, 63]
                
                # pts + dirs -> [65536, 90]
                inputs = torch.cat([pts, dirs], dim=-1)
                # print(inputs.shape) # [65536, 90]
                # NeRF
                pts_channel = pts.shape[-1]
                output_channel = 4 # rgb + density
                dir_channel = dirs.shape[-1]
                model = NeRF(pts_channel, output_channel, dir_channel, self.batch_size, self.sample_num)
                inputs = np.array(inputs).astype(np.float32)
                outputs = model(inputs)
                # print(outputs.shape) # [1024, 64, 4]
                
                # classic volume rendering -> weights, z_vals ...
                
                
                
                
                # Hierarchical sampling + viewing_directions
                
                # Positional Encoding + viewing_directions
                
                # NeRF
                
                # classic volume rendering
                
                # loss -> reconstruction loss
                
                # optimizer
        x = 0    
        return x