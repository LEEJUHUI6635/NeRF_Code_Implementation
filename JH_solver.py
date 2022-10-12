from JH_model import Stratified_Sampling, Positional_Encoding, Hierarchical_Sampling, NeRF, viewing_directions 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# train, validation, test, Classic Volume Rendering 함수 or class + Data_loader 설정
import numpy as np
import cv2 as cv
import os
##############################################################################
##############################################################################
##############################################################################
# device : cuda0

# Checkpoints 저장 -> epoch 마다 저장 or 마지막 epoch만 저장
class Save_Checkpoints(object):
    def __init__(self, epoch, model, optimizer, loss, save_path, select='epoch'): # select -> epoch or last
        # epoch, model_state_dict, optimizer_state_dict, loss 저장
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.save_path = save_path
        self.select = select
        # epoch 마다 저장 or 마지막 epoch만 저장 -> optional
        if self.select == 'epoch':
            self.save_checkpoints_epoch()
        elif self.select == 'last':
            self.save_checkpoints_last()
    def save_checkpoints_epoch(self):
        torch.save({'epoch': self.epoch, 
                    'model': self.model, 
                    'optimizer': self.optimizer, 
                    'loss': self.loss}, os.path.join(self.save_path, 'checkpoints_{}.pt'.format(self.epoch))) # self.save_path + 'checkpoints_{}.pt'
    def save_checkpoints_last(self):
        torch.save({'epoch': self.epoch,
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'loss': self.loss}, os.path.join(self.save_path, 'checkpoints_last.pt'))
        
# Train 자체가 잘 못 되었다. -> Train 시 rgb의 값 확인해 보기
class Solver(object):
    def __init__(self, data_loader, test_data_loader, config, i_val):
        self.data_loader = data_loader # rays dataloader
        self.test_data_loader = test_data_loader
        
        self.nb_epochs = config.nb_epochs
        self.batch_size = config.batch_size
        self.coarse_num = config.coarse_num # 64
        self.fine_num = config.fine_num # 64
        self.sample_num = self.coarse_num # 64
        self.near = config.near
        self.far = config.far
        self.L_pts = config.L_pts # 10
        self.L_dirs = config.L_dirs # 4
        self.learning_rate = config.learning_rate
        
        # pts_channel, output_channel, dir_channel 설정
        self.pts_channel = 3 + 2 * self.L_pts * 3 # 3 + 2 x 10 x 3
        self.output_channel = 4
        self.dir_channel = 3 + 2 * self.L_dirs * 3 # 3 + 2 x 4 x 3
        
        
        # save path
        self.save_model = config.save_model
        
        # validation
        self.i_val = i_val
        
        # self.classic_volume_rendering()
        self.basic_setting()
        # self.train()
    
    def basic_setting(self): # Q. 2개의 network를 학습?
        # model -> Coarse + Fine
        # Coarse + Fine Network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, self.batch_size, self.sample_num, self.device).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        
        # loss function 
        self.criterion = lambda x, y : torch.mean((x - y) ** 2)
        
    def classic_volume_rendering(self, raw, z_vals, rays, device): # input -> Network의 outputs [1024, 64, 4] + z_vals / output -> 2D color [1024, 3] -> rgb
        rays_d = rays[:,1:2,:]
        rays_d = rays_d.to(device)
        # print('rays_d : ', rays_d.shape) # [1024, 1, 3]
        # print('raw : ', raw.shape)
        rgb_3d = torch.sigmoid(raw[:,:,:3])
        rgb_3d = rgb_3d.to(device)
        ##########################################################
        ##########################################################
        ##########################################################
        # print('rgb_3d : ', rgb_3d.shape) # [1024, 128, 3] # debugging
        density = raw[:,:,3:]
        density = density.to(device)
        # print('density : ', density.shape) # [1024, 128, 1]
        # print('z_vals : ', z_vals.shape) # [1024, 128]
        dists = z_vals[:,1:] - z_vals[:,:-1] # [1024, 128]
        dists = dists.to(device)
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists.shape[0], 1).to(device)], dim=-1)
        # print('dists : ', dists.shape) # [1024, 1, 3]
        dists = dists * torch.norm(rays_d, dim=-1)
        # print('dists : ', dists.shape) # [1024, 128]
        active_func = nn.ReLU()
        noise = torch.randn_like(dists)
        # print('noise : ', noise.shape)
        alpha = 1 - torch.exp(-active_func((density.squeeze() + noise) * dists))
        # print('alpha : ', alpha.shape) # [1024, 64]
        transmittance = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(device), 1 - alpha + 1e-10], dim=-1), dim=-1)[:,:-1]
        # print('transmittance : ', transmittance.shape) # [1024, 64]
        weights = alpha * transmittance
        # print('weights : ', weights.shape) # [1024, 64]
        rgb_2d = torch.sum(weights[...,None] * rgb_3d, dim=-2)
        # print('rgb_2d : ', rgb_2d.shape) # [1024, 64]
        return rgb_2d, weights
    
    # def mapping(self, rgb_2d, weights, z_vals): # test image map 만들기?
    #     rgb_map = 0
    #     depth_map = 0
    #     disparity_map = 0
    #     accuracy_map = 0
    #     return rgb_map, depth_map, disparity_map, accuracy_map
    
    # 마지막으로 들어오는 batch의 size가 지정해 놓은 batch_size보다 작을 때 matching 되지 않는다는 문제가 발생한다.
    def train(self): # device -> dataset, model
        for epoch in range(self.nb_epochs):
            self.train_image_list = []
            for idx, rays in enumerate(self.data_loader):
                # print(rays) # torch.float64
                # float64를 float32로 변환해야 한다.
                rays = rays.to(torch.float32)
                # Coarse sampling과 Fine sampling을 번갈아 가면서 학습 진행
                # print(rays.shape) # [1024, 3, 3]
                batch_size = rays.shape[0] # 1024
                # print(batch_size)
                rays = rays.to(self.device)
                rays_rgb = rays[:,2:3,:]
                rays_rgb = rays_rgb.to(self.device)
                # Stratified sampling + viewing_directions
                pts, z_vals = Stratified_Sampling(rays, batch_size, self.sample_num, self.near, self.far, self.device).outputs()

                pts = pts.to(self.device)
                z_vals = z_vals.to(self.device)
                # print(z_vals.shape) # [1024, 64]
                # print(pts.shape) # [65536, 3]
                dirs = viewing_directions(rays)
                dirs = dirs.to(self.device)
                # print(dirs.shape) # [1024, 3]
                # Positional Encoding + viewing_directions
                pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                dirs = Positional_Encoding(self.L_dirs).outputs(dirs) # viewing direction
                pts = pts.to(self.device)
                dirs = dirs.to(self.device)
                ###########################################################################
                ###########################################################################
                # debugging
                dirs = dirs[:,None,:]
                dirs = dirs.expand([batch_size, self.sample_num, dirs.shape[-1]])
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
                # pts_channel = pts.shape[-1]
                # output_channel = 4 # rgb + density
                # dir_channel = dirs.shape[-1]
                # model = NeRF(pts_channel, output_channel, dir_channel, self.batch_size, self.sample_num)
                inputs = inputs.cpu().detach().numpy()
                # inputs = inputs.to(self.device)
                outputs = self.model(inputs, sampling='coarse')
                # print(outputs.shape) # [1024, 64, 4]
                outputs = outputs.to(self.device)
                z_vals = z_vals.to(self.device)
                rays = rays.to(self.device)

                # classic volume rendering -> weights, z_vals ...
                rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                rgb_2d = rgb_2d.to(self.device)
                weights = weights.to(self.device)
                ########################################################################################################################
                ########################################################################################################################
                ########################################################################################################################
                # Hierarchical sampling + viewing_directions
                # *****각 클래스에서의 sampling number를 어떻게 할 것인가?*****
                # print('Hierarchical sampling')
                fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                # print(fine_pts.shape) # [1024, 128, 3]
                fine_pts = fine_pts.to(self.device)
                # print(fine_pts.shape) # [1024, 128, 3]
                fine_z_vals = fine_z_vals.to(self.device)
                # print(fine_z_vals.shape) # [1024, 128]
                fine_dirs = viewing_directions(rays)
                fine_dirs = fine_dirs.to(self.device)
                # print(fine_dirs.shape) # [1024, 3]
                # Positional Encoding + viewing_directions
                fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                fine_dirs = Positional_Encoding(self.L_dirs).outputs(fine_dirs)
                # print(fine_pts.shape, fine_dirs.shape) # [1024, 128, 63], [1024, 27]
                fine_pts = fine_pts.to(self.device)
                fine_dirs = fine_dirs.to(self.device)
                
                # NeRF
                # pts_channel = fine_pts.shape[-1]
                # output_channel = 4
                # dir_channel = fine_dirs.shape[-1]
                # model = NeRF(pts_channel, output_channel, dir_channel, self.batch_size, self.sample_num * 2)
                
                fine_dirs = fine_dirs[:,None,:]
                fine_dirs = fine_dirs.expand([batch_size, self.sample_num * 2, fine_dirs.shape[-1]])
                fine_dirs = fine_dirs.reshape(-1, fine_dirs.shape[-1])
                # print(fine_dirs.shape) # [131072, 27]
                
                fine_inputs = torch.cat([fine_pts.reshape(-1, fine_pts.shape[-1]), fine_dirs], dim=-1)
                # print(fine_inputs.shape) # [131072, 90]
                fine_inputs = fine_inputs.cpu().detach().numpy()
                # fine_inputs = np.array(fine_inputs).astype(np.float32)
                
                fine_outputs = self.model(fine_inputs, sampling='fine')
                # print(fine_outputs.shape) # [1024, 128, 4]
                
                # print("Fine Sampling")
                
                # classic volume rendering
                fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                # print(fine_rgb_2d.shape, fine_weights.shape) # [1024, 1, 3], [1024, 128]
                #################################################################################################################################
                #################################################################################################################################
                #################################################################################################################################
                # loss -> reconstruction loss, rgb_2d - GT_rgb_2d
                # criterion = lambda x, y : torch.mean((x - y) ** 2)
                # Loss function = coarse sampling + fine sampling
                loss = self.criterion(fine_rgb_2d, rays_rgb) # Coarse + Fine
                print('rgb_2d : ', rgb_2d * 255)
                print('rays_rgb : ', rays_rgb * 255)
                print('fine_rgb_2d : ', fine_rgb_2d * 255)
            
                # optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(rays.shape) # [1024, 3, 3] 
                print(idx, loss)
                
                # 하나의 iteration이 끝남 -> 하나의 batch가 쌓여야 한다.
                rgb_2d = rgb_2d.cpu().detach().numpy()
                
                self.train_image_list.append(rgb_2d) # [1024, 3]
            
            # 하나의 epoch가 다 돌아서, iteration mode를 빠져나왔으면,
            self.train_image_arr = np.concatenate(self.train_image_list, axis=0)
            print('############################################################')
            print('train_image_arr.shape : ', self.train_image_arr.shape) # [전체 rays, 3]
            print('############################################################')
            self.train_image_arr = self.train_image_arr.reshape(20, 378, 504, 3)
            print(self.train_image_arr) # 0~1 사이의 값이 나와야 한다.
            print('############################################################')
            for i in range(20):
                image = self.train_image_arr[i,:,:,:]
                image = image * 255.
                print('pixel range : ', image)
                cv.imwrite('./data/LJH_epoch_{}.png'.format(epoch), image) # debugging
                            
            if epoch % 1 == 0 and epoch > 0:
                Save_Checkpoints(epoch, self.model, self.optimizer, loss, self.save_model, 'epoch')
                
            height = 378
            width = 504
            
            if epoch % 10 == 0 and epoch > 0:
                with torch.no_grad():
                    print("test")
                    rgb_list = []
                    for idx, rays in enumerate(self.test_data_loader):
                        rays = rays.to(self.device)
                        x = rays.shape[0]
                        print(x)
                        pts, z_vals = Stratified_Sampling(rays, x, self.sample_num, self.near, self.far, self.device).outputs()
                        # print(pts.shape) # [65536, 3] -> [batch_size(image 1개의 ray의 수) x image 개수, [x,y,z]]
                        # print(z_vals.shape) # [1024, 64] -> [batch_size, sample_num]
                        pts = pts.to(self.device)
                        z_vals = z_vals.to(self.device)
                        dirs = viewing_directions(rays)
                        dirs = dirs.to(self.device)
                        pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                        dirs = Positional_Encoding(self.L_dirs).outputs(dirs) # viewing direction
                        pts = pts.to(self.device)
                        dirs = dirs.to(self.device)
                        dirs = dirs[:,None,:]
                        dirs = dirs.expand([x, self.sample_num, dirs.shape[-1]])
                        dirs = dirs.reshape(-1, dirs.shape[-1])
                        inputs = torch.cat([pts, dirs], dim=-1)
                        inputs = inputs.cpu().detach().numpy()
                        # print("1")
                        outputs = self.model(inputs, sampling='coarse') # model의 batch_size에 영향을 끼친다.
                        # print("2")
                        outputs = outputs.to(self.device)
                        # print(outputs.shape) # [1024, 64, 4]
                        rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                        # print(rgb_2d.shape) # [1024, 3] -> 3 : [r,g,b]
                        
                        # Hierarchical sampling 추가

                        rgb_2d = rgb_2d.cpu().detach().numpy()
                        rgb_2d = rgb_2d * 255
                        if idx % 10 == 0:
                            print(rgb_2d)
                        # print(rgb_2d.shape) # [1024, 3]
                        rgb_list.append(rgb_2d)
                        print(idx)
                    print(len(rgb_list))
                    # print(len(rgb_list)) # 3
                    # print(rgb_list[0].shape) # [1024, 3]
                    rgb_arr = np.concatenate(rgb_list, axis=0)
                    # rgb_arr = np.array(rgb_list)
                    print(rgb_arr.shape) # [22860800, 3]
                    # rgb_arr = torch.stack(rgb_list, dim=0)
                    # print(rgb_arr.shape)
                    rgb_list = np.array_split(rgb_arr, 120, axis=0) # 120개의 image로 나눈다.
                    # rgb_list -> 120의 length, list 하나의 요소 -> [378x504]의 array로 나타낸다.
                    rgb_list = [image.reshape([height, width, 3]) for image in rgb_list]
                    for idx, image in enumerate(rgb_list):
                        cv.imwrite('./data/epoch{}_rendered{}.png'.format(epoch, idx), image)
                    print("nice job")