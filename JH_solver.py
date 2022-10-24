from JH_model import Stratified_Sampling, Positional_Encoding, Hierarchical_Sampling, NeRF, viewing_directions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# train, validation, test, Classic Volume Rendering 함수 or class + Data_loader 설정
import numpy as np
import cv2 as cv
import os
import sys
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
    
    # 고민 : optimizer가 학습할 parameter -> 한 번에 이렇게 학습해도 되나?
    def basic_setting(self): # Q. 2개의 network를 학습?
        # model -> Coarse + Fine
        # Coarse + Fine Network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu' -> cpu로 처리해도 결과가 똑같이 나온다.
        self.coarse_model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, self.batch_size, self.sample_num, self.device).to(self.device)
        grad_variables = list(self.coarse_model.parameters()) # grad_variables = [self.coarse_model.parameters()]
        # self.model = NeRF(D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True).to(self.device)
        self.fine_model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, self.batch_size, self.sample_num, self.device).to(self.device)
        grad_variables += list(self.fine_model.parameters()) # grad_variables.append(self.fine_model.parameters())
        # optimizer
        self.optimizer = optim.Adam(params=grad_variables, lr=self.learning_rate, betas=(0.9, 0.999))
        # loss function
        self.criterion = lambda x, y : torch.mean((x - y) ** 2)

    # Classic Volume Rendering -> rays_d = rays_d
    def classic_volume_rendering(self, raw, z_vals, rays, device): # input -> Network의 outputs [1024, 64, 4] + z_vals / output -> 2D color [1024, 3] -> rgb
        rays_d = rays[:,1:2,:]
        raw = raw.cpu()
        z_vals = z_vals.cpu()
        rays = rays.cpu()
        rays_d = rays_d.cpu()
        # rays_d = torch.squeeze(rays_d)
        # print(rays_d.shape) # [1024, 3]
        rgb_3d = torch.sigmoid(raw[:,:,:3])
        # rgb_3d = rgb_3d.to(device)
        # ##########################################################
        # ##########################################################
        # ##########################################################
        # # print('rgb_3d : ', rgb_3d.shape) # [1024, 128, 3] # debugging
        density = raw[:,:,3:]
        # density = density.to(device)
        # print('density : ', density.shape) # [1024, 128, 1]
        # print('z_vals : ', z_vals.shape) # [1024, 128]
        dists = z_vals[:,1:] - z_vals[:,:-1] # [1024, 128]
        # dists = dists.to(device)
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists.shape[0], 1)], dim=-1)
        # print(dists.shape) # [1024, 64]
        # print('dists : ', dists.shape) # [1024, 1, 3]
        dists = dists * torch.norm(rays_d, dim=-1)
        # print('dists : ', dists.shape) # [1024, 128]
        active_func = nn.ReLU()
        noise = torch.randn_like(dists)
        # print('noise : ', noise.shape)
        alpha = 1 - torch.exp(-active_func((density.squeeze() + noise) * dists))
        # print('alpha : ', alpha.shape) # [1024, 64]
        transmittance = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1), 1 - alpha + 1e-10], dim=-1), dim=-1)[:,:-1]
        # print('transmittance : ', transmittance.shape) # [1024, 64]
        weights = alpha * transmittance
        # print('weights : ', weights.shape) # [1024, 64]
        rgb_2d = torch.sum(weights[...,None] * rgb_3d, dim=-2)
        return rgb_2d, weights
    # Depth_map, RGB_map
    
    # Colmap과 이미지와의 문제일까? -> 다른 데이터셋으로도 학습해 봐야 한다.
    
    # 마지막으로 들어오는 batch의 size가 지정해 놓은 batch_size보다 작을 때 matching 되지 않는다는 문제가 발생한다.
    def train(self): # device -> dataset, model
        for epoch in range(self.nb_epochs):
            self.train_image_list = []
            for idx, [rays, view_dirs] in enumerate(self.data_loader):
                batch_size = rays.shape[0]
                # view_dirs -> NDC 처리 전의 get_rays로부터
                view_dirs = viewing_directions(view_dirs) # [1024, 3]
                rays_o = rays[:,0,:]
                rays_d = rays[:,1,:]
                rays_rgb = rays[:,2,:] # True
                
                # Stratified Sampling -> rays_o + rays_d -> view_dirs x
                
                pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
                
                pts = pts.reshape(batch_size, 64, 3)
                coarse_view_dirs = view_dirs[:,None].expand(pts.shape)
                # print(view_dirs.shape) # [1024, 64, 3]
                pts = pts.reshape(-1, 3)
                coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                # Debugging -> OK
                
                coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
  
                # print(pts.shape, view_dirs.shape) # [65536, 63], [65536, 27]
                inputs = torch.cat([coarse_pts, coarse_view_dirs], dim=-1)
                inputs = inputs.to(self.device)
                
                # Network -> debugging
                torch.manual_seed(1234)
                outputs = self.coarse_model(inputs, sampling='coarse')
                outputs = outputs.reshape(rays.shape[0], 64, 4)
                rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                
                # Hierarchical sampling + viewing_directions
                # sampling의 단계에서는 inverse CDF의 방법으로 64개의 fine points를 sampling 한다.
                fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                fine_pts = fine_pts.reshape(batch_size, 128, 3) # [1024, 128, 3]

                # Positional Encoding + viewing_directions
                # Positional Encoding의 단계에서는 sampling된 fine points의 Positional Encoding을 취한다.
                # 고민 : Positional Encoding을 하기 전에 128개의 point로 합칠 것인가? 아니면 Network에 들어갔다 나온 후 합칠 것인가?
                # Positional Encoding을 하기 전 128개의 point로 합쳐 sorting 해야 한다.
                # fine_pts = torch.sort(torch.cat([coarse_pts, fine_pts], dim=1), dim=1) # [1024, 128, 3]
                
                fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape)
                fine_pts = fine_pts.reshape(-1, 3)
                fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)

                # Coarse Sampling 단계에서 이미 view_dirs의 Positional encoding을 처리했다.
                # 고민 : Network에 들어가기 전에 128개의 point로 합칠 것인가? 아니면 Network에 들어갔다 나온 후 합칠 것인가?
                # Network에 들어가기 전 128개의 point로 합쳐야 한다.
                fine_pts = fine_pts.to(self.device)
                fine_view_dirs = fine_view_dirs.to(self.device)
                fine_inputs = torch.cat([fine_pts, fine_view_dirs], dim=-1)
                fine_inputs = fine_inputs.to(self.device)
                
                fine_outputs = self.fine_model(fine_inputs, sampling='fine')
                fine_outputs = fine_outputs.reshape(rays.shape[0], 128, 4)

                # classic volume rendering
                fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                # print(fine_rgb_2d.shape, fine_weights.shape) # [1024, 1, 3], [1024, 128]
                #################################################################################################################################
                #################################################################################################################################
                #################################################################################################################################
                # loss -> reconstruction loss, rgb_2d - GT_rgb_2d
                # criterion = lambda x, y : torch.mean((x - y) ** 2)
                # Loss function = coarse sampling + fine sampling
                # rgb_2d = rgb_2d.to(self.device)
                loss = self.criterion(fine_rgb_2d, rays_rgb) # Coarse + Fine

                # optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(rays.shape) # [1024, 3, 3] 
                print(idx, loss)
                
                # # 하나의 iteration이 끝남 -> 하나의 batch가 쌓여야 한다.
                fine_rgb_2d = fine_rgb_2d.cpu().detach().numpy()
                # print(type(rgb_2d))
                fine_rgb_2d = (255*np.clip(fine_rgb_2d,0,1)).astype(np.uint8)

                self.train_image_list.append(fine_rgb_2d) # [1024, 3]
                # print(self.train_image_list[-1].shape) # [1024, 3] + [864, 3]
                # rays_rgb = rays_rgb.cpu().detach().numpy()
                # self.train_image_list.append(rays_rgb)
            # print(len(self.train_image_list))
            # 하나의 epoch가 다 돌아서, iteration mode를 빠져나왔으면,
            self.train_image_arr = np.concatenate(self.train_image_list, axis=0)
            print('############################################################')
            # print('train_image_arr.shape : ', self.train_image_arr.shape) # [전체 rays, 3]
            print('############################################################')
            self.train_image_arr = self.train_image_arr.reshape(18, 378, 504, 3) # image의 개수
            print(self.train_image_arr) # 0~1 사이의 값이 나와야 한다.
            print('############################################################')
            for i in range(1): # 첫 번째 이미지만을 출력
                image = self.train_image_arr[i,:,:,:]
                # image = image * 255.
                print('pixel range : ', image)
                cv.imwrite('./data/Hierarchical_sampling_{}.png'.format(epoch), image) # debugging

            if epoch % 10 == 0 and epoch > 0:
                Save_Checkpoints(epoch, self.coarse_model, self.optimizer, loss, os.path.join(self.save_model, 'coarse'), 'epoch')
                Save_Checkpoints(epoch, self.fine_model, self.optimizer, loss, os.path.join(self.save_model, 'fine'), 'epoch')

            # height = 378
            # width = 504
            
            # if epoch % 10 == 0 and epoch > 0:
            #     with torch.no_grad():
            #         print("test")
            #         rgb_list = []
            #         for idx, rays in enumerate(self.test_data_loader):
            #             rays = rays.to(self.device)
            #             x = rays.shape[0]
            #             print(x)
            #             pts, z_vals = Stratified_Sampling(rays, x, self.sample_num, self.near, self.far, self.device).outputs()
            #             # print(pts.shape) # [65536, 3] -> [batch_size(image 1개의 ray의 수) x image 개수, [x,y,z]]
            #             # print(z_vals.shape) # [1024, 64] -> [batch_size, sample_num]
            #             pts = pts.to(self.device)
            #             z_vals = z_vals.to(self.device)
            #             dirs = viewing_directions(rays)
            #             dirs = dirs.to(self.device)
            #             pts = Positional_Encoding(self.L_pts).outputs(pts) # position
            #             dirs = Positional_Encoding(self.L_dirs).outputs(dirs) # viewing direction
            #             pts = pts.to(self.device)
            #             dirs = dirs.to(self.device)
            #             dirs = dirs[:,None,:]
            #             dirs = dirs.expand([x, self.sample_num, dirs.shape[-1]])
            #             dirs = dirs.reshape(-1, dirs.shape[-1])
            #             inputs = torch.cat([pts, dirs], dim=-1)
            #             inputs = inputs.cpu().detach().numpy()
            #             # print("1")
            #             outputs = self.model(inputs, sampling='coarse') # model의 batch_size에 영향을 끼친다.
            #             # print("2")
            #             outputs = outputs.to(self.device)
            #             # print(outputs.shape) # [1024, 64, 4]
            #             rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
            #             # print(rgb_2d.shape) # [1024, 3] -> 3 : [r,g,b]
                        
            #             # Hierarchical sampling 추가

            #             rgb_2d = rgb_2d.cpu().detach().numpy()
            #             rgb_2d = rgb_2d * 255
            #             if idx % 10 == 0:
            #                 print(rgb_2d)
            #             # print(rgb_2d.shape) # [1024, 3]
            #             rgb_list.append(rgb_2d)
            #             print(idx)
            #         print(len(rgb_list))
            #         # print(len(rgb_list)) # 3
            #         # print(rgb_list[0].shape) # [1024, 3]
            #         rgb_arr = np.concatenate(rgb_list, axis=0)
            #         # rgb_arr = np.array(rgb_list)
            #         print(rgb_arr.shape) # [22860800, 3]
            #         # rgb_arr = torch.stack(rgb_list, dim=0)
            #         # print(rgb_arr.shape)
            #         rgb_list = np.array_split(rgb_arr, 120, axis=0) # 120개의 image로 나눈다.
            #         # rgb_list -> 120의 length, list 하나의 요소 -> [378x504]의 array로 나타낸다.
            #         rgb_list = [image.reshape([height, width, 3]) for image in rgb_list]
            #         for idx, image in enumerate(rgb_list):
            #             cv.imwrite('./data/epoch{}_rendered{}.png'.format(epoch, idx), image)
            #         print("nice job")
