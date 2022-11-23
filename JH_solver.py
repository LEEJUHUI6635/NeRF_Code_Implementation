from JH_model import Stratified_Sampling, Positional_Encoding, Hierarchical_Sampling, NeRF, viewing_directions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# train, validation, test, Classic Volume Rendering 함수 or class + Data_loader 설정
import numpy as np
import cv2 as cv
import os
import time
import tqdm
# train 할 때에는 random하게 ray를 섞어야 하기 때문에, ray를 합쳐 하나의 image로 만드는 작업 -> 하나의 함수
# learning rate decay -> iteration이 한 번씩 돌 때마다

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
                    'model': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(), 
                    'loss': self.loss}, os.path.join(self.save_path, 'checkpoints_{}.pt'.format(self.epoch))) # self.save_path + 'checkpoints_{}.pt'
    def save_checkpoints_last(self):
        torch.save({'epoch': self.epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': self.loss}, os.path.join(self.save_path, 'checkpoints_last.pt'))

class Solver(object):
    def __init__(self, data_loader, val_data_loader, test_data_loader, config, i_val, height, width):
        self.data_loader = data_loader # rays dataloader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        
        # iterations
        self.resume_iters = config.resume_iters
        self.nb_epochs = config.nb_epochs
        self.save_val_iters = config.save_val_iters
        self.save_model_iters = config.save_model_iters # 15
        self.batch_size = config.batch_size
        self.coarse_num = config.coarse_num # 64
        self.fine_num = config.fine_num # 64
        self.sample_num = self.coarse_num # 64
        self.near = config.near # 0.
        self.far = config.far # 1.
        self.L_pts = config.L_pts # 10
        self.L_dirs = config.L_dirs # 4
        self.learning_rate = config.learning_rate
        
        # pts_channel, output_channel, dir_channel 설정
        self.pts_channel = 3 + 2 * self.L_pts * 3 # 3 + 2 x 10 x 3
        self.output_channel = 4
        self.dir_channel = 3 + 2 * self.L_dirs * 3 # 3 + 2 x 4 x 3
        
        # save path
        self.save_results_path = config.save_results_path
        self.save_train_path = config.save_train_path
        self.save_test_path = config.save_test_path
        self.save_model_path = config.save_model_path
        self.save_coarse_path = config.save_coarse_path
        self.save_fine_path = config.save_fine_path
        
        # validation
        self.i_val = i_val
        self.factor = config.factor
        self.height = height
        self.width = width
    
        self.basic_setting()
    
    # 고민 : optimizer가 학습할 parameter -> 한 번에 이렇게 학습해도 되나?
    def basic_setting(self): # Q. 2개의 network를 학습?
        # model -> Coarse + Fine
        # Coarse + Fine Network
        # 고민 -> 두 개의 network가 아니라 한 개의 network를 학습해야 하는 것이 아닌가? 즉, forward 부분도 하나로 통일해야 gradient가 한 번에 학습되는 것이 아닌가?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coarse_model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, self.batch_size, self.sample_num, self.device).to(self.device)
        # self.coarse_model = NeRF().to(self.device)
        grad_variables = list(self.coarse_model.parameters())
        self.fine_model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, self.batch_size, self.sample_num, self.device).to(self.device)
        # self.fine_model = NeRF().to(self.device)
        grad_variables += list(self.fine_model.parameters())
        
        # optimizer
        self.optimizer = optim.Adam(params=grad_variables, lr=self.learning_rate, betas=(0.9, 0.999))
        # learning rate decay를 실행시키기 위해서는 self.learning_rate가 self.optimizer에 반영되어야 한다. 하지만, 위 함수는 한 번만 정의되기 때문에, 새로운 learning rate를 반영하지 못할 것이다.
        # check -> optimizer를 출력해보면 된다.
        
        # loss function
        self.criterion = lambda x, y : torch.mean((x - y) ** 2)
        
        # evaluation metric -> PSNR
        self.psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(self.device))

    # Classic Volume Rendering -> rays_d = rays_d
    def classic_volume_rendering(self, raw, z_vals, rays, device): # input -> Network의 outputs [1024, 64, 4] + z_vals / output -> 2D color [1024, 3] -> rgb
        rays_d = rays[:,1:2,:]        
        raw = raw.to(self.device)
        z_vals = z_vals.to(self.device)
        rays = rays.to(self.device)
        rays_d = rays_d.to(self.device)
        rgb_3d = torch.sigmoid(raw[:,:,:3])
        rgb_3d = rgb_3d.to(self.device)
        density = raw[:,:,3:]
        density = density.to(self.device)
        dists = z_vals[:,1:] - z_vals[:,:-1] # [1024, 128]
        dists = dists.to(device)
        dists = torch.cat([dists, (torch.Tensor([1e10]).expand(dists.shape[0], 1)).to(self.device)], dim=-1)

        dists = dists * torch.norm(rays_d, dim=-1)
        dists = dists.to(self.device)
        active_func = nn.ReLU()
        noise = torch.randn_like(dists)
        noise = noise.to(self.device)
        alpha = 1 - torch.exp(-active_func((density.squeeze() + noise) * dists))
        transmittance = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(self.device), (1 - alpha + 1e-10).to(self.device)], dim=-1), dim=-1)[:,:-1]
        weights = alpha * transmittance
        
        rgb_2d = torch.sum(weights[...,None] * rgb_3d, dim=-2)
        return rgb_2d, weights
    
    # *****net_chunk*****
    def train(self): # device -> dataset, model
        # 학습 재기 -> 마지막에 저장된 checkpoints의 epoch, coarse_model, fine_model, optimizer를 가져온다.
        start_iters = 0
        if self.resume_iters != None: # self.resume_iters
            coarse_ckpt = torch.load(os.path.join(self.save_coarse_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
            fine_ckpt = torch.load(os.path.join(self.save_fine_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
            self.coarse_model.load_state_dict(coarse_ckpt['model'])
            self.fine_model.load_state_dict(fine_ckpt['model'])
            self.optimizer.load_state_dict(coarse_ckpt['optimizer'])
            start_iters = self.resume_iters + 1
            self.coarse_model.train()
            self.fine_model.train()
            
        # Time check
        start_time = time.time()
        # for epoch in range(start_iters, self.nb_epochs):
        for epoch in tqdm.tqdm(range(start_iters, self.nb_epochs)):
            # Dataloader -> 1024로 나눠 학습
            self.train_image_list = []
            for idx, [rays, view_dirs] in enumerate(tqdm.tqdm(self.data_loader)): # Dataloader -> rays = rays_o + rays_d + rays_rgb / view_dirs
            # for idx, [rays, view_dirs] in enumerate(self.data_loader):
                rays = rays.float()
                view_dirs = view_dirs.float()
                batch_size = rays.shape[0]
                # view_dirs -> NDC 처리 전의 get_rays로부터
                view_dirs = viewing_directions(view_dirs) # [1024, 3]
                rays_o = rays[:,0,:]
                rays_d = rays[:,1,:]
                rays_rgb = rays[:,2,:] # True

                # Stratified Sampling -> rays_o + rays_d -> view_dirs x
                pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
                pts = pts.reshape(batch_size, self.coarse_num, 3) # sample_num, [1024, 64, 3]
                coarse_view_dirs = view_dirs[:,None].expand(pts.shape) # [1024, 64, 3]
                pts = pts.reshape(-1, 3) # [65536, 3]
                coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
                coarse_pts = coarse_pts.to(self.device)
                coarse_view_dirs = coarse_view_dirs.to(self.device)

                inputs = torch.cat([coarse_pts, coarse_view_dirs], dim=-1)
                inputs = inputs.to(self.device)
                
                # Coarse Network
                outputs = self.coarse_model(inputs, sampling='coarse')
                # outputs = self.coarse_model(inputs)
                outputs = outputs.reshape(batch_size, self.coarse_num, 4) # rgb + density
                rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                
                # Hierarchical sampling + viewing_directions
                fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                fine_pts = fine_pts.reshape(batch_size, self.coarse_num + self.fine_num, 3) # [1024, 128, 3] -> [1024, self.coarse_num + self.fine_num, 3]
                
                fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape) # [1024, 128, 3]
                fine_pts = fine_pts.reshape(-1, 3)
                fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)
                fine_pts = fine_pts.to(self.device)
                fine_view_dirs = fine_view_dirs.to(self.device)
                fine_inputs = torch.cat([fine_pts, fine_view_dirs], dim=-1)
                fine_inputs = fine_inputs.to(self.device)
                
                # Fine model
                fine_outputs = self.fine_model(fine_inputs, sampling='fine')
                # fine_outputs = self.fine_model(fine_inputs)
                fine_outputs = fine_outputs.reshape(rays.shape[0], self.coarse_num + self.fine_num, 4) # 128 = self.coarse_num + self.fine_num

                # classic volume rendering
                fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                fine_rgb_2d = fine_rgb_2d.to(self.device)
                rgb_2d = rgb_2d.to(self.device)
                rays_rgb = rays_rgb.to(self.device)
                loss = self.criterion(fine_rgb_2d, rays_rgb) + self.criterion(rgb_2d, rays_rgb) # Coarse + Fine
                
                # optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = self.psnr(loss) # psnr 조금 수정해야 할 듯.
                # print(rays.shape) # [1024, 3, 3]
                # iteration n번 마다 출력
                
            # 한 epoch가 지날 때마다,
            print(idx, loss, psnr)
            print('----{}s seconds----'.format(time.time() - start_time))
            
            # Learning rate decay -> self.optimizer에도 적용되어야 한다.
            # epoch마다
            decay_rate = 0.1
            decay_steps = 250 * 1000
            new_lrate = self.learning_rate * (decay_rate ** (epoch / decay_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate

            # 한 epoch마다 model, optimizer 저장 -> 15 epoch마다
            # Save_Checkpoints -> 하나로 합치기
            if epoch % self.save_model_iters == 0 and epoch > 0: 
            # if epoch % 1 == 0 and epoch > 0:
                Save_Checkpoints(epoch, self.coarse_model, self.optimizer, loss, self.save_coarse_path, 'epoch')
                Save_Checkpoints(epoch, self.fine_model, self.optimizer, loss, self.save_fine_path, 'epoch')
            
            # Validation -> Validataion Dataloader = rays + NDC space 전에 추출한 get_rays의 view_dirs -> Train과 똑같이 처리한다. 다만, rays_rgb는 가져올 필요 없다.
            if epoch % self.save_val_iters == 0 and epoch > 0: # if epoch % 10 == 0 and epoch > 0:
            # if epoch % 1 == 0 and epoch > 0:
                with torch.no_grad():
                    val_image_list = []
                    for idx, [rays, view_dirs] in enumerate(tqdm.tqdm(self.val_data_loader)): # rays + view_dirs
                        batch_size = rays.shape[0]
                        # view_dirs -> NDC 처리 전의 get_rays로부터
                        view_dirs = viewing_directions(view_dirs) # [1024, 3]
                        rays_o = rays[:,0,:]
                        rays_d = rays[:,1,:]
                        rays_rgb = rays[:,2,:] # True
                        
                        # Stratified Sampling -> rays_o + rays_d -> view_dirs x
                        pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
                        pts = pts.reshape(batch_size, self.coarse_num, 3) # sample_num
                        coarse_view_dirs = view_dirs[:,None].expand(pts.shape) # [1024, 64, 3]
                        pts = pts.reshape(-1, 3)
                        coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                        
                        # Positional Encoding
                        coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                        coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
                        coarse_pts = coarse_pts.to(self.device)
                        coarse_view_dirs = coarse_view_dirs.to(self.device)

                        inputs = torch.cat([coarse_pts, coarse_view_dirs], dim=-1)
                        inputs = inputs.to(self.device)
                        
                        # Coarse Network
                        outputs = self.coarse_model(inputs, sampling='coarse')
                        # outputs = self.coarse_model(inputs)
                        outputs = outputs.reshape(batch_size, self.coarse_num, 4)
                        rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                        
                        # Hierarchical sampling + viewing_directions
                        fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                        fine_pts = fine_pts.reshape(batch_size, self.coarse_num + self.fine_num, 3) # [1024, 128, 3] -> 128 = self.coarse_num + self.fine_num
                        
                        fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape) # [1024, 128, 3]
                        fine_pts = fine_pts.reshape(-1, 3)
                        fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                        
                        # Positional Encoding
                        fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                        fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)
                        fine_pts = fine_pts.to(self.device)
                        fine_view_dirs = fine_view_dirs.to(self.device)
                        fine_inputs = torch.cat([fine_pts, fine_view_dirs], dim=-1)
                        fine_inputs = fine_inputs.to(self.device)
                        
                        # Fine model
                        fine_outputs = self.fine_model(fine_inputs, sampling='fine')
                        # fine_outputs = self.fine_model(fine_inputs)
                        fine_outputs = fine_outputs.reshape(rays.shape[0], self.coarse_num + self.fine_num, 4) # 128 = self.coarse_num + self.fine_num

                        # classic volume rendering
                        fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                        fine_rgb_2d = fine_rgb_2d.to(self.device)
                        # print(fine_rgb_2d.shape) # [1024, 3]
                        fine_rgb_2d = fine_rgb_2d.cpu().detach().numpy()
                        fine_rgb_2d = (255*np.clip(fine_rgb_2d,0,1)).astype(np.uint8)
                        val_image_list.append(fine_rgb_2d)
                        
                    val_image_arr = np.concatenate(val_image_list, axis=0)
                    val_image_arr = val_image_arr.reshape(2, self.height, self.width, 3) # validation image 개수만큼 -> flexible
                    for i in range(2): # 2 -> flexible
                        image = val_image_arr[i,:,:,:]
                        # cv.imwrite('./results/val/{}_{}.png'.format(epoch, i), image)
                        cv.imwrite(os.path.join(self.save_train_path, 'validation_epoch_{}_{}.png'.format(epoch, i)), image)
    
    def test(self):
        # render_only -> model checkpoints 가져오기
        # validation과 비슷하게 수행 -> test_dataloader에서 가져오기
        # 학습 재기 -> 마지막에 저장된 checkpoints의 coarse model과 fine model을 가져온다.
        coarse_ckpt = torch.load(os.path.join(self.save_coarse_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
        fine_ckpt = torch.load(os.path.join(self.save_fine_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
        self.coarse_model.load_state_dict(coarse_ckpt['model'])
        self.fine_model.load_state_dict(fine_ckpt['model'])
        self.coarse_model.eval()
        self.fine_model.eval()
        
        with torch.no_grad():
            test_image_list = []
            start_time = time.time()
            i = 0
            for idx, [rays, view_dirs] in enumerate(tqdm.tqdm(self.test_data_loader)): # rays + view_dirs
                batch_size = rays.shape[0]
                # view_dirs -> NDC 처리 전의 get_rays로부터
                view_dirs = viewing_directions(view_dirs) # [1024, 3]
                rays_o = rays[:,0,:]
                rays_d = rays[:,1,:]
                # rays_rgb = rays[:,2,:] # Test -> 쓸모 없다.
                
                # Stratified Sampling -> rays_o + rays_d -> view_dirs x
                pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
                pts = pts.reshape(batch_size, self.coarse_num, 3) # sample_num
                coarse_view_dirs = view_dirs[:,None].expand(pts.shape) # [1024, 64, 3]
                pts = pts.reshape(-1, 3)
                coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
                coarse_pts = coarse_pts.to(self.device)
                coarse_view_dirs = coarse_view_dirs.to(self.device)

                inputs = torch.cat([coarse_pts, coarse_view_dirs], dim=-1)
                inputs = inputs.to(self.device)
                
                # Coarse Network
                outputs = self.coarse_model(inputs, sampling='coarse')
                # outputs = self.coarse_model(inputs)
                outputs = outputs.reshape(batch_size, self.coarse_num, 4)
                rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                
                # Hierarchical sampling + viewing_directions
                fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                fine_pts = fine_pts.reshape(batch_size, self.coarse_num + self.fine_num, 3) # [1024, 128, 3], 128 = self.coarse_num + self.fine_num
                
                fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape) # [1024, 128, 3]
                fine_pts = fine_pts.reshape(-1, 3)
                fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)
                fine_pts = fine_pts.to(self.device)
                fine_view_dirs = fine_view_dirs.to(self.device)
                fine_inputs = torch.cat([fine_pts, fine_view_dirs], dim=-1)
                fine_inputs = fine_inputs.to(self.device)
                
                # Fine model
                fine_outputs = self.fine_model(fine_inputs, sampling='fine')
                # fine_outputs = self.fine_model(fine_inputs)
                fine_outputs = fine_outputs.reshape(rays.shape[0], self.coarse_num + self.fine_num, 4) # 128 = self.coarse_num + self.fine_num

                # classic volume rendering
                fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                fine_rgb_2d = fine_rgb_2d.to(self.device)
                # print(fine_rgb_2d.shape) # [1024, 3]
                fine_rgb_2d = fine_rgb_2d.cpu().detach().numpy()
                fine_rgb_2d = (255*np.clip(fine_rgb_2d,0,1)).astype(np.uint8)
                # 하나의 image 씩 -> batch_size가 1024가 아닐 때, (다른 dataset의 경우) image list의 길이가 378 x 504를 넘을 때, 하나의 이미지로 만든다.
                test_image_list.append(fine_rgb_2d)
        
                if batch_size != self.batch_size or len(test_image_list) >= self.height * self.width: # self.batch_size = 1024
                    test_image_arr = np.concatenate(test_image_list, axis=0)
                    test_image_arr = test_image_arr.reshape(120, self.height, self.width, 3)
                    # cv.imwrite('./results/test/{}.png'.format(), test_image_arr)
                    cv.imwrite(os.path.join(self.save_test_path, 'test_{}.png'.format(i)), test_image_arr)
                    test_image_list = [] # 이미지 1개 만들어내면, list 비우기
                    i += 1
                    
            # val_image_arr = np.concatenate(test_image_list, axis=0)
            # val_image_arr = val_image_arr.reshape(120, 378, 504, 3) # validation image 개수만큼 -> flexible
            # for i in range(120): # 120개의 image
            #     image = val_image_arr[i,:,:,:]
            #     cv.imwrite('./results/test/{}.png'.format(i), image)
            #     print('----{}s seconds----'.format(time.time() - start_time)) # enumerate -> tqdm -> 진행 사항 표시 -> 21분 걸림 -> 시간 줄여야 한다.