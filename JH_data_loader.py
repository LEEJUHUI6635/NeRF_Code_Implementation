from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2 as cv
import sys
# Dataloader에서 ***bds_factor = 0.75***

def normalize(x):
    return x / np.linalg.norm(x)

def new_matrix(vec2, up, center):
    vec2 = normalize(vec2) # Z축
    vec1_avg = up # Y축
    vec0 = normalize(np.cross(vec1_avg, vec2)) # X축 = Y축 x Z축
    vec1 = normalize(np.cross(vec2, vec0)) # Y축 = Z축 x X축
    matrix = np.stack([vec0, vec1, vec2, center], axis=1)
    return matrix

def new_origin(poses): # input -> poses[20, 3, 5], output -> average pose[3, 5]
    hwf = poses[0,:,-1:]
    # center -> translation의 mean
    center = poses[:,:,3].mean(0) # 이미지 개수에 대한 mean
    # vec2 -> [R3, R6, R9] rotation의 Z축에 대해 sum + normalize
    vec2 = normalize(poses[:,:,2].sum(0)) # 이미지 개수에 대한 sum
    # up -> [R2, R5, R8] rotation의 Y축에 대한 sum
    up = poses[:,:,1].sum(0) # 이미지 개수에 대한 sum
    new_world = new_matrix(vec2, up, center)
    new_world = np.concatenate([new_world, hwf], axis=1)
    return new_world

class LLFF(object): # *** bd_factor를 추가해야 한다. ***
    def __init__(self, base_dir, factor, bd_factor=0.75): # bd_factor = 0.75로 고정
        self.base_dir = base_dir
        self.factor = factor
        self.bd_factor = bd_factor # ***
        self.preprocessing() # Test
        self.load_images()
        self.pre_poses() # Test
        self.spiral_path() # Test
        
    def preprocessing(self): # Q. colmap의 순서를 고려해야 하나?
        # poses_bounds.npy 파일에서 pose와 bds를 얻는다.
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        self.image_num = poses_bounds.shape[0]
        poses = poses_bounds[:,:-2] # depth를 제외한 pose
        self.bds = poses_bounds[:,-2:] # depth
        self.poses = poses.reshape(-1, 3, 5) # [20, 3, 5]
        
    def load_images(self): # images -> [height, width, 3, image_num] + 255로 나누어 normalize
        # 모든 image를 불러온다.
        image_dir = os.path.join(self.base_dir, 'images')
        files = sorted([file for file in os.listdir(image_dir)]) # Q. colmap의 순서?
        images_list = []
        for file in files:
            images_RGB = cv.imread(os.path.join(image_dir, file), flags=cv.IMREAD_COLOR) # RGB로 읽기
            self.width = images_RGB.shape[1]
            self.height = images_RGB.shape[0]
            images_resize = cv.resize(images_RGB, dsize=(self.width // self.factor, self.height // self.factor)) # width, height 순서 
            images_resize = images_resize / 255 # normalization
            images_list.append(images_resize)
        self.images = np.array(images_list)
    
    def pre_poses(self): # bds_factor에 대해 rescale을 처리해야 한다.
        # 좌표축 변환, [-u, r, -t] -> [r, u, -t]
        sc = 1. if self.bd_factor is None else 1./(self.bds.min() * self.bd_factor)
        self.poses = np.concatenate([self.poses[:,:,1:2], -self.poses[:,:,0:1], self.poses[:,:,2:]], axis=-1)
        
        # bd_factor로 rescaling
        self.poses[:,:3,3] *= sc # translation에 해당하는 부분
        self.bds *= sc
        
        image_num = self.poses.shape[0]
        # 20개 pose들의 average pose를 구하고, 새로운 world coordinate를 생성한다. -> camera to world coordinate
        new_world = new_origin(self.poses) # 새로운 world coordinate, c2w
        # 새로운 world coordinate를 중심으로 새로운 pose를 계산한다. -> poses = np.linalg.inv(c2w) @ poses
        last = np.array([0, 0, 0, 1]).reshape(1, 4)
        # image_height, image_width, focal_length를 factor로 나눈다.
        hwf = new_world[:,-1].reshape(1, 3, 1) // self.factor
        self.focal = np.squeeze(hwf[:,-1,:])

        new_world = np.concatenate([new_world[:,:4], last], axis=0)
        last = last.reshape(1, 1, 4)
        lasts = np.repeat(last, image_num, axis=0)
        
        self.new_poses = np.concatenate([self.poses[:,:,:4], lasts], axis=1)
        self.new_poses = np.linalg.inv(new_world) @ self.new_poses
        self.new_poses = self.new_poses[:,:3,:4]
        hwfs = np.repeat(hwf, image_num, axis=0)
        self.poses = np.concatenate([self.new_poses, hwfs], axis=-1)
        # i_test -> recentered pose의 average pose의 translation과 가장 적게 차이가 나는 pose를 validation set으로 이용
        avg_pose = new_origin(self.poses)
        trans = np.sum(np.square(avg_pose[:3,3] - self.poses[:,:3,3]), -1) # avg_pose - poses = [20, 3, 3]
        self.i_val = np.argmin(trans)
    # 나선형으로 new rendering path 만들기 -> *****rendering path 이해하기*****
    
    def spiral_path(self):
        # new global origin에 대하여 recentered된 pose들의 새로운 origin을 다시 구한다. -> 이 말은 곧 recentered pose의 average pose
        # new global origin과 위처럼 새롭게 구한 origin은 다른 값이 나온다.
        z_rate = 0.5
        
        poses_avg = new_origin(self.poses) # recentered pose의 average pose
        hwf = poses_avg[:,-1:]
        # recenter된 pose의 up vector를 구한다. -> Y축
        up = self.poses[:,:3,1].sum(0)
        
        # focus depth를 구한다.
        # close_depth -> bds.min x 0.9, inf_depth -> bds.max x 5.
        close_depth, inf_depth = self.bds.min()*0.9, self.bds.max()*5.
        dt = 0.75
        # mean_focal
        mean_focal = 1/((1-dt)/close_depth + dt/inf_depth)
        
        # spiral path의 radius를 구한다.
        # recentered poses의 translation
        trans = self.poses[:,:3,3]
        # radius = translation의 절대값 -> 90 퍼센트에 해당하는 크기
        radius = np.percentile(a=trans, q=0.9, axis=0)
        
        # 새로 만들 view 개수 -> 120
        view_num = 120
        # rotation 횟수 -> 2
        rotation_num = 2
        # radius -> [1, 3] 마지막에 1을 붙여서 homogeneous coordinate으로 만들기
        last = np.array([1]).reshape(1, 1)
        radius = radius.reshape(1, 3)
        radius = np.concatenate([radius, last], axis=1)
        radius = radius.reshape(4, 1)
        # 두 바퀴를 회전하는 spiral path를 생성 -> 2 바퀴를 돌고 마지막 index를 제외 -> 120개의 new rendered path
        render_poses_list = []
        for theta in np.linspace(start=0., stop=rotation_num*2*np.pi, num=view_num+1)[:-1]:
            # Look vector -> recentered poses의 average pose의 R|t @ [radius * cos(theta), -radius * sin(theta), -radius * sin(z_rate * theta)]        
            look = poses_avg[:3,:4] @ (np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1]).reshape(4, 1) * radius)
            look = look.reshape([3,])
            # [3, 4] x [cos, -sin, -sin, 1] -> [4, 1]
            # z = Look vector(target 위치) - eye vector(현재 camera 위치)
            eye = poses_avg[:3,:4] @ np.array([0, 0, -mean_focal, 1]).reshape(4, 1)
            eye = eye.reshape([3,])
            z = normalize(look - eye)
            z = z.reshape([3,])
            # 새로운 pose에서의 camera to world matrix -> new matrix 함수 이용
            render_poses = new_matrix(vec2=z, up=up, center=look)
            render_poses = np.concatenate([render_poses, hwf], axis=-1)
            render_poses_list.append(render_poses)
        self.render_poses = np.array(render_poses_list)
        
    def outputs(self):
        images = self.images.astype(np.float32)
        poses = self.poses.astype(np.float32) # Test
        bds = self.bds # Test 
        render_poses = self.render_poses.astype(np.float32) # Test
        i_val = self.i_val # Test
        focal = self.focal
        return images, poses, bds, render_poses, i_val, focal # focal length 추가해야 할 듯.

# poses <- poses : Train or Validation / poses <- render_poses : Test
class Rays_DATASET(Dataset):
    def __init__(self, height, width, intrinsic, poses, i_val, images, near=1.0, ndc_space=True, test=False, train=True): # pose -> [20, 3, 5] / Test
        super(Rays_DATASET, self).__init__()
        self.height = height
        self.width = width
        self.intrinsic = intrinsic
        self.pose = poses[:,:,:4] # [?, 3, 4]
        self.i_val = i_val
        self.images = images
        self.near = near
        self.ndc_space = ndc_space
        self.test = test
        self.train = train
    
        self.focal = self.intrinsic[0][0]

        self.image_num = self.pose.shape[0] # Train과 Test 모두에 사용된다.
        
        if self.test == False: # Train or Validation
            train_idx = []
            val_idx = []
            for i in range(self.image_num):
                if i % self.i_val == 0:
                    val_idx.append(i) # [0, 12]
                    # print(val_idx) 
                else:
                    train_idx.append(i) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
        
            if self.train == True: # Train
                self.pose = self.pose[train_idx,:,:]
                self.images = self.images[train_idx,:,:,:]
                self.image_num = self.pose.shape[0]
                
            elif self.train == False: # Validation
                self.pose = self.pose[val_idx,:,:]
                self.images = self.images[val_idx,:,:,:]
                self.image_num = self.pose.shape[0]
                
        if self.ndc_space == False:
            self.all_rays()
                
        elif self.ndc_space == True: # default -> LLFF dataloader
            self.all_rays() # view_dirs
            self.ndc_all_rays()
        
        # train과 test의 차이는 train -> rays_o + rays_d + rays_rgb VS test -> rays_o + rays_d
        
        # # test_data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, render_poses, None, None, near, config.ndc_space, True, False, shuffle=False, drop_last=False).data_loader() # Test
        # elif self.test == True:
        #     # Test dataloader -> rays_o + rays_d, rays_rgb는 필요 없다.
        #     if self.ndc_space == False:
        #         self.test_all_rays()
        #     elif self.ndc_space == True:
        #         self.test_ndc_all_rays()
        
    # 하나의 image에 대한 camera to world -> rays_o는 image마다 다르기 때문
    def get_rays(self, pose): # rays_o, rays_d
        # height, width에 대하여 pixel마다 sampling -> pixel 좌표 sampling
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32), indexing='xy')
        
        # pixel 좌표계 -> metric 좌표계
        # (i - cx) / fx, (j - cy) / fy
        # cx = K[0][2], cy = K[1][2], fx = K[0][0], fy = K[1][1]
        # metric 좌표계의 y축은 pixel 좌표계의 v축과 반대
        # z축에 해당하는 값 -1 -> ray는 3D point에서 2D point로의 방향이기 때문
        
        pix2metric = np.stack([(u-self.intrinsic[0][2])/self.intrinsic[0][0], -(v-self.intrinsic[1][2])/self.intrinsic[1][1], -np.ones_like(u)], axis=-1)
        # print(pix2metric.shape) # [378, 504, 3]
        
        # camera 좌표계 -> world 좌표계
        rays_d = np.sum(pose[:3,:3] @ pix2metric.reshape(self.height, self.width, 3, 1), axis=-1)

        # 하나의 image에 대한 모든 픽셀의 ray에 대한 원점은 같아야 한다. ray는 3D point에서 2D point로 향하는 방향이다.
        rays_o = np.broadcast_to(pose[:3,3], rays_d.shape)
        return rays_o, rays_d
    
    # NDC -> projection matrix
    # *****NDC 수식 이해하기***** -> Q. near = 1.?
    def ndc_rays(self, near, focal, rays_o, rays_d): # optional
        # rays_o, rays_d = self.get_rays(self.pose[0,:3,:4])
        # rays_o, rays_d -> [378, 504, 3]
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        # print(t.shape) # [378, 504]
        rays_o = rays_o + t[...,np.newaxis] * rays_d
        # print(rays_o.shape) # [378, 504, 3]
        o1 = -1.*focal/(self.width/2) * rays_o[...,0] / rays_o[...,2]
        o2 = -1.*focal/(self.height/2) * rays_o[...,1] / rays_o[...,2]
        o3 = 1. + 2. * near / rays_o[...,2]
        
        d1 = -1.*focal/(self.width/2) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d2 = -1.*focal/(self.height/2) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d3 = -2.* near / rays_o[...,2]
        
        rays_o = np.stack([o1, o2, o3], axis=0)
        rays_d = np.stack([d1, d2, d3], axis=0)
        rays = np.stack([rays_o, rays_d], axis=0) # [2, 378, 504, 3]

        return rays
    
    def all_rays(self): # 모든 image에 대한 rays -> rays_o + rays_d + rgb
        # rays + rgb -> [2, 378, 504, 3(x, y, -1)] + [1, 378, 504, 3(r, g, b)]
        # get_rays -> rays_o + rays_d
        rays = np.stack([np.stack(self.get_rays(poses), axis=0) for poses in self.pose[:,:3,:4]], axis=0)
        # print(rays.shape) # [18, 2, 378, 504, 3]
        if self.test == False:
            self.images = self.images[:,np.newaxis,...]
            rays = np.concatenate([rays, self.images], axis=1)
        
        rays = np.moveaxis(rays, source=1, destination=3)
        # print(rays.shape) # [18, 378, 504, 3, 3] -> test : [120, 378, 504, 2, 3]
        if self.test == False:
            rays = rays.reshape([-1, 3, 3])
        else:
            rays = rays.reshape([-1, 2, 3])
        self.rays = rays.astype(np.float32)
        self.rays_rgb_list_2 = np.split(self.rays, self.rays.shape[0], axis=0)
        if self.test == False:
            self.rays_rgb_list_2 = [self.rays_rgb_list_2[i].reshape(3, 3) for i in range(len(self.rays_rgb_list_2))]
        else:
            self.rays_rgb_list_2 = [self.rays_rgb_list_2[i].reshape(2, 3) for i in range(len(self.rays_rgb_list_2))]
        self.view_dirs_list = [self.rays_rgb_list_2[i][1:2,:] for i in range(len(self.rays_rgb_list_2))]
        
    def ndc_all_rays(self):
        rays_list = []
        for i in range(self.image_num):
            rays_o, rays_d = self.get_rays(self.pose[i,:3,:4])
            rays = self.ndc_rays(self.near, self.focal, rays_o, rays_d)
            rays = np.moveaxis(rays, source=1, destination=-1)
            rays_list.append(rays)
        rays_arr = np.array(rays_list)

        if self.test == False:
            rays_arr = np.concatenate([rays_arr, self.images], axis=1)
        rays_arr = np.moveaxis(rays_arr, source=1, destination=3)

        if self.test == False:
            rays_arr = rays_arr.reshape(-1, 3, 3)
        else:
            rays_arr = rays_arr.reshape(-1, 2, 3)
        self.rays_rgb_list = [rays_arr[i,:,:] for i in range(rays_arr.shape[0])]
        
    def __len__(self): # should be iterable
        return len(self.rays_rgb_list)
        
    def __getitem__(self, index): # should be iterable
        samples = self.rays_rgb_list[index]
        view_dirs = self.view_dirs_list[index] # Debugging -> test시에는 없다.
        results = [samples, view_dirs]
        # return samples, view_dirs # rays_o + rays_d + rgb
        return results

class Rays_DATALOADER(object):
    def __init__(self, batch_size, height, width, intrinsic, poses, i_val, images, near, ndc_space, test, train, shuffle, drop_last):
        self.height = height
        self.width = width
        self.intrinsic = intrinsic
        self.poses = poses
        self.i_val = i_val
        self.images = images
        self.near = near # 1.0 -> default
        self.ndc_space = ndc_space
        self.test = test
        self.train = train
        self.batch_size = batch_size
        self.results = Rays_DATASET(self.height, self.width, self.intrinsic, self.poses, self.i_val, self.images, self.near, self.ndc_space, self.test, self.train)
        self.shuffle = shuffle # 나중에 train이면 shuffle = True / validation 혹은 test이면 shuffle = False 되게끔 만들기 !
        self.drop_last = drop_last # 나중에 train이면 drop_last = True / validation 혹은 test이면 drop_last = False 되게끔 만들기 !
        
    def data_loader(self): # shuffle = False
        dataloader = DataLoader(dataset=self.results, batch_size=self.batch_size, shuffle = self.shuffle, num_workers=4, pin_memory=True, drop_last=False) # drop_last = False -> 마지막 batch 또한 학습한다.
        return dataloader
