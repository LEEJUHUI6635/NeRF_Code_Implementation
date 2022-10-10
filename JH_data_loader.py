import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
# LLFF Custom Dataset 구성 -> rays(or rays_np) and rays_ndc -> output : rays_o(orientation), rays_d(direction), r(t) = o + td -> o = rays_o, d = rays_d, t -> stratified sampling or Hierarchical sampling
# sample로 무엇이 나와야 하는가? = train 돌려야 할 대상 -> 각 이미지에 해당하는 batch ray들 
# 1) poses_bounds.npy 파일에서 pose와 boundaries를 가져와서 처리한다. images, poses, bds, render_poses, i_test 출력
# 2) 각 rays와 rgb를 sample로 출력한다. -> dictionary or list?

# LLFF Dataloader -> 각 이미지에 해당하는 batch ray들을 dataloader로 구성한다. batch size를 결정하고, shuffle 한다.
# rays를 배치화
base_dir = './data/nerf_llff_data/fern'
factor = 8 # image_height, image_width, focal_length
batch_size = 1024 # 한 번에 1024개의 ray를 학습한다.

def normalize(x):
    return x / np.linalg.norm(x)

def new_matrix(vec2, up, center):
    # print(vec2.shape) # [3, ]
    vec2 = normalize(vec2) # Z축
    vec1_avg = up # Y축
    vec0 = normalize(np.cross(vec1_avg, vec2)) # X축 = Y축 x Z축
    vec1 = normalize(np.cross(vec2, vec0)) # Y축 = Z축 x X축
    matrix = np.stack([vec0, vec1, vec2, center], axis=1)
    return matrix

# 새로운 world coordinate를 구하는 함수
def new_origin(poses): # input -> poses[20, 3, 5], output -> average pose[3, 5]
    # hwf
    hwf = poses[0,:,-1:]
    # print(hwf.shape) # [3, 1]
    # center -> translation의 mean
    center = poses[:,:,3].mean(0) # 이미지 개수에 대한 mean
    # vec2 -> [R3, R6, R9] rotation의 Z축에 대해 sum + normalize
    vec2 = normalize(poses[:,:,2].sum(0)) # 이미지 개수에 대한 sum
    # up -> [R2, R5, R8] rotation의 Y축에 대한 sum
    up = poses[:,:,1].sum(0) # 이미지 개수에 대한 sum
    
    new_world = new_matrix(vec2, up, center)
    # print(new_world.shape) # [3, 4]
    new_world = np.concatenate([new_world, hwf], axis=1)
    return new_world

# 고민인 점 -> LLFF poses_bounds.npy 파일로부터 poses, bds, images를 추출하는 것과 이로부터 rays_o, rays_d를 생성해 내는 것을 다른 클래스로 둘까, 하나의 클래스로 둘까.
# 먼저 다른 클래스로 만들어 둔다.

class LLFF(object):
    def __init__(self, base_dir, factor):
        self.base_dir = base_dir
        self.factor = factor
        self.preprocessing()
        self.load_images()
        self.pre_poses()
        self.spiral_path()

    def preprocessing(self): # Q. colmap의 순서를 고려해야 하나?
        # poses_bounds.npy 파일에서 pose와 bds를 얻는다.
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        # print(poses_bounds.shape) # [20, 17]
        self.image_num = poses_bounds.shape[0]
        # print(self.image_num) # 20
        poses = poses_bounds[:,:-2] # depth를 제외한 pose
        self.bds = poses_bounds[:,-2:] # depth
        # print(self.poses.shape) # [20, 15]
        # poses -> 3x5 matrix
        # print(poses)
        self.poses = poses.reshape(-1, 3, 5) # [20, 3, 5]
        # print(poses)
    
    def load_images(self): # images -> [height, width, 3, image_num] + 255로 나누어 normalize
        # 모든 image를 불러온다.
        image_dir = os.path.join(self.base_dir, 'images')
        # print(image_dir) # ./data/nerf_llff_data/fern/images

        files = sorted([file for file in os.listdir(image_dir)]) # Q. colmap의 순서?
        # print(files)
        
        images_list = []
        for file in files:
            images_BGR = cv.imread(os.path.join(image_dir, file))
            self.width = images_BGR.shape[1]
            self.height = images_BGR.shape[0] 
            images_resize = cv.resize(images_BGR, dsize=(self.width // self.factor, self.height // self.factor)) # width, height 순서 
            images_RGB = cv.cvtColor(images_resize, cv.COLOR_BGR2RGB) / 255 # normalization
            # print(images_RGB) # [3024, 4032, 3]
            images_list.append(images_RGB)
        self.images = np.array(images_list)
        # print(self.images.shape) # [20, 3024, 4032, 3]
    
    def pre_poses(self):
        # 좌표축 변환, [-u, r, -t] -> [r, u, -t]
        self.poses = np.concatenate([self.poses[:,:,1:2], -self.poses[:,:,0:1], self.poses[:,:,2:]], axis=-1)

        # 20개 pose들의 average pose를 구하고, 새로운 world coordinate를 생성한다. -> camera to world coordinate
        new_world = new_origin(self.poses) # 새로운 world coordinate, c2w
        # print(new_world.shape) # [3, 5]
        
        # 새로운 world coordinate를 중심으로 새로운 pose를 계산한다. -> poses = np.linalg.inv(c2w) @ poses
        last = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        # image_height, image_width, focal_length를 factor로 나눈다.
        hwf = new_world[:,-1].reshape(1, 3, 1) // self.factor
        # print(hwf.shape) # [1, 3, 1]
        new_world = np.concatenate([new_world[:,:4], last], axis=0)
        # print(new_world)
        last = last.reshape(1, 1, 4)
        lasts = np.repeat(last, 20, axis=0)
        # print(lasts.shape) # [20, 1, 4]
        
        self.new_poses = np.concatenate([self.poses[:,:,:4], lasts], axis=1)
        # print(self.poses.shape) # [20, 4, 4]
        self.new_poses = np.linalg.inv(new_world) @ self.new_poses
        # print(self.new_poses.shape) # [20, 4, 4]
        self.new_poses = self.new_poses[:,:3,:4]

        # poses + hwfs = [20, 3, 4] + [20, 3, 1]
        # for문 대신 repeat
        hwfs = np.repeat(hwf, 20, axis=0)
        # print(hwfs.shape) # [20, 3, 1]

        self.poses = np.concatenate([self.new_poses, hwfs], axis=-1)
        # print(self.poses.shape) # [20, 3, 5]

        # i_test -> recentered pose의 average pose의 translation과 가장 적게 차이가 나는 pose를 validation set으로 이용
        avg_pose = new_origin(self.poses)
        # print(avg_pose.shape) # [3, 5]
        trans = np.sum(np.square(avg_pose[:3,3] - self.poses[:,:3,3]), -1) # avg_pose - poses = [20, 3, 3]
        self.i_val = np.argmin(trans)
        # print(i_val) # 12
    ################################################
    ################################################
    ################################################
    # 나선형으로 new rendering path 만들기 -> *****rendering path 이해하기*****
    def spiral_path(self):
        # new global origin에 대하여 recentered된 pose들의 새로운 origin을 다시 구한다. -> 이 말은 곧 recentered pose의 average pose
        # new global origin과 위처럼 새롭게 구한 origin은 다른 값이 나온다.
        # print(self.poses.shape) # [20, 3, 5]
        z_rate = 0.5
        
        poses_avg = new_origin(self.poses) # recentered pose의 average pose
        # print(poses_avg.shape) # [3, 5]
        hwf = poses_avg[:,-1:]
        # print(hwf.shape) # [3, 1]
        # recenter된 pose의 up vector를 구한다. -> Y축
        up = self.poses[:,:3,1].sum(0)
        
        # focus depth를 구한다.
        # close_depth -> bds.min x 0.9, inf_depth -> bds.max x 5.
        # print(self.bds.shape) # [20, 2]
        close_depth, inf_depth = self.bds.min()*0.9, self.bds.max()*5.
        dt = 0.75
        # mean_focal -> 1/((1-dt)/close_depth + dt/inf_depth)
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
        # print(last.shape) # [1, 1]
        radius = radius.reshape(1, 3)
        # print(radius.shape) # [1, 3]
        radius = np.concatenate([radius, last], axis=1)
        # print(radius.shape) # [1, 4]
        radius = radius.reshape(4, 1)
        # print(radius.shape) # [4,]
        # 두 바퀴를 회전하는 spiral path를 생성 -> 2 바퀴를 돌고 마지막 index를 제외 -> 120개의 new rendered path
        render_poses_list = []
        for theta in np.linspace(start=0., stop=rotation_num*2*np.pi, num=view_num+1)[:-1]:
            # Look vector -> recentered poses의 average pose의 R|t @ [radius * cos(theta), -radius * sin(theta), -radius * sin(z_rate * theta)]
            # print(np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1]).shape)
        
            look = poses_avg[:3,:4] @ (np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1]).reshape(4, 1) * radius)
            # print(look.shape) # [3, 1]
            look = look.reshape([3,])
            # [3, 4] x [cos, -sin, -sin, 1] -> [4, 1]
            # print(look.shape) # [3, 1]
            # z = Look vector(target 위치) - eye vector(현재 camera 위치)
            eye = poses_avg[:3,:4] @ np.array([0, 0, -mean_focal, 1]).reshape(4, 1)
            eye = eye.reshape([3,])
            # print(eye.shape) # [3, 1]
            z = normalize(look - eye)
            z = z.reshape([3,])
            # print(z.shape) # [3, 1]
            # print(up.shape) # [3, ]
            # 새로운 pose에서의 camera to world matrix -> new matrix 함수 이용
            render_poses = new_matrix(vec2=z, up=up, center=look)
            # print(render_poses.shape) # [3, 4]
            render_poses = np.concatenate([render_poses, hwf], axis=-1)
            # print(render_poses)
            render_poses_list.append(render_poses)
        self.render_poses = np.array(render_poses_list)
        # print(self.render_poses.shape) # [120, 3, 5]
        
    def outputs(self):
        images = self.images
        poses = self.poses
        bds = self.bds
        render_poses = self.render_poses
        i_val = self.i_val
        return images, poses, bds, render_poses, i_val

images, poses, bds, render_poses, i_val = LLFF(base_dir, factor).outputs()
# print(i_val) # 12

height = 3024 // factor
width = 4032 // factor
focal = 3260.526333 // factor
# K = [3, 3] -> K[0][2] = 0.5*W, K[0][0] = focal, K[1][2] = 0.5*H, K[1][1] = focal
intrinsic = np.array([
            [focal, 0, 0.5*width], # 0.5*W = x축 방향의 주점
            [0, focal, 0.5*height], # 0.5*H = y축 방향의 주점
            [0, 0, 1]])
# print('focal : ', focal) # 407.0
##################################################
##################################################
##################################################
# Custom Dataset -> train, validation
# validation -> 가장 average pose와 가까운 image / train -> validation image를 제외한 image
##################################################
##################################################
##################################################

# Dataloader -> rays_o, rays_d + NDC space(optional)
# 고민 : render_poses는 어떻게 처리하지?
class Rays_DATASET(Dataset):
    def __init__(self, height, width, intrinsic, poses, i_val, images, near=1.0, ndc_space=True, train=True): # pose -> [20, 3, 5]
        super(Rays_DATASET, self).__init__()
        self.height = height
        self.width = width
        self.intrinsic = intrinsic
        self.pose = poses[:,:,:4]
        self.i_val = i_val
        self.images = images
        self.near = near
        self.ndc_space = ndc_space
        self.train = train
                    
        self.image_num = self.pose.shape[0] # train or test
        self.focal = self.intrinsic[0][0]
        
        total_image = self.pose.shape[0] # 전체 이미지 20개
        # print(total_image) # 20
        # optional
        if self.ndc_space == False:
            self.all_rays()
        elif self.ndc_space == True:
            self.ndc_all_rays()
        
        # val_idx -> 0, 12
        train_idx = []
        test_idx = []
        for i in range(total_image):
            if i % self.i_val == 0:
                test_idx.append(i)
            else:
                train_idx.append(i)
        # print(train_idx) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
        # print(test_idx) # [0, 12]
        # print(self.pose.shape) # [20, 3, 4]
        # print(self.images.shape) # [20, 378, 504, 3]
        # Dataset split : validation -> 0, 12
        if self.train == True:
            # test_idx만 제거하면 된다.
            self.pose = self.pose[train_idx,:,:]
            self.images = self.images[train_idx,:,:,:]
        elif self.train == False:
            # test_idx만 넣으면 된다.
            self.pose = self.pose[test_idx,:,:]
            self.images = self.images[test_idx,:,:,:]
        
        # print(self.pose.shape) # [18, 3, 4]
        # print(self.images.shape) # [18, 378, 504, 3]
        
        self.image_num = self.pose.shape[0] # train or test
        # print(self.image_num)
        
    # 하나의 image에 대한 camera to world -> rays_o는 image마다 다르기 때문
    def get_rays(self, pose): # rays_o, rays_d
        # height, width에 대하여 pixel마다 sampling -> pixel 좌표 sampling
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32), indexing='xy')
        # print(u.shape) # [378, 504]
        # print(v.shape) # [378, 504]
        
        # pixel 좌표계 -> metric 좌표계
        # (i - cx) / fx, (j - cy) / fy
        # cx = K[0][2], cy = K[1][2], fx = K[0][0], fy = K[1][1]
        # metric 좌표계의 y축은 pixel 좌표계의 v축과 반대
        # z축에 해당하는 값 -1 -> ray는 3D point에서 2D point로의 방향이기 때문
        
        pix2metric = np.stack([(u-self.intrinsic[0][2])/self.intrinsic[0][0], -(v-self.intrinsic[1][2])/self.intrinsic[1][1], -np.ones_like(u)], axis=-1)
        # print(pix2metric.shape) # [378, 504, 3]
        
        # camera 좌표계 -> world 좌표계
        rays_d = np.sum(pose[:3,:3] @ pix2metric.reshape(self.height, self.width, 3, 1), axis=-1)
        # print(rays_d)
        
        # rays_o -> translation, self.pose[:,3]
        # 하나의 image에 대한 모든 픽셀의 ray에 대한 원점은 같아야 한다. ray는 3D point에서 2D point로 향하는 방향이다.
        # [3042, 4032, 3]
        rays_o = np.broadcast_to(pose[:3,3], rays_d.shape)
        # print(rays_o.shape) # [378, 504, 3]
        # rays = np.stack([rays_o, rays_d], axis=0)
        # # print(rays.shape) # [2, 378, 504, 3]
        return rays_o, rays_d
    
    # NDC -> projection matrix
    # *****NDC 수식 이해하기***** -> Q. near = 1.?
    def ndc_rays(self, near, focal, rays_o, rays_d): # optional
        
        rays_o, rays_d = self.get_rays(self.pose[0,:3,:4])
        # rays_o, rays_d -> [378, 504, 3]
        # print(rays_o.shape, rays_d.shape)
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        # print(t.shape) # [378, 504]
        rays_o = rays_o + t[...,np.newaxis] * rays_d
        # print(rays_o.shape) # [378, 504, 3]
        o1 = -focal/(self.width/2) * rays_o[...,0]/rays_o[...,2]
        o2 = -focal/(self.height/2) * rays_o[...,1]/rays_o[...,2]
        o3 = 1 + 2*near/rays_o[...,2]
        
        d1 = -focal/(self.width/2) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d2 = -focal/(self.height/2) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d3 = -2*near/rays_o[...,2]
        # print(o1.shape, o2.shape, o3.shape) # [378, 504]
        # print(d1.shape, d2.shape, d3.shape)
        rays_o = np.stack([o1, o2, o3], axis=0)
        rays_d = np.stack([d1, d2, d3], axis=0)
        rays = np.stack([rays_o, rays_d], axis=0) # [2, 378, 504, 3]
        return rays
    
    def all_rays(self): # 모든 image에 대한 rays -> rays_o + rays_d + rgb
        # rays + rgb -> [2, 378, 504, 3(x, y, -1)] + [1, 378, 504, 3(r, g, b)]
        # get_rays -> rays_o + rays_d
        rays = np.stack([np.stack(self.get_rays(poses), axis=0) for poses in self.pose[:,:3,:4]], axis=0)
        # print(rays.shape) # [20, 2, 378, 504, 3]
        # print(self.images.shape) # [20, 378, 504, 3]
        self.images = self.images[:,np.newaxis,...]
        # print(self.images.shape) # [20, 1, 378, 504, 3]
        rays_rgb = np.concatenate([rays, self.images], axis=1)
        # print(rays_rgb.shape) # [20, 3, 378, 504, 3]
        rays_rgb = np.moveaxis(rays_rgb, source=1, destination=3)
        # print(rays_rgb.shape) # [20, 378, 504, 3, 3]
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        # print(rays_rgb.shape) # [3810240, 3, 3]
        self.rays_rgb = rays_rgb.astype(np.float32)
        # print(self.rays_rgb[0,:,:], self.rays_rgb[1,:,:])
        self.rays_rgb_list = np.split(self.rays_rgb, self.rays_rgb.shape[0], axis=0)
        # print(self.rays_rgb_list[0].shape) # [1, 3, 3] -> 맨 앞에 있는 1 제거
        # print(self.rays_rgb_list[0])
        self.rays_rgb_list = [self.rays_rgb_list[i].reshape(3, 3) for i in range(len(self.rays_rgb_list))]
    
    def ndc_all_rays(self):
        rays_list = []
        # print(self.image_num)
        for i in range(self.image_num):
            rays_o, rays_d = self.get_rays(poses[i,:3,:4])
            rays = self.ndc_rays(self.near, self.focal, rays_o, rays_d)
            # print(rays.shape) # [2, 3, 378, 504]
            rays = np.moveaxis(rays, source=1, destination=-1)
            # print(rays.shape) # [2, 378, 504, 3]
            rays_list.append(rays)
        rays_arr = np.array(rays_list)
        # print(rays_arr.shape) # [20, 2, 378, 504, 3]
        # print(self.images.shape) # [20, 378, 504, 3]
        rays_rgb_arr = np.concatenate([rays_arr, self.images[:,np.newaxis,...]], axis=1)
        # print(self.rays_rgb_arr.shape) # [20, 3, 378, 504, 3]
        rays_rgb_arr = np.moveaxis(rays_rgb_arr, source=1, destination=3)
        # print(self.rays_rgb_arr.shape) # [20, 378, 504, 3, 3]
        rays_rgb_arr = rays_rgb_arr.reshape(-1, 3, 3)
        # print(rays_rgb_arr.shape) # [3810240, 3, 3]
        self.rays_rgb_list = [rays_rgb_arr[i,:,:] for i in range(rays_rgb_arr.shape[0])]
        # print(len(self.rays_rgb_list)) # 3810240
        
    def __len__(self): # should be iterable
        return len(self.rays_rgb_list)
    
    def __getitem__(self, index): # should be iterable
        samples = self.rays_rgb_list[index]
        # print(samples.shape)
        return samples # rays_o + rays_d + rgb

samples = Rays_DATASET(height, width, intrinsic, poses, i_val, images, near=1.0, ndc_space=True, train=True)

# Rays DataLoader
class Rays_DATALOADER(object):
    def __init__(self, batch_size, height, width, intrinsic, poses, i_val, images):
        self.height = height
        self.width = width
        self.intrinsic = intrinsic
        self.poses = poses
        self.i_val = i_val
        self.images = images
        self.batch_size = batch_size
        self.samples = Rays_DATASET(self.height, self.width, self.intrinsic, self.poses, self.i_val, self.images)
        
    def data_loader(self): # shuffle = False
        dataloader = DataLoader(dataset=self.samples, batch_size=self.batch_size)
        return dataloader

data_loader = Rays_DATALOADER(batch_size, height, width, intrinsic, poses, i_val, images).data_loader()

for idx, samples in enumerate(data_loader):
    # print(samples.shape) # [1024, 3, 3]
    break