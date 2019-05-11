import os
import cv2
import math
import json
import torch
import torch.utils.data as data
import random
import numpy as np

class Resized(object):
    
    def __init__(self, num_kpt, size):
        self.num_kpt = num_kpt
        self.size = size
        
    def resize(self, img, keypoints):
        new_img = np.empty((self.size, self.size, 3), dtype=np.float32)
        new_img.fill(128)
        height, width, _ = img.shape
        ratio = min(self.size / height, self.size / width)
        ratio = random.uniform(0.75 * ratio, ratio)
        img = np.ascontiguousarray(cv2.resize(img, (0, 0), fx=ratio, fy=ratio))
        height, width, _ = img.shape
        offset_left = random.randint(0, self.size - width)
        offset_up = random.randint(0, self.size - height)
        for px in range(self.num_kpt):
            keypoints[px*3+0] = keypoints[px*3+0] * ratio + offset_left
            keypoints[px*3+1] = keypoints[px*3+1] * ratio + offset_up
        new_img[offset_up:offset_up+height, offset_left:offset_left+width, :] = img[:, :, :]
        return np.ascontiguousarray(new_img), keypoints
        
    def __call__(self, img, keypoints):
        return self.resize(img, keypoints)

# class TestResized(object):

#     def __init__(self, num_kpt, size):
#         self.num_kpt = num_kpt
#         self.size = size

#     def resize(self, img, keypoints):
#         new_img = np.empty((self.size, self.size, 3), dtype=np.float32)
#         new_img.fill(128)
#         height, width, _ = img.shape
#         ratio = min(self.size / height, self.size / width)
#         img = np.ascontiguousarray(cv2.resize(img, (0, 0), fx=ratio, fy=ratio))
#         height, width, _ = img.shape
#         for px in range(self.num_kpt):
#             keypoints[px*3+0] = keypoints[px*3+0] * ratio 
#             keypoints[px*3+1] = keypoints[px*3+1] * ratio
#         new_img[:height, :width, :] = img[:, :, :]
#         return np.ascontiguousarray(new_img), keypoints

#     def __call__(self, img, keypoints):
#         return self.resize(img, keypoints)

class TestResized(object):

    def __init__(self, num_kpt, size):
        self.num_kpt = num_kpt
        self.size = size

    def resize(self, img, keypoints):
        new_img = np.empty((self.size, self.size, 3), dtype=np.float32)
        new_img.fill(128)
        height, width, _ = img.shape
        ratio = min(self.size / height, self.size / width)
        img = np.ascontiguousarray(cv2.resize(img, (0, 0), fx=ratio, fy=ratio))
        height, width, _ = img.shape
        offset_x = (self.size-width) // 2
        offset_y = (self.size-height) // 2
        for px in range(self.num_kpt):
            keypoints[px*3+0] = keypoints[px*3+0] * ratio + offset_x
            keypoints[px*3+1] = keypoints[px*3+1] * ratio + offset_y
        new_img[offset_y:offset_y+height, offset_x:offset_x+width, :] = img[:, :, :]
        return np.ascontiguousarray(new_img), keypoints, ratio

    def __call__(self, img, keypoints):
        return self.resize(img, keypoints)

class RandomRotate(object):
    
    def __init__(self, num_kpt, max_degree=40):
        self.num_kpt = num_kpt
        self.max_degree = max_degree
        
    def rotate(self, img, keypoints):
        degree = random.randint(0, self.max_degree)
        height, width, _ = img.shape
        img_center = (width / 2.0, height / 2.0)
        rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
        cos_val = np.abs(rotateMat[0, 0])
        sin_val = np.abs(rotateMat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotateMat[0, 2] += (new_width / 2.) - img_center[0]
        rotateMat[1, 2] += (new_height / 2.) - img_center[1]
        
        img = cv2.warpAffine(img, rotateMat, (new_width, new_height), borderValue=(128, 128, 128))
        for px in range(self.num_kpt):
            x = keypoints[px*3+0]
            y = keypoints[px*3+1]
            p = np.array([x, y, 1])
            p = rotateMat.dot(p)
            keypoints[px*3+0] = p[0]
            keypoints[px*3+1] = p[1]
            
        return np.ascontiguousarray(img), keypoints
    
    def __call__(self, img, keypoints):
        return self.rotate(img, keypoints)

class RandomCrop(object):
    
    def __init__(self, num_kpt, ratio_max_x=0.1, ratio_max_y=0.01, center_perturb_max=10):
        self.num_kpt = num_kpt
        self.center_perturb_max = center_perturb_max
        self.ratio_max_x = ratio_max_x
        self.ratio_max_y = ratio_max_y
        
    def crop(self, img, keypoints):
        height, width, _ = img.shape
        ratio_x = random.uniform(0., self.ratio_max_x)
        ratio_y = random.uniform(0., self.ratio_max_y)
        x_offset = int(width * ratio_x)
        y_offset = int(height * ratio_y)
        
        for px in range(self.num_kpt):
            if keypoints[px*3+2]:
                keypoints[px*3+0] -= x_offset
                keypoints[px*3+1] -= y_offset
                if keypoints[px*3+0] < 0 or keypoints[px*3+1] < 0:
                    keypoints[px*3+2] = 0
        
        new_img = np.empty((height-2*y_offset, width-2*x_offset, 3), dtype=np.float32)
        new_img.fill(128)
            
        new_img[:, :, :] = img[y_offset: height-y_offset, x_offset: width-x_offset, :].copy()
        
        return np.ascontiguousarray(new_img), keypoints
        
    def __call__(self, img, keypoints):
        return self.crop(img, keypoints)

class RandomHorizontalFlip(object):
    
    def __init__(self, num_kpt, prob=0.5):
        self.num_kpt = num_kpt
        self.prob = prob
        
    def hflip(self, img, keypoints):
        height, width, _ = img.shape
        img = img[:, ::-1, :]
        for px in range(self.num_kpt):
            if keypoints[px*3+2]:
                keypoints[px*3+0] = width - 1 - keypoints[px*3+0]
        swap_pair = [(0, 5), (1, 4), (2, 3), (6, 11), (7, 10), (8, 9)]
        for (px1, px2) in swap_pair:
            temp_x = keypoints[px1*3+0]
            temp_y = keypoints[px1*3+1]
            temp_v = keypoints[px1*3+2]
            keypoints[px1*3+0] = keypoints[px2*3+0]
            keypoints[px1*3+1] = keypoints[px2*3+1]
            keypoints[px1*3+2] = keypoints[px2*3+2]
            keypoints[px2*3+0] = temp_x
            keypoints[px2*3+1] = temp_y
            keypoints[px2*3+2] = temp_v
        
        return np.ascontiguousarray(img), keypoints
        
    def __call__(self, img, keypoints):
        if random.random() < self.prob:
            return self.hflip(img, keypoints)
        return img, keypoints

class RandomAddColorNoise(object):
    
    def __init__(self, num_kpt, min_gauss=0, max_gauss=4, percentage=0.1):
        self.num_kpt = num_kpt
        self.min_gauss = min_gauss
        self.max_gauss = max_gauss
        self.percentage = percentage
    
    def noise(self, img, keypoints):
        height, width, _ = img.shape
        num = int(self.percentage * height * width)
        for i in range(num):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            img[y, x, 0] = max(min(img[y, x, 0]+random.gauss(self.min_gauss, self.max_gauss), 255), 0)
            img[y, x, 1] = max(min(img[y, x, 1]+random.gauss(self.min_gauss, self.max_gauss), 255), 0)
            img[y, x, 2] = max(min(img[y, x, 2]+random.gauss(self.min_gauss, self.max_gauss), 255), 0)
        
        return np.ascontiguousarray(img), keypoints
    
    def __call__(self, img, keypoints):
        return self.noise(img, keypoints)

class RandomBrightnessContrast(object):

    def __init__(self, num_kpt, min_alpha=0, max_alpha=3):
        self.num_kpt = num_kpt
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def bricon(self, img, keypoints):
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        beta = 125 * (1 - alpha)
        img = np.uint8(np.clip((alpha * img + beta), 0, 255))
        return np.ascontiguousarray(img), keypoints

    def __call__(self, img, keypoints):
        return self.bricon(img, keypoints)

class Compose(object):
    
    def __init__(self, trans):
        self.trans = trans
    
    def __call__(self, img, keypoints):
        for t in self.trans:
            img, keypoints = t(img, keypoints)
        return img, keypoints

def normalize(tensor, mean=[128.0, 128.0, 128.0], std=[256.0, 256.0, 256.0]):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def to_tensor(pic):
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    return img.float()