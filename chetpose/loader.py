import os
import cv2
import math
import json
import torch
import torch.utils.data as data
import random
import numpy as np
from transforms import normalize, to_tensor, TestResized

class LSPMPIILIP(data.DataLoader):
    
    def __init__(self, json_file, img_dir, size, num_kpt, s=7, sigma=7, trans=None):
        self.json_data = self.read_json_file(json_file)
        self.img_dir = img_dir
        self.size = size
        self.num_kpt = num_kpt
        self.s = s
        self.sigma = sigma
        self.trans = trans
        self.grid_size = self.size / self.s
        
    def read_json_file(self, json_file):
        fp = open(json_file)
        json_data = json.load(fp)
        fp.close()
        return json_data
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.json_data[index]['img_fn'])
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        keypoints = self.json_data[index]['keypoints']

        if self.trans is not None:
            img, keypoints = self.trans(img, keypoints)

        label = np.zeros((self.s, self.s, self.num_kpt), dtype=np.float32)
        offset = np.zeros((self.s, self.s, self.num_kpt*2), dtype=np.float32)
        for px in range(self.num_kpt):
            if keypoints[px*3+2] == 0 or keypoints[px*3+0] <= 0 or keypoints[px*3+0] >= self.size or keypoints[px*3+1] <= 0 or keypoints[px*3+1] >= self.size or keypoints[px*3+2] == 0:
                continue
            else:
                grid_loc_x = math.floor(keypoints[px*3+0] // self.grid_size)
                grid_loc_y = math.floor(keypoints[px*3+1] // self.grid_size)
                label[grid_loc_y][grid_loc_x][px] = 1
                offset[grid_loc_y][grid_loc_x][px] = (keypoints[px*3+0] % self.grid_size) / self.grid_size
                offset[grid_loc_y][grid_loc_x][self.num_kpt+px] = (keypoints[px*3+1] % self.grid_size) / self.grid_size
 
        img = normalize(to_tensor(img))
        label = to_tensor(label)
        offset = to_tensor(offset)
        
        return img, label, offset

    def __len__(self):
        return len(self.json_data)

class LSPMPIILIPVAL(data.DataLoader):
    
    def __init__(self, json_file, img_dir, size, num_kpt, s=7, sigma=7):
        self.json_data = self.read_json_file(json_file)
        self.img_dir = img_dir
        self.size = size
        self.num_kpt = num_kpt
        self.s = s
        self.sigma = sigma
        self.trans = TestResized(self.num_kpt, self.size)
        self.grid_size = self.size / self.s
        
    def read_json_file(self, json_file):
        fp = open(json_file)
        json_data = json.load(fp)
        fp.close()
        return json_data

    def _enhance(self, img, alpha):
        beta = 125 * (1 - alpha)
        img = np.uint8(np.clip((alpha * img + beta), 0, 255))
        return np.ascontiguousarray(img)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.json_data[index]['img_fn'])
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        keypoints = self.json_data[index]['keypoints']
        if 'bodysize' in self.json_data[index]:
            norm = self.json_data[index]['bodysize']
        elif 'headsize' in self.json_data[index]:
            norm = self.json_data[index]['headsize']
        else:
            norm = self.json_data[index]['normalize']

        img, keypoints, ratio = self.trans(img, keypoints)

        label = np.zeros((self.s, self.s, self.num_kpt), dtype=np.float32)
        offset = np.zeros((self.s, self.s, self.num_kpt*2), dtype=np.float32)

        for px in range(self.num_kpt):
            if keypoints[px*3+2] == 0 or keypoints[px*3+0] <= 0 or keypoints[px*3+0] >= self.size or keypoints[px*3+1] <= 0 or keypoints[px*3+1] >= self.size or keypoints[px*3+2] == 0:
                continue
            else:
                grid_loc_x = math.floor(keypoints[px*3+0] // self.grid_size)
                grid_loc_y = math.floor(keypoints[px*3+1] // self.grid_size)
                label[grid_loc_y][grid_loc_x][px] = 1
                offset[grid_loc_y][grid_loc_x][px] = (keypoints[px*3+0] % self.grid_size) / self.grid_size
                offset[grid_loc_y][grid_loc_x][self.num_kpt+px] = (keypoints[px*3+1] % self.grid_size) / self.grid_size

        img1 = self._enhance(img.copy(), 1.0)
        img2 = self._enhance(img.copy(), 1.5)
        img3 = self._enhance(img.copy(), 2.0)
        img0 = normalize(to_tensor(img)).unsqueeze(dim=0)
        img1 = normalize(to_tensor(img1)).unsqueeze(dim=0)
        img2 = normalize(to_tensor(img2)).unsqueeze(dim=0)
        img3 = normalize(to_tensor(img3)).unsqueeze(dim=0)
        img = img0
        label = to_tensor(label)
        offset = to_tensor(offset)
        norm = norm * ratio
        
        return img, label, offset, norm

    def __len__(self):
        return len(self.json_data)