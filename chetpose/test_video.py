import numpy as np
import cv2
import os
import torch
import sys
sys.path.append('..')
from models.build_model import build_model
from utils import Config
import time

joints = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle', 'Right wrist', 'Right elbow', \
          'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist', 'Neck', 'Head top']
limbSeq = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 8], [3, 9], [8, 12], [9, 12], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
colors = [(255, 0, 0), (255, 84, 0), (255, 168, 0), (255, 255, 0), (168, 255, 0), (84, 255, 0), (0, 255, 0), \
        (0, 255, 84), (0, 255, 168), (0, 255, 255), (0, 168, 255), (0, 84, 255), (0, 0, 255), (84, 0, 255), \
        (168, 0, 255), (255, 0, 255), (255, 0, 168), (255, 0, 84)]

def eval_imgs(model, batch_imgPaths, target_size, num_kpt, grid_size, threshold=0.3):

    batch_size = len(batch_imgPaths)
    imgs = np.zeros((batch_size, 3, target_size, target_size), dtype=np.float32)
    ori_imgs = []
    ratios = np.zeros((batch_size), dtype=np.float32)
    
    for (idx, imgPath) in enumerate(batch_imgPaths):
        img = np.array(cv2.imread(imgPath), dtype=np.float32)
        ori_img = img.copy()
        ori_imgs.append(ori_img)
        h, w, _ = img.shape
        ratio = min(target_size/h, target_size/w)
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        h, w, _ = img.shape
        new_img = np.empty((target_size, target_size, 3), dtype=np.float32)
        new_img.fill(128)
        new_img[:h, :w, :] = img[:, :, :]
        new_img = new_img.transpose((2, 0, 1))
        new_img = np.ascontiguousarray(new_img)
        new_img = (new_img - 128.) / 256.
        imgs[idx] = new_img
        ratios[idx] = ratio

    imgs = torch.from_numpy(imgs)
    imgs = imgs.cuda().float()
    logits1, logits2, _, _ = model(imgs)
    logits1 = (logits1.data).cpu().numpy()
    logits2 = (logits2.data).cpu().numpy()
    keypoints = np.zeros((batch_size, num_kpt*3), np.int32)

    for b in range(batch_size):
        ori_img = ori_imgs[b]
        height, width, _ = ori_img.shape
        new_img = np.zeros((ori_img.shape[0], int(ori_img.shape[1]*3), ori_img.shape[2]))
        new_img.fill(128)
        another_img = np.zeros((height, width, _))
        another_img.fill(255)
        new_img[:, :ori_img.shape[1], :] = ori_img
        for px in range(num_kpt):
            logit = logits1[b, px]
            if np.max(logit) < threshold:
                continue
            logit_loc = np.argwhere(logit == np.max(logit))
            logit_y, logit_x = logit_loc[0][0], logit_loc[0][1]
            keypoints[b, px*3+0] = int(((logit_x + logits2[b, px, logit_y, logit_x]) * grid_size) / ratios[b])
            keypoints[b, px*3+1] = int(((logit_y + logits2[b, num_kpt+px, logit_y, logit_x]) * grid_size) / ratios[b])
            keypoints[b, px*3+2] = 1
            cv2.circle(ori_img, (keypoints[b, px*3+0], keypoints[b, px*3+1]), 4, colors[px], 4)
            cv2.circle(another_img, (keypoints[b, px*3+0], keypoints[b, px*3+1]), 4, colors[px], 4)

        for (i, (px1, px2)) in enumerate(limbSeq):
            if keypoints[b, px1*3+2] and keypoints[b, px2*3+2]:
                cv2.line(ori_img, (keypoints[b, px1*3+0], keypoints[b, px1*3+1]), (keypoints[b, px2*3+0], keypoints[b, px2*3+1]), colors[i], 3)
                cv2.line(another_img, (keypoints[b, px1*3+0], keypoints[b, px1*3+1]), (keypoints[b, px2*3+0], keypoints[b, px2*3+1]), colors[i], 3)

        new_img[:, width:2*width, :] = ori_img
        new_img[:, -width:, :] = another_img
        cv2.imwrite(batch_imgPaths[b].replace(imgs_dir, save_dir), new_img)

def picvideo(path, save_video):
    filelist = os.listdir(path)

    writer = None
    for i in range(1, len(filelist)+1):
        item = os.path.join(path, '{}.jpg'.format(i))
        img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(save_video, fourcc, 20,
                                 (img.shape[1], img.shape[0]), True)
        if writer is not None:
            writer.write(img)

def parsevideo(path, save_path):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    i=0
    while(cap.isOpened()):
        i=i+1
        if i > 200:
            break
        ret, frame = cap.read()
        if ret==True:
            cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), frame[250:-250, 300:-300, :])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

cv2.destroyAllWindows()

config = Config('../config.yaml')
config.pre_trained = True
config.model_path = '../checkpoint/best_models/resnet101_best.pth.tar'
config.model_type = 'resnet101'
model = build_model(config)
model = model.eval()
imgs_dir = '/home/czifan/SingleYoloPose/cache/images/'
save_dir = '/home/czifan/SingleYoloPose/cache/results/'
save_video = '/home/czifan/SingleYoloPose/cache/test.avi'
video_path = '/home/czifan/SingleYoloPose/cache/video2.mp4'
imgs_num = len(os.listdir(imgs_dir))
batch_size = 32

parsevideo(video_path, imgs_dir)

for i in range(1, imgs_num+1, batch_size):
    print('Process {}'.format(i))
    batch_imgPaths = [os.path.join(imgs_dir, '{}.jpg'.format(idx)) for idx in range(i, min(imgs_num+1, i+batch_size))]
    eval_imgs(model, batch_imgPaths, config.size, config.num_kpt, config.size//config.s)

picvideo(save_dir, save_video)
