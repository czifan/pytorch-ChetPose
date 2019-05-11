import torch
import os
import math
import yaml
import numpy as np
from easydict import EasyDict as edict 

def display_one_img(plt, keypoints, limbSeq, colors):        
    new_keypoints = [0] * 48
    if keypoints[2*3+2] and keypoints[3*3+2]:
        new_keypoints[15*3+0] = (keypoints[2*3+0] + keypoints[3*3+0]) / 2
        new_keypoints[15*3+1] = (keypoints[2*3+1] + keypoints[3*3+1]) / 2
        new_keypoints[15*3+2] = 1
    if keypoints[8*3+2] and keypoints[9*3+2]:
        new_keypoints[14*3+0] = (keypoints[8*3+0] + keypoints[9*3+0]) / 2
        new_keypoints[14*3+1] = (keypoints[8*3+1] + keypoints[9*3+1]) / 2
        new_keypoints[14*3+2] = 1
    new_keypoints[:42] = keypoints[:]
    keypoints = new_keypoints
    for px in range(16):
        if keypoints[px*3+2]:
            x = keypoints[px*3+0]
            y = keypoints[px*3+1]
            plt.scatter([[x]], [[y]], color=colors[px])
    for (i, (px1, px2)) in enumerate(limbSeq):
        if keypoints[px1*3+2] and keypoints[px2*3+2]:
            plt.plot([keypoints[px1*3+0], keypoints[px2*3+0]], 
                     [keypoints[px1*3+1], keypoints[px2*3+1]], 
                     color=colors[i], linewidth=2)

def save_model(state, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_lr(optimizer, epoch, decay, lr_gamma):
    if epoch in decay:
        for (i, param_group) in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * lr_gamma
    return optimizer.state_dict()['param_groups'][0]['lr']

def adjust_weights(weights, epoch):
    # TODO
    pass
 
def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))

    return parser

def Accuracy(logits, targets, num_kpt):

    batch_size = targets.shape[0]
    acc = 0.
    uacc = 0.
    for b in range(batch_size):
        for px in range(num_kpt):
            logit = logits[b, px]
            target = targets[b, px]
            logit_loc = np.argwhere(logit == np.max(logit))
            target_loc = np.argwhere(target > 0)
            if len(target_loc) == 0:
                continue
            if (logit_loc == target_loc).all():
                acc += 1
            else:
                uacc += 1
    return acc / (acc + uacc)

def PCK(logits1, logits2, targets1, targets2, num_kpt, grid_size, threshold=0.2):

    batch_size = targets1.shape[0]
    # items = {'Head': [8, 9], 'Shou': [12, 13], 'Elbow': [11, 14], \
    #     'Wrist': [10, 15], 'Hip': [2, 3], 'Knee': [1, 4], 'Ankle': [0, 5]}
    items = {'Head': [12, 13], 'Shou': [8, 9], 'Elbow': [7, 10], \
        'Wrist': [6, 11], 'Hip': [2, 3], 'Knee': [1, 4], 'Ankle': [0, 5]}
    # acc = np.zeros(7)
    # uacc = np.zeros(7)
    acc  = {}
    uacc = {}

    pre_offset_x = np.zeros((batch_size, num_kpt))
    pre_offset_y = np.zeros((batch_size, num_kpt))
    lab_offset_x = np.zeros((batch_size, num_kpt))
    lab_offset_y = np.zeros((batch_size, num_kpt))
    vis = np.ones((batch_size, num_kpt))
    norm = np.ones((batch_size, 1))

    for b in range(batch_size):
        for px in range(num_kpt):
            logit = logits1[b, px]
            target = targets1[b, px]
            logit_loc = np.argwhere(logit == np.max(logit))
            target_loc = np.argwhere(target > 0)
            if len(target_loc) == 0:
                vis[b, px] = 0
                continue
            logit_y, logit_x = logit_loc[0][0], logit_loc[0][1]
            target_y, target_x = target_loc[0][0], target_loc[0][1]
            pre_offset_x[b, px] = (logit_x + logits2[b, px, logit_y, logit_x]) * grid_size
            pre_offset_y[b, px] = (logit_y + logits2[b, num_kpt+px, logit_y, logit_x]) * grid_size
            lab_offset_x[b, px] = (target_x + targets2[b, px, target_y, target_x]) * grid_size
            lab_offset_y[b, px] = (target_y + targets2[b, num_kpt+px, target_y, target_x]) * grid_size

        # dis1 = math.sqrt(math.pow(lab_offset_x[b, 9]-lab_offset_x[b, 8], 2)+math.pow(lab_offset_y[b, 9]-lab_offset_y[b, 8], 2))
        # dis2 = math.sqrt(math.pow(lab_offset_x[b, 8]-lab_offset_x[b, 3], 2)+math.pow(lab_offset_y[b, 8]-lab_offset_y[b, 3], 2))
        # norm[b, 0] = max(max(dis1, dis2), grid_size)
        dis1 = math.sqrt(math.pow(lab_offset_x[b, 9]-lab_offset_x[b, 3], 2)+math.pow(lab_offset_y[b, 9]-lab_offset_y[b, 3], 2))
        # norm[b, 0] = max(dis1, grid_size*2)
        if dis1 == 0:
             dis1 = math.sqrt(math.pow(lab_offset_x[b, 8]-lab_offset_x[b, 2], 2)+math.pow(lab_offset_y[b, 8]-lab_offset_y[b, 2], 2))
        norm[b, 0] = dis1

    # print('Model    ', end='\t')
    for (idx, (key, value)) in enumerate(items.items()):
        # print(key, end='\t')
        tmp_vis = vis[:, value]
        tmp_pre_offset_x = pre_offset_x[:, value] * tmp_vis
        tmp_pre_offset_y = pre_offset_y[:, value] * tmp_vis
        tmp_lab_offset_x = lab_offset_x[:, value] * tmp_vis
        tmp_lab_offset_y = lab_offset_y[:, value] * tmp_vis
        dis = ((tmp_pre_offset_x-tmp_lab_offset_x)*(tmp_pre_offset_x-tmp_lab_offset_x) + 
            (tmp_pre_offset_y-tmp_lab_offset_y)*(tmp_pre_offset_y-tmp_lab_offset_y)) / (norm * norm)
        # acc[idx] = (dis <= (threshold * threshold)).sum()
        # uacc[idx] = (dis > (threshold * threshold)).sum()
        acc[key] = (dis <= (threshold * threshold)).sum()
        uacc[key] = (dis > (threshold * threshold)).sum()
        # pckh[key] = acc / (acc + uacc)
        # mean_pckh += (acc / (acc + uacc))
    # print('Mean')

    # print('PCKh@{:.1f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
    #     threshold, pckh['Head'], pckh['Shou'], pckh['Elbow'], pckh['Wrist'], pckh['Hip'], pckh['Knee'],
    #     pckh['Ankle'], mean_pckh/7))

    return acc, uacc
