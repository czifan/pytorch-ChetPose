import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from loader import LSPMPIILIPVAL
import sys
from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import os

mpii_dict = {'head': 13, 'lsho': 9, 'lelb': 10, 'lwri': 11, 'lhip': 3, 'lkne': 4, 'lank': 5,
	'rsho': 8, 'relb': 7, 'rwri': 6, 'rkne': 1, 'rank': 0, 'rhip': 2}

def parse_result(out1_1, out1_2, index, config):
	pos_pred_src = np.zeros((config.num_kpt, 2))

	out1_1 = (out1_1.data).cpu().numpy()[index]
	out1_2 = (out1_2.data).cpu().numpy()[index]

	for px in range(config.num_kpt):
		logit = out1_1[px]
		logit_loc = np.argwhere(logit == np.max(logit))
		logit_y, logit_x = logit_loc[0][0], logit_loc[0][1]
		pre_x = (logit_x + out1_2[px, logit_y, logit_x]) * (config.size / config.s)
		pre_y = (logit_y + out1_2[config.num_kpt+px, logit_y, logit_x]) * (config.size / config.s)
		pos_pred_src[px, 0] = pre_x
		pos_pred_src[px, 1] = pre_y

	return pos_pred_src

def compute_pck(config, valloader, model, data_type):

	pos_pred_src = np.zeros((config.num_kpt, 2, len(valloader)))
	pos_gt_src = np.zeros((config.num_kpt, 2, len(valloader)))
	jnt_visible = np.zeros((config.num_kpt, len(valloader)))
	norms = np.zeros((len(valloader)))

	for (idx, (img, label, offset, norm)) in enumerate(valloader):
		if config.cuda:
			img = img.cuda()
			label = label.cuda()
			offset = offset.cuda()
		img = img.float()[0]
		norm = norm.float().numpy()[0]

		out1_1, out1_2 = model(img)

		lab1_1 = (label.data).cpu().numpy()[0]
		lab1_2 = (offset.data).cpu().numpy()[0]
		norms[idx] = norm

		for px in range(config.num_kpt):
			target = lab1_1[px]
			target_loc = np.argwhere(target >= 1.)
			if len(target_loc) == 0:
				continue
			target_y, target_x = target_loc[0][0], target_loc[0][1]
			lab_x = (target_x + lab1_2[px, target_y, target_x]) * (config.size / config.s)
			lab_y = (target_y + lab1_2[config.num_kpt+px, target_y, target_x]) * (config.size / config.s)
			pos_gt_src[px, 0, idx] = lab_x
			pos_gt_src[px, 1, idx] = lab_y
			jnt_visible[px, idx] = 1

		pos_pred_srcs = [parse_result(out1_1, out1_2, index, config) for index in range(img.size(0))]
		for px in range(config.num_kpt):
			for index in range(len(pos_pred_srcs)):
				pos_pred_src[px, 0, idx] += pos_pred_srcs[index][px, 0] / len(pos_pred_srcs)
				pos_pred_src[px, 1, idx] += pos_pred_srcs[index][px, 1] / len(pos_pred_srcs)

	if data_type == 'mpii':
		threshold = config.mpii_pckh_thres
		SC_BIAS = config.SC_BIAS
		model_name = 'MPII:PCKh@{:.1f}'.format(threshold)
		norms *= SC_BIAS
	elif data_type == 'lsp':
		threshold = config.lsp_pck_thres
		model_name = 'LSP:PCK@{:.1f}'.format(threshold)
	elif data_type == 'lip':
		threshold = config.lip_pck_thres
		model_name = 'LIP:PCK@{:.1f}'.format(threshold)

	uv_error = pos_pred_src - pos_gt_src
	uv_err = np.linalg.norm(uv_error, axis=1)
	scale = np.multiply(norms, np.ones((len(uv_err), 1)))
	scaled_uv_err = np.divide(uv_err, scale)
	scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
	jnt_count = np.sum(jnt_visible, axis=1)
	less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
	PCK = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)
	PCK = np.ma.array(PCK, mask=False)

	print("Model\t\tHead\tSho.\tElb.\tWri.\tHip\tKnee\tAnk.\tMean")
	print('{:8s}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(
			model_name, 
			PCK[mpii_dict['head']], 
			0.5 * (PCK[mpii_dict['lsho']] + PCK[mpii_dict['rsho']]), 
	        0.5 * (PCK[mpii_dict['lelb']] + PCK[mpii_dict['relb']]), 
	        0.5 * (PCK[mpii_dict['lwri']] + PCK[mpii_dict['rwri']]), 
	        0.5 * (PCK[mpii_dict['lhip']] + PCK[mpii_dict['rhip']]), 
	        0.5 * (PCK[mpii_dict['lkne']] + PCK[mpii_dict['rkne']]), 
	        0.5 * (PCK[mpii_dict['lank']] + PCK[mpii_dict['rank']]), 
	        np.mean(PCK)))

	return np.mean(PCK)


def evaluate(config, model):

	ave_pck = 0.
	model = model.eval()

	valloader = DataLoader(
		LSPMPIILIPVAL(config.val_json_file,
			config.val_img_dir,
			config.size,
			config.num_kpt,
			config.s,
			), batch_size=1, shuffle=False, 
		num_workers=config.num_workers, pin_memory=True)

	ave_pck = compute_pck(config, valloader, model, 'lsp')

	return ave_pck