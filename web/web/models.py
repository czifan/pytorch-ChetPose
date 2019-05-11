import sys
sys.path.append('..')
from detection.model import ModelMain
from detection.loss import YOLOLoss
from detection.utils import non_max_suppression, bbox_iou
import cv2
import os
import torch
import random
import matplotlib 
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from chetpose.models.chetpose import resnet101
import numpy as np
import shutil
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 100)]

config = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "confidence_threshold": 0.5,
    "classes_names_path": "../detection/coco.names",
    "img_h": 416,
    "img_w": 416,
}

def display_image(plt, keypoints):
    limbSeq = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [12, 14], [14, 15]]
    colors = [[1., 0., 0.], [1., 0.33, 0], [1., 0.66, 0], [1., 1., 0], [0.66, 1., 0], [0.33, 1., 0], [0, 1., 0], [0, 1., 0.33], [0, 1., 0.66], [0, 1., 1.], [0, 0.66, 1.], [0, 0.33, 1.], [0, 0, 1.], [0.33, 0, 1.], [0.66, 0, 1.], [1., 0, 1.], [1., 0, 0.66], [1., 0, 0.33]]
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

def tensor2keypoints(tensor, threshold=0.3):
	num_kpt = tensor.shape[0] // 3
	heat_tensor = tensor[:num_kpt]
	offx_tensor = tensor[num_kpt:2*num_kpt]
	offy_tensor = tensor[2*num_kpt:]
	keypoints = [0] * (num_kpt * 3)
	for px in range(num_kpt):
		tmp_max = np.max(heat_tensor[px])
		coor = np.argwhere(heat_tensor[px] == tmp_max)
		if len(coor) <= 0 or tmp_max <= threshold:
			continue
		grid_y, grid_x = coor[0][0], coor[0][1]
		offset_y, offset_x = offy_tensor[px, grid_y, grid_x], offx_tensor[px, grid_y, grid_x]
		keypoints[px*3+0] = (grid_x + offset_x) * 32
		keypoints[px*3+1] = (grid_y + offset_y) * 32
		keypoints[px*3+2] = 1
	return keypoints

def resized_image(img):
	new_img = np.empty((224, 224, 3), dtype=np.float32)
	new_img.fill(128)
	height, width, _ = img.shape
	ratio = min(224 / height, 224 / width)
	img = np.ascontiguousarray(cv2.resize(img, (0, 0), fx=ratio, fy=ratio))
	height, width, _ = img.shape
	offset_left = (224 - width) // 2
	offset_up = (224 - height) // 2
	new_img[offset_up:offset_up+height, offset_left:offset_left+width, :] = img[:, :, :]
	details = {}
	details['ratio'] = ratio
	details['offset_left'] = offset_left
	details['offset_up'] = offset_up
	return new_img, details

def chetpose_test_image(model, imgs):
	model.eval()
	images, ratios, offset_lefts, offset_ups, results = [], [], [], [], []
	for img in imgs:
		img, details = resized_image(img)
		img = img / 255. - 0.5
		img = np.transpose(img, (2, 0, 1))
		img = img.astype(np.float32)
		images.append(img)
		ratios.append(details['ratio'])
		offset_lefts.append(details['offset_left'])
		offset_ups.append(details['offset_up'])
	images = np.asarray(images)
	ori_images = np.transpose(((images + 0.5) * 255.).copy(), (0, 2, 3, 1)).astype(np.uint8)
	images = torch.from_numpy(images)
	with torch.no_grad():
		outputs = model(images)
		for idx, output in enumerate(outputs):
			save_dir = './static/output/predict/{}/'.format(idx)
			if not os.path.isdir(save_dir):
				os.makedirs(save_dir)
			plt.figure()
			fig, axarr = plt.subplots(1)
			axarr.imshow(ori_images[idx])
			axarr.axis('off')
			plt.gca().xaxis.set_major_locator(NullLocator())
			plt.gca().yaxis.set_major_locator(NullLocator())
			plt.savefig(os.path.join(save_dir, 'input.jpg'.format()), bbox_inches='tight', pad_inches=0.0)
			plt.close('all')
			print(output.shape, outputs.shape)
			keypoints = tensor2keypoints((output.data).cpu().numpy())
			for px in range(14):
				if not keypoints[px*3+2]:
					plt.figure()
					fig, axarr = plt.subplots(1)
					axarr.imshow(np.zeros((7, 7)))
					axarr.axis('off')
					plt.gca().xaxis.set_major_locator(NullLocator())
					plt.gca().yaxis.set_major_locator(NullLocator())
					plt.savefig(os.path.join(save_dir, 'heat_{}.jpg'.format(px)), bbox_inches='tight', pad_inches=0.0)
					plt.close('all')
					plt.figure()
					fig, axarr = plt.subplots(1)
					axarr.imshow(ori_images[idx], alpha=0.8)
					axarr.axis('off')
					plt.gca().xaxis.set_major_locator(NullLocator())
					plt.gca().yaxis.set_major_locator(NullLocator())
					plt.savefig(os.path.join(save_dir, 'offset_{}.jpg'.format(px)), bbox_inches='tight', pad_inches=0.0)
					plt.close('all')
					continue
				grid_y, grid_x = keypoints[px*3+1] // 32, keypoints[px*3+0] // 32
				offset_y, offset_x = (keypoints[px*3+1] % 32) / 32, (keypoints[px*3+0] % 32) / 32
				# heat
				plt.figure()
				fig, axarr = plt.subplots(1)
				axarr.imshow((output[px].data).cpu().numpy())
				# axarr.imshow(ori_images[idx])
				axarr.axis('off')
				# rect = patches.Rectangle((grid_x * 32, grid_y * 32), 32, 32, edgecolor='r', facecolor='r')
				# axarr.add_patch(rect)
				plt.gca().xaxis.set_major_locator(NullLocator())
				plt.gca().yaxis.set_major_locator(NullLocator())
				plt.savefig(os.path.join(save_dir, 'heat_{}.jpg'.format(px)), bbox_inches='tight', pad_inches=0.0)
				plt.close('all')
				# offset
				plt.figure()
				fig, axarr = plt.subplots(1)
				axarr.imshow(ori_images[idx], alpha=0.8)
				axarr.axis('off')
				bbox = patches.Rectangle((max(2, grid_x * 32), max(2, grid_y * 32)), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
				axarr.add_patch(bbox)
				axarr.plot([keypoints[px*3+0], grid_x * 32], [keypoints[px*3+1], keypoints[px*3+1]], color='r', linewidth=3.5, linestyle=':')
				axarr.plot([keypoints[px*3+0], keypoints[px*3+0]], [keypoints[px*3+1], grid_y * 32], color='r', linewidth=3.5, linestyle=':')
				plt.gca().xaxis.set_major_locator(NullLocator())
				plt.gca().yaxis.set_major_locator(NullLocator())
				plt.savefig(os.path.join(save_dir, 'offset_{}.jpg'.format(px)), bbox_inches='tight', pad_inches=0.0)
				plt.close('all')
				keypoints[px*3+0] = (keypoints[px*3+0] - offset_lefts[idx]) / ratios[idx]
				keypoints[px*3+1] = (keypoints[px*3+1] - offset_ups[idx]) / ratios[idx]
			results.append(keypoints)
	return results

def detection_test_image(model, pose_model, image_path):
	model.eval()

	yolo_losses = []
	for i in range(3):
		yolo_losses.append(YOLOLoss(config['yolo']['anchors'][i],
			config['yolo']['classes'], (config['img_w'], config['img_h'])))

	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	height, width, _ = image.shape
	size = max(height, width)
	new_image = np.empty((size, size, 3), dtype=np.float32)
	new_image.fill(128)
	new_image[(size-height)//2:(size-height)//2+height, (size-width)//2:(size-width)//2+width, :] = image[:, :, :]
	image = new_image.astype(np.uint8)
	image_origin = image.copy()
	image = cv2.resize(image, (config['img_w'], config['img_h']),
		interpolation=cv2.INTER_LINEAR)
	image = image.astype(np.float32)
	image /= 255.
	image = np.transpose(image, (2, 0, 1))
	image = image.astype(np.float32)
	images = np.asarray([image])
	images = torch.from_numpy(images)
	# images = images.cuda()
	with torch.no_grad():
		outputs = model(images)
		output_list = []
		for i in range(3):
			output_list.append(yolo_losses[i](outputs[i]))
		output = torch.cat(output_list, 1)
		batch_detections = non_max_suppression(output, config['yolo']['classes'],
			conf_thres=config['confidence_threshold'], nms_thres=0.45)
	classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
	if not os.path.isdir("./static/output/"):
		os.makedirs("./static/output/")
	if not os.path.isdir("./static/output/predict/"):
		os.makedirs("./static/output/predict/")
	if not os.path.isdir("./static/output/predict/"):
		os.makedirs("./static/output/predict/")
	print(os.getcwd())
	shutil.rmtree("./static/output/predict")
	save_image_paths = []
	if len(batch_detections) > 0:
		detections = batch_detections[0]
		unique_labels = detections[:, -1].cpu().unique()
		n_cls_preds = len(unique_labels)
		index = 0
		crop_images, offset_lefts, offset_ups = [], [], []

		for focus_idx, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
			if detections[focus_idx][-1] != 0:
				continue
			plt.figure()
			fig, ax = plt.subplots(1)
			ax.imshow(image_origin)
			# Rescale coordinates to original dimensions
			ori_h, ori_w = image_origin.shape[:2]
			pre_h, pre_w = config["img_h"], config["img_w"]
			box_h = ((y2 - y1) / pre_h) * ori_h
			box_w = ((x2 - x1) / pre_w) * ori_w
			y1 = (y1 / pre_h) * ori_h
			x1 = (x1 / pre_w) * ori_w
			if box_w * box_h < 15000:
				continue
			person_x1 = max(0, int(x1 - 0.1 * ori_w))
			person_y1 = max(0, int(y1 - 0.1 * ori_h))
			person_x2 = min(ori_w, int(x1 + 1.1 * box_w))
			person_y2 = min(ori_h, int(y1 + 1.1 * box_h))
			crop_images.append(image_origin[person_y1: person_y2, person_x1: person_x2, :].copy())
			offset_lefts.append(person_x1)
			offset_ups.append(person_y1)	
			# Create a Rectangle patch
			bbox = patches.Rectangle((max(x1, 1), max(y1, 1)), box_w-1, box_h-1, linewidth=1,
									 edgecolor=colors[1],
									 facecolor='none')
			# Add the bbox to the plot
			ax.add_patch(bbox)
			# Add label
			plt.text(max(x1, 1), max(y1, 1), s=classes[int(cls_pred)], color='white',
					 verticalalignment='top',
					 bbox={'color': colors[1], 'pad': 0})
			save_dir = './static/output/predict/{}/'.format(index)
			if not os.path.isdir(save_dir):
				os.makedirs(save_dir)
			plt.axis('off')
			plt.gca().xaxis.set_major_locator(NullLocator())
			plt.gca().yaxis.set_major_locator(NullLocator())
			plt.savefig(os.path.join(save_dir, 'detection.jpg'), bbox_inches='tight', pad_inches=0.0)
			plt.close('all')
			index += 1
			save_image_paths.append(os.path.join(save_dir, 'detection.jpg'))

		# for focus_idx in range(len(detections)):
		# 	if detections[focus_idx][-1] != 0:
		# 		continue
		# 	plt.figure()
		# 	fig, ax = plt.subplots(1)
		# 	ax.imshow(image_origin)
		# 	flag = True
		# 	for idx, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
		# 		if cls_pred != 0:
		# 			continue
		# 		if idx == focus_idx:
		# 			color = colors[50]
		# 		else:
		# 			color = colors[1]
		# 		# Rescale coordinates to original dimensions
		# 		ori_h, ori_w = image_origin.shape[:2]
		# 		pre_h, pre_w = config["img_h"], config["img_w"]
		# 		box_h = ((y2 - y1) / pre_h) * ori_h
		# 		box_w = ((x2 - x1) / pre_w) * ori_w
		# 		y1 = (y1 / pre_h) * ori_h
		# 		x1 = (x1 / pre_w) * ori_w
		# 		if box_w * box_h < 15000:
		# 			if idx == focus_idx:
		# 				flag = False
		# 			continue
		# 		if idx == focus_idx:
		# 			person_x1 = max(0, int(x1 - 0.1 * ori_w))
		# 			person_y1 = max(0, int(y1 - 0.1 * ori_h))
		# 			person_x2 = min(ori_w, int(x1 + 1.1 * box_w))
		# 			person_y2 = min(ori_h, int(y1 + 1.1 * box_h))
		# 			crop_images.append(image_origin[person_y1: person_y2, person_x1: person_x2, :].copy())
		# 			offset_lefts.append(person_x1)
		# 			offset_ups.append(person_y1)
		# 		# Create a Rectangle patch
		# 		bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1,
		# 								 edgecolor=color,
		# 								 facecolor='none')
		# 		# Add the bbox to the plot
		# 		ax.add_patch(bbox)
		# 		# Add label
		# 		plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
		# 				 verticalalignment='top',
		# 				 bbox={'color': color, 'pad': 0})	
		# 	# Save generated image with detections
		# 	if not flag:
		# 		continue
		# 	save_dir = './static/output/predict/{}/'.format(index)
		# 	if not os.path.isdir(save_dir):
		# 		os.makedirs(save_dir)
		# 	plt.axis('off')
		# 	plt.gca().xaxis.set_major_locator(NullLocator())
		# 	plt.gca().yaxis.set_major_locator(NullLocator())
		# 	plt.savefig(os.path.join(save_dir, 'detection.jpg'), bbox_inches='tight', pad_inches=0.0)
		# 	plt.close('all')
		# 	index += 1
		# 	save_image_paths.append(os.path.join(save_dir, 'detection.jpg'))

	results = chetpose_test_image(pose_model, crop_images)

	for (image_idx, save_imaga_path) in enumerate(save_image_paths):
		plt.figure()
		fig, ax = plt.subplots(1)
		image_detection = cv2.imread(save_imaga_path, cv2.IMREAD_COLOR)
		image_detection = cv2.cvtColor(image_detection, cv2.COLOR_BGR2RGB)
		d_h, d_w, _ = image_detection.shape
		o_h, o_w, _ = image_origin.shape

		ax.imshow(image_detection)
		for (i, keypoints) in enumerate(results):
			if i != image_idx:
				continue
			for px in range(14):
				keypoints[px*3+0] = (keypoints[px*3+0] + offset_lefts[i]) / o_w * d_w
				keypoints[px*3+1] = (keypoints[px*3+1] + offset_ups[i]) / o_h * d_h
			display_image(ax, keypoints.copy())
		plt.axis('off')
		plt.gca().xaxis.set_major_locator(NullLocator())
		plt.gca().yaxis.set_major_locator(NullLocator())
		plt.savefig(save_imaga_path.replace('detection', 'pose'), bbox_inches='tight', pad_inches=0.0)
		plt.close('all')

