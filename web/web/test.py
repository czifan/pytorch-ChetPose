import sys
sys.path.append('../..')
from chetpose.models.chetpose import resnet101
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def tensor2keypoints(tensor):
	num_kpt = tensor.shape[0] // 3
	heat_tensor = tensor[:num_kpt]
	offx_tensor = tensor[num_kpt:2*num_kpt]
	offy_tensor = tensor[2*num_kpt:]
	keypoints = [0] * (num_kpt * 3)
	for px in range(num_kpt):
		tmp_max = np.max(heat_tensor[px])
		coor = np.argwhere(heat_tensor[px] == tmp_max)
		if len(coor) <= 0 or tmp_max <= 0:
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
	for px in range(self.num_kpt):
		keypoints[px*3+0] = keypoints[px*3+0] * ratio + offset_left
		keypoints[px*3+1] = keypoints[px*3+1] * ratio + offset_up
	new_img[offset_up:offset_up+height, offset_left:offset_left+width, :] = img[:, :, :]
	details = {}
	details['ratio'] = ratio
	details['offset_left'] = offset_left
	details['offset_up'] = offset_up
	return new_img, details

def chetpose_test_image(model, imgs):
	model.eval()
	images, ratios, offset_lefts, offset_ups, results = [], [], [], []
	for img in imgs:
		img, details = resized_image(img)
		img = img / 255. - 0.5
		img = np.transpose(img, (2, 0, 1))
		img = img.astype(np.float32)
		images.append(img)
	images = np.asarray(images)
	ori_images = np.transpose(((images + 0.5) * 255.).copy(), (0, 2, 3, 1))
	images = torch.from_numpy(images)
	with torch.no_grad():
		outputs = model(images)
		for idx, output in enumerate(outputs):
			if not os.path.isdir('../output/{}/'.format(idx)):
				os.makedirs('../output/{}/'.format(idx))
			plt.figure()
			fig, axarr = plt.subplots(1)
			axarr.imshow(ori_images[idx])
			axarr.axis('off')
			plt.savefig(os.path.join(save_dir, 'input.jpg'.format()))
			plt.close()
			keypoints = tensor2keypoints((output.data).cpu().numpy())
			for px in range(14):
				grid_y, grid_x = keypoints[px*3+1] // 32, keypoints[px*3+0] // 32
				offset_y, offset_x = (keypoints[px*3+1] % 32) / 32, (keypoints[px*3+0] % 32) / 32
				# heat
				plt.figure()
				fig, axarr = plt.subplots(1)
				axarr.imshow(ori_images[idx])
				axarr.axis('off')
				rect = plt.Rectangle((grid_x * 32, grid_y * 32), 32, 32, edgecolor='r',facecolor='r')
				axarr.add_patch(rect)
				plt.savefig(os.path.join(save_dir, 'heat_{}.jpg'.format(px)))
				plt.close()
				# offset
				plt.figure()
				fig, axarr = plt.subplots(1)
				axarr.imshow(ori_images[idx])
				axarr.axis('off')
				bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
				ax.add_patch(bbox)
				axarr.add_patch(rect)
				axarr.plot([keypoints[px*3+0], grid_x * 32], [keypoints[px*3+1], keypoints[px*3+1]], color='r', linewidth=3, linestyle=':')
				axarr.plot([keypoints[px*3+0], keypoints[px*3+0]], [keypoints[px*3+1], grid_y * 32], color='r', linewidth=3, linestyle=':')
				plt.savefig(os.path.join(save_dir, 'heat_{}.jpg'.format(px)))
				plt.close()
				keypoints[px*3+0] = (keypoints[px*3+0] - offset_left) / ratio 
				keypoints[px*3+1] = (keypoints[px*3+1] - offset_up) / ratio 
			results.append(keypoints)
	return results



model_path = '../../chetpose/weights/chetpose.pth.tar'
model = resnet101(14, pretrained=False)
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location='cpu')
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict) 