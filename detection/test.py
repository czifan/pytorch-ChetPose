import sys
sys.path.append('..')
from detection.model import ModelMain
from detection.loss import YOLOLoss
from detection.utils import non_max_suppression, bbox_iou
import cv2
import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]


def test(config):
	model = ModelMain(config, is_training=False)
	model.eval()
	# model = model.cuda()

	if config['pretrain_snapshot']:
		state_dict = torch.load(config['pretrain_snapshot'], map_location='cpu')
		state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
		model.load_state_dict(state_dict)

	yolo_losses = []
	for i in range(3):
		yolo_losses.append(YOLOLoss(config['yolo']['anchors'][i],
			config['yolo']['classes'], (config['img_w'], config['img_h'])))

	images_name = os.listdir(config['images_path'])
	images_path = [os.path.join(config['images_path'], name) for name in images_name]

	batch_size = config['batch_size']
	for step in range(0, len(images_path), batch_size):
		images = []
		images_origin = []
		for path in images_path[step*batch_size: (step+1)*batch_size]:
			image = cv2.imread(path, cv2.IMREAD_COLOR)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			images_origin.append(image)
			image = cv2.resize(image, (config['img_w'], config['img_h']),
				interpolation=cv2.INTER_LINEAR)
			image = image.astype(np.float32)
			image /= 255.
			image = np.transpose(image, (2, 0, 1))
			image = image.astype(np.float32)
			images.append(image)
		images = np.asarray(images)
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
		if not os.path.isdir("./output/"):
			os.makedirs("./output/")
		for idx, detections in enumerate(batch_detections):
			plt.figure()
			fig, ax = plt.subplots(1)
			ax.imshow(images_origin[idx])
			if detections is not None:
				unique_labels = detections[:, -1].cpu().unique()
				n_cls_preds = len(unique_labels)
				print(n_cls_preds, len(colors))
				bbox_colors = random.sample(colors, n_cls_preds)
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
					if cls_pred != 0:
						continue
					color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
					# Rescale coordinates to original dimensions
					ori_h, ori_w = images_origin[idx].shape[:2]
					pre_h, pre_w = config["img_h"], config["img_w"]
					box_h = ((y2 - y1) / pre_h) * ori_h
					box_w = ((x2 - x1) / pre_w) * ori_w
					y1 = (y1 / pre_h) * ori_h
					x1 = (x1 / pre_w) * ori_w

					# Create a Rectangle patch
					bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
											 edgecolor=color,
											 facecolor='none')
					# Add the bbox to the plot
					ax.add_patch(bbox)
					# Add label
					plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
							 verticalalignment='top',
							 bbox={'color': color, 'pad': 0})
			# Save generated image with detections
			plt.axis('off')
			plt.gca().xaxis.set_major_locator(NullLocator())
			plt.gca().yaxis.set_major_locator(NullLocator())
			plt.savefig('output/{}_{}.jpg'.format(step, idx), bbox_inches='tight', pad_inches=0.0)
			plt.close()

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
    "batch_size": 16,
    "confidence_threshold": 0.5,
    "images_path": "./images/",
    "classes_names_path": "./coco.names",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "./weights/yolov3_weights_pytorch.pth",
}

if __name__ == '__main__':
	test(config)