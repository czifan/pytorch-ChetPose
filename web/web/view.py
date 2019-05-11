from django.shortcuts import render, render_to_response
from django.http import HttpResponse
import os
import torch
from .models import detection_test_image, config
from detection.model import ModelMain
from detection.loss import YOLOLoss
from detection.utils import non_max_suppression, bbox_iou
from chetpose.models.chetpose import resnet101

image_dir = os.path.join('output', 'predict')
image_path = None
cur_index = 0
model, pose_model = None, None

def index(request):
	global image_dir
	global cur_index
	cur_index = 0
	print(image_dir, cur_index)
	sample_img0 = os.path.join(image_dir, str(cur_index), 'pose.jpg')
	sample_img1 = os.path.join(image_dir, str(cur_index), 'input.jpg')
	sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_0.jpg')
	sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_0.jpg')
	print(sample_img0)
	return render_to_response('index.html', 
		{'images0': sample_img0, 'images1': sample_img1,
		 'images2': sample_img2, 'images3': sample_img3})

def process(request):
	global image_dir
	global cur_index
	global model
	global pose_model
	global image_path
	keys = request.GET.keys()
	num_files = len(os.listdir(os.path.join('static', image_dir)))
	image_path = request.GET['path']
	if 'update.x' in keys:
		cur_index = (cur_index + 1) % num_files
	elif 'predict.x' in keys:
		cur_index = 0
		if model is None or pose_model is None:
			model = ModelMain(config, is_training=False)
			state_dict = torch.load('../detection/weights/yolov3_weights_pytorch.pth', map_location='cpu')
			state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
			model.load_state_dict(state_dict)
			pose_model = resnet101(14, pretrained=False)
			pose_model_dict = pose_model.state_dict()
			pose_pretrained_dict = torch.load('../chetpose/weights/chetpose.pth.tar', map_location='cpu')
			pose_model_dict.update(pose_pretrained_dict)
			pose_model.load_state_dict(pose_model_dict) 
		image_dir = os.path.join('output', 'predict')
		detection_test_image(model, pose_model, image_path)
	sample_img0 = os.path.join(image_dir, str(cur_index), 'pose.jpg')
	sample_img1 = os.path.join(image_dir, str(cur_index), 'input.jpg')
	sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_0.jpg')
	sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_0.jpg')
	return render_to_response('index.html', 
		{'images0': sample_img0, 'images1': sample_img1,
		 'images2': sample_img2, 'images3': sample_img3})


def change_px(request):
	global image_dir
	global cur_index
	sample_img0 = os.path.join(image_dir, str(cur_index), 'pose.jpg')
	sample_img1 = os.path.join(image_dir, str(cur_index), 'input.jpg')
	key = request.GET.keys()
	if 'rankle' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_0.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_0.jpg')
	elif 'rknee' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_1.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_1.jpg')
	elif 'rhip' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_2.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_2.jpg')
	elif 'lhip' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_3.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_3.jpg')
	elif 'lknee' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_4.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_4.jpg')
	elif 'lankle' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_5.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_5.jpg')
	elif 'rwrist' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_6.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_6.jpg')
	elif 'reblow' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_7.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_7.jpg')
	elif 'rshoulder' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_8.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_8.jpg')
	elif 'lshoulder' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_9.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_9.jpg')
	elif 'leblow' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_10.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_10.jpg')
	elif 'lwrist' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_11.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_11.jpg')
	elif 'neck' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_12.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_12.jpg')
	elif 'headtop' in key:
		sample_img2 = os.path.join(image_dir, str(cur_index), 'heat_13.jpg')
		sample_img3 = os.path.join(image_dir, str(cur_index), 'offset_13.jpg')

	return render_to_response('index.html', 
		{'images0': sample_img0, 'images1': sample_img1,
		 'images2': sample_img2, 'images3': sample_img3})
