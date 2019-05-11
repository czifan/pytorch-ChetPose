import torch
import torch.nn as nn
import numpy as np 
import math 
from detection.utils import bbox_iou

class YOLOLoss(nn.Module):
	def __init__(self, anchors, num_classes, img_size):
		super(YOLOLoss, self).__init__()
		self.anchors = anchors
		self.num_anchors = len(anchors)
		self.num_classes = num_classes
		self.bbox_attrs = 5 + num_classes
		self.img_size = img_size 

		self.ignore_threshold = 0.5
		self.lambda_xy = 2.5
		self.lambda_wh = 2.5
		self.lambda_conf = 1.0
		self.lambda_cls = 1.0

		self.mse_loss = nn.MSELoss()
		self.bce_loss = nn.BCELoss()

	def forward(self, input, targets=None):
		bs = input.size(0)
		in_h = input.size(2)
		in_w = input.size(3)
		stride_h = self.img_size[1] / in_h
		stride_w = self.img_size[0] / in_w
		scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

		prediction = input.view(bs,  self.num_anchors, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

		# Get outputs
		x = torch.sigmoid(prediction[..., 0])		  # Center x
		y = torch.sigmoid(prediction[..., 1])		  # Center y
		w = prediction[..., 2]						 # Width
		h = prediction[..., 3]						 # Height
		conf = torch.sigmoid(prediction[..., 4])	   # Conf
		pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.


		if targets is None:
			FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
			LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
			# Calculate offstes for each grid
			# Calculate offsets for each grid
			grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
				bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
			grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
				bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
			# Calculate anchor w, h
			anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
			anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
			anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
			anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
			# Add offset and scale with anchors
			pred_boxes = FloatTensor(prediction[..., :4].shape)
			pred_boxes[..., 0] = x.data + grid_x
			pred_boxes[..., 1] = y.data + grid_y
			pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
			pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
			# Results
			_scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
			output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
								conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
			return output.data