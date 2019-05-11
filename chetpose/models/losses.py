import torch
import torch.nn as nn

import torch.nn as nn
class Focal2DLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2):
        super(Focal2DLoss, self).__init__()
        self.base_criterion = nn.MSELoss()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logit, label):
        pos_label = label
        pos_logit = logit * pos_label
        pos_beta = (pos_label.sum().item() - pos_logit.sum().item()) / (pos_label.sum().item() + 1e-5)
        pos_loss = self.alpha * (pos_beta ** self.gamma) * self.base_criterion(pos_logit, pos_label)
        
        neg_label = 1 - label
        neg_logit = logit * neg_label
        neg_beta = 1 - (neg_label.sum().item() - neg_logit.sum().item()) / (neg_label.sum().item() + 1e-5)
        neg_loss = (1 - self.alpha) * (neg_beta ** self.gamma) * self.base_criterion(neg_logit + label, label)
        
        return pos_loss + neg_loss

class cpnLoss(nn.Module):
	def __init__(self, num_kpt):
		super(cpnLoss, self).__init__()
		self.criterion1 = nn.MSELoss()
		self.criterion2 = nn.MSELoss(reduction='none')
		self.num_kpt = num_kpt

	def _ohkm(self, loss, top_k):
		ohkm_loss = 0.
		for i in range(loss.size()[0]):
			sub_loss = loss[i]
			topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
			tmp_loss = torch.gather(sub_loss, 0, topk_idx)
			ohkm_loss += torch.sum(tmp_loss) / top_k
		ohkm_loss /= loss.size()[0]
		return ohkm_loss

	def _compute_heat_loss(self, logit, target):
		y_logit = logit * target
		n_logit = logit * (1 - target) + target
		y_weight = (logit.size(2) * logit.size(3) - 1) / 2
		n_weight = 1
		loss = self.criterion1(y_logit, target) * y_weight \
			+ self.criterion1(n_logit, target) * n_weight
		return loss

	def _compute_off_loss(self, logit, target, target_heat, reduce=True):
		x_logit = logit[:, :self.num_kpt, :, :] * target_heat
		y_logit = logit[:, self.num_kpt:, :, :] * target_heat
		x_target = target[:, :self.num_kpt, :, :]
		y_target = target[:, self.num_kpt:, :, :]
		if reduce:
			loss = self.criterion1(x_logit, x_target) + \
				self.criterion1(y_logit, y_target)
		else:
			loss = self.criterion2(x_logit, x_target) + \
				self.criterion2(y_logit, y_target)
		return loss

	def forward(self, global_outs, global_off_out, 
		refine_heat_out, refine_off_out, target_heat, target_off):
		loss = 0.
		global_loss_record = 0.
		refine_loss_record = 0.
		for global_out in global_outs:
			global_heat_loss = self._compute_heat_loss(global_out, target_heat)
			loss += global_heat_loss 
			global_loss_record += global_heat_loss.data.item()
		global_off_loss = self._compute_off_loss(global_off_out, target_off, target_heat)
		loss += global_off_loss
		global_loss_record += global_off_loss.data.item()
		refine_heat_loss = self.criterion2(refine_heat_out, target_heat)
		refine_off_loss = self._compute_off_loss(refine_off_out, target_off, target_heat)
		refine_loss = refine_heat_loss + refine_off_loss
		refine_loss = refine_loss.mean(dim=3).mean(dim=2)
		refine_loss = self._ohkm(refine_loss, self.num_kpt // 2)
		loss += refine_loss
		refine_loss_record += refine_loss.data.item()
		return loss, global_loss_record, refine_loss_record

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import  autograd

class MSELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logit, target):
        batch_size = logit.size(0)
        loss = (logit - target)
        loss = (loss * loss) / 2 / batch_size
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class HRPLoss(nn.Module):

    def __init__(self, num_kpt, s):
        super(HRPLoss, self).__init__()
        self.num_kpt = num_kpt
        self.s = s
        self.mean_criterion = nn.MSELoss()
        self.none_criterion = MSELoss(reduction='none')
        
    def _generate_mask(self, indexs, cuda):
        mask = torch.zeros(indexs.shape[0], self.num_kpt)
        if cuda:
            mask = mask.cuda()
        for b in range(indexs.shape[0]):
            for p in range(indexs.shape[1]):
                mask[b, indexs[b, p]] = 1
        return mask

    def forward(self, out1_1, out1_2, out2_1, out2_2, targets1, targets2, weights):
        loss1_1 = self.mean_criterion(out1_1, targets1)
        pre_offset_x = out1_2[:, :self.num_kpt, :, :] * targets1 
        pre_offset_y = out1_2[:, self.num_kpt:, :, :] * targets1 
        lab_offset_x = targets2[:, :self.num_kpt, :, :]
        lab_offset_y = targets2[:, self.num_kpt:, :, :]
        loss1_2 = self.mean_criterion(pre_offset_x, lab_offset_x) + self.mean_criterion(pre_offset_y, lab_offset_y)
        # refine
        loss2_1 = self.none_criterion(out2_1, targets1).sum(dim=2).sum(dim=2)
        _, indexs = torch.topk(loss2_1, self.num_kpt//2, 1)
        mask = self._generate_mask(indexs, targets1.is_cuda)
        loss2_1 = (loss2_1 * mask).mean()
        mask = mask.reshape(*mask.shape, 1, 1)
        pre_offset_x = out2_2[:, :self.num_kpt, :, :] * (targets1 * mask) 
        pre_offset_y = out2_2[:, self.num_kpt:, :, :] * (targets1 * mask) 
        lab_offset_x = targets2[:, :self.num_kpt, :, :] * (targets1 * mask)
        lab_offset_y = targets2[:, self.num_kpt:, :, :] * (targets1 * mask)
        loss2_2 = self.mean_criterion(pre_offset_x, lab_offset_x) + self.mean_criterion(pre_offset_y, lab_offset_y)
        # total
        loss = (loss1_1 + loss2_1) * weights[0] + (loss1_2 + loss2_2 * 5) * weights[1]
        # loss = loss1_1 * weights[0] + loss1_2 * weights[1]
        return loss