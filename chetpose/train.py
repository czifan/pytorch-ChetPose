import utils
import time
import torch
from torch.utils.data import DataLoader
from models.losses import Focal2DLoss
from utils import save_model, adjust_lr, Config, adjust_weights, Accuracy
from transforms import Resized, RandomRotate, RandomCrop, RandomHorizontalFlip, RandomAddColorNoise, TestResized, RandomBrightnessContrast, Compose
from loader import LSPMPIILIP
import numpy as np
from models.build_model import build_model
from evaluate import evaluate
from config import config
import torch.nn as nn
import os

trainloader = DataLoader(
    LSPMPIILIP(config.train_json_file, 
        config.train_img_dir, 
        config.size, 
        config.num_kpt,
        config.s,
        trans = Compose([
            RandomAddColorNoise(config.num_kpt, config.min_gauss, config.max_gauss, config.percentage),
            RandomBrightnessContrast(config.num_kpt, config.min_alpha, config.max_alpha),
            RandomCrop(config.num_kpt, config.ratio_max_x, config.ratio_max_y, config.center_perturb_max),
            RandomRotate(config.num_kpt, config.max_degree),
            RandomHorizontalFlip(config.num_kpt, config.prob),
            Resized(config.num_kpt, config.size),
        ])), batch_size=config.batch_size, shuffle=True, 
    num_workers=config.num_workers, pin_memory=True)

model = build_model(config)

heat_criterion = Focal2DLoss()
offset_criterion = nn.SmoothL1Loss()
if config.cuda:
    heat_criterion = heat_criterion.cuda()
    offset_criterion = offset_criterion.cuda()
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0)
max_pck = config.max_pck
weights = [config.s * config.s * config.num_kpt, config.s * config.s * config.num_kpt * 2]

evaluate(config, model)

for epoch in range(config.start_epoch, config.end_epoch):

    model = model.train()
    lr = adjust_lr(optimizer, epoch, config.decay, config.lr_gamma)
    batch_loss, batch_hloss, batch_xloss, batch_yloss, batch_acc, batch = 0., 0., 0., 0., 0., 0.
    for (idx, (img, label, offset)) in enumerate(trainloader):
        if config.cuda:
            img = img.cuda()
            label = label.cuda()
            offset = offset.cuda()
        img = img.float()

        out1_1, out1_2 = model(img)
        optimizer.zero_grad()
        heat_loss = heat_criterion(out1_1, label)
        offx_loss = offset_criterion(out1_2[:, :config.num_kpt] * label, offset[:, :config.num_kpt])
        offy_loss = offset_criterion(out1_2[:, config.num_kpt:] * label, offset[:, config.num_kpt:])
        loss = heat_loss * 1000 + offx_loss * 100 + offy_loss * 100
        loss.backward()
        optimizer.step()
       
        batch_loss += loss.item()
        batch_hloss += heat_loss.item()
        batch_xloss += offx_loss.item()
        batch_yloss += offy_loss.item()
        batch_acc += Accuracy((out1_1.data).cpu().numpy(), (label.data).cpu().numpy(), config.num_kpt) * 100
        batch += 1
        if idx % config.display == 0 and idx:
            print('epo.:{} iter.:{} loss:{:.6f} hloss:{:.6f} xloss:{:.6f} yloss:{:.6f} acc.:{:.2f}%'.format(
                epoch, idx, batch_loss/batch, batch_hloss/batch, batch_xloss/batch, batch_yloss/batch, batch_acc/batch))
            batch_loss, batch_hloss, batch_xloss, batch_yloss, batch_acc, batch = 0., 0., 0., 0., 0., 0.

    if epoch % config.evaluation == 0:
        ave_pck = evaluate(config, model)

        if ave_pck > max_pck:
            max_pck = ave_pck
            torch.save(model.state_dict(), os.path.join(config.checkpoint, config.model_type+config.filename))
