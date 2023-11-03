#!/usr/bin/env python3

import os
import numpy as np
import cv2

import torch
import lpips
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader

from dataset import BroDataset
from loss import SmoothLoss

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()
smooth_loss = SmoothLoss(device)


def getModel():
    # NOTE - To Try: 
    # 1. resnet50 (inspired from DeepLens)
    # 2. vgg_19
    # 3. resnet101
    # Mix Visual Transformers: mit_b0(3M) - mit_b5 (81M)

    model = smp.UnetPlusPlus(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=5,                  # model input channels (2 for dp channels + RGB)
        classes=1,                      # model output channels (1 disp channels)
    )
    
    return model


def getOptimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=3e-6)
    return optimizer, scheduler


def train_val(experiment_name, model, opt, sch):
    epochs = 50
    val_every_epoch = 1

    txt_files="/data2/aryan/lfvr/train_inputs/Pixel4_3DP_frame_skip10"
    train_loader = DataLoader(BroDataset(txt_files, mode='train'), batch_size=4, shuffle=True)
    val_loader = DataLoader(BroDataset(txt_files, mode='val'), batch_size=4, shuffle=True)

    model.to(device)

    best_val_loss = np.inf
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, sample in pbar:
            opt.zero_grad()

            dp_input, disp = sample['dp_input'], sample['disp']
            # rgb = dp_input[:,2:,:,:]
            dp_input, disp = dp_input.to(device), disp.to(device)
            
            pred = model(dp_input)
            
            loss = loss_fn_vgg.forward(pred, disp).mean() + l1_loss(pred, disp) + l2_loss(pred, disp) # + 0.2 * smooth_loss(rgb.to(device), disp)
            loss.backward()
            
            opt.step()
            sch.step()

            pbar.set_description(f"Ep: {epoch+1} | Loss: {loss.item():.3f}")
            
        
        if (epoch+1) % val_every_epoch == 0:
            this_epoch_val_loss = 0
            for i, sample in enumerate(val_loader):
                dp_input, disp = sample['dp_input'], sample['disp']
                dp_input, disp = dp_input.to(device), disp.to(device)
                pred = model(dp_input)
                loss = loss_fn_vgg.forward(pred, disp).mean() + l1_loss(pred, disp) + l2_loss(pred, disp)
                this_epoch_val_loss += loss.item()

            this_epoch_val_loss /= len(val_loader)
            
            if best_val_loss > this_epoch_val_loss:
                best_val_loss = this_epoch_val_loss
                torch.save(model.state_dict(), f'./checkpoints/{experiment_name}/{epoch+1}.pth')
                print("Best val loss:", loss.item())


if __name__ == '__main__':
    experiment_name = 'unetPP_resnet50'
    if not os.path.exists(f"./checkpoints/{experiment_name}"):
        os.makedirs(f"./checkpoints/{experiment_name}")

    model = getModel()

    opt, sch = getOptimizer(model)
    train_val(experiment_name, model, opt, sch)