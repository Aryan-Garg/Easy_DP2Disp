#!/usr/bin/env python3

import os
import numpy as np
import cv2

import torch
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from dataset import BroDataset
from loss import SmoothLoss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

def getModel(rgb_input):
    in_chans = 2
    if rgb_input:
        in_chans += 3
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_chans,                  # model input channels (2 for dp channels)
        classes=1,                      # model output channels (1 depth channels)
    )
    return model

def test(experiment_name, model, rgb_input):
    os.makedirs(f'./save_{experiment_name}', exist_ok=True)

    txt_files="/data2/aryan/lfvr/train_inputs/dummy_run"

    ckpts = os.listdir(f'./checkpoints/{experiment_name}')
    ckpt = torch.load(f'./checkpoints/{experiment_name}/{ckpts[-1]}', map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    test_loader = DataLoader(BroDataset(txt_files, mode='test', rgb_input=rgb_input), batch_size=6, shuffle=True)

    model.to(device)
    for epoch in range(1):
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, sample in pbar:
            dp_input, disp = sample['dp_input'], sample['disp']
            dp_input, disp = dp_input.to(device), disp.to(device)
            
            pred = model(dp_input).cpu().detach()
            for i in range(pred.shape[0]):
                plt.imsave(f'./save_{experiment_name}/{i+1}_pred.png', pred[i,0,:,:].numpy()*255)
                plt.imsave(f'./save_{experiment_name}/{i+1}_dp_left.png', dp_input[i,0,:,:].cpu().numpy()*255)
                plt.imsave(f"./save_{experiment_name}/{i+1}_gt.png", disp[i,0,:,:].cpu().numpy()*255)
            # loss = loss_fn_vgg.forward(pred, depth).mean() + l1_loss(pred, depth) + l2_loss(pred, depth)


if __name__ == '__main__':
    experiment_name = 'RGB+DP_upp_r50'
    rgb_input = True
    assert os.path.exists(f"./checkpoints/{experiment_name}"), "Experiment name does not exist"

    model = getModel(rgb_input)
    test(experiment_name, model, rgb_input)