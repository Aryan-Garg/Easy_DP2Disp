import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothLoss(nn.Module):
    def __init__(self, device):
        super(SmoothLoss, self).__init__()
        self.name = 'Smoothness Loss'
        gradx = torch.FloatTensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]).to(device)
        grady = torch.FloatTensor([[-1, -2, -1],
                                   [0,   0,  2],
                                   [1,   0,  1]]).to(device)
        self.disp_gradx = gradx.unsqueeze(0).unsqueeze(0)
        self.disp_grady = grady.unsqueeze(0).unsqueeze(0)
        self.img_gradx = self.disp_gradx.repeat(1, 3, 1, 1)
        self.img_grady = self.disp_grady.repeat(1, 3, 1, 1)

        self.min_depth = 15.
        self.max_depth = 75.


    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        grad_disp_x = torch.abs(F.conv2d(disp, self.disp_gradx, padding=1, stride=1))
        grad_disp_y = torch.abs(F.conv2d(disp, self.disp_grady, padding=1, stride=1))

        grad_img_x = torch.abs(torch.mean(F.conv2d(img, self.img_gradx, padding=1, stride=1), dim=1, keepdim=True))
        grad_img_y = torch.abs(torch.mean(F.conv2d(img, self.img_grady, padding=1, stride=1), dim=1, keepdim=True))

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        loss_x = 10 * (torch.sqrt(torch.var(grad_disp_x) + 0.15 * torch.pow(torch.mean(grad_disp_x), 2)))
        loss_y = 10 * (torch.sqrt(torch.var(grad_disp_y) + 0.15 * torch.pow(torch.mean(grad_disp_y), 2)))

        return loss_x + loss_y


    def compute_disp(self, depth):
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = 1 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return disp
    

    def forward(self, rgb, depth):
        N, C, H, W = rgb.shape
        disp = self.compute_disp(depth)
        loss = self.get_smooth_loss(disp, rgb)
        return loss