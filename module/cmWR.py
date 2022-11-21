import torch
import torch.nn as nn
import torch.nn.functional as F
from module.BaseBlock import BaseConv2d

class cmWR(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=2):
        """
        The implementation of cross-modality weighting refinement

        Args:
            in_channels: The number of channels for three inputs
            squeeze_ratio: The squeeze ratio of mid-channels
        """
        super(cmWR, self).__init__()
        inter_channels = in_channels // squeeze_ratio

        self.conv_r = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_d = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_rd1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_rd2 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

    def forward(self, rgb, depth, rgbd):
        """
        Args:
            rgb: the output rgb feature of smAR unit
            depth: the output depth feature of smAR unit
            rgbd: the output rgbd feature of smAR unit
        
        Returns:
            rgb_final: the refinement feature of original rgb feature
            depth_final: the refinement feature of original depth feature
            rgbd_final: the refinement feature of original rgbd feature

        """
        B, C, H, W = rgb.size()
        P = H * W

        rgb_t = self.conv_r(rgb).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C]
        depth_t = self.conv_d(depth).view(B, -1, P)  # [B, C, HW]
        rd_matrix = F.softmax(torch.bmm(rgb_t, depth_t), dim=-1)  # [B, HW, HW]

        rgbd_t1 = self.conv_rd1(rgbd).view(B, -1, P).permute(0, 2, 1)  # [B, HW, C]
        rgbd_t2 = self.conv_rd2(rgbd).view(B, -1, P)  # [B, C, HW]
        rgbd_matrix = F.softmax(torch.bmm(rgbd_t1, rgbd_t2), dim=-1)

        # cross-modality global dependency weights
        weight_com = F.softmax(torch.mul(rd_matrix, rgbd_matrix), dim=-1)  # [B, HW, HW]

        rgb_m = rgb.view(B, -1, P)  # [B, C, HW]
        rgb_refine = torch.bmm(rgb_m, weight_com).view(B, C, H, W)
        rgb_final = rgb + rgb_refine

        depth_m = depth.view(B, -1, P)  # [B, C, HW]
        depth_refine = torch.bmm(depth_m, weight_com).view(B, C, H, W)
        depth_final = depth + depth_refine

        rgbd_m = rgbd.view(B, -1, P)  # [B, C, HW]
        rgbd_refine = torch.bmm(rgbd_m, weight_com).view(B, C, H, W)
        rgbd_final = rgbd + rgbd_refine

        return rgb_final, depth_final, rgbd_final
