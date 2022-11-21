import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.ResNet import Backbone_ResNet50
from module.cmWR import cmWR
from module.BaseBlock import BaseConv2d, SpatialAttention, ChannelAttention
from module.Decoder import Decoder


class CIRNet_R50(nn.Module):
    """
    The implementation of "CIR-Net: Cross-Modality Interaction and Refinement for RGB-D Salient Object Detection"
    """
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d):
        # 
        super(CIRNet_R50, self).__init__()
        (
            self.rgb_block1,
            self.rgb_block2,
            self.rgb_block3,
            self.rgb_block4,
            self.rgb_block5,
        ) = Backbone_ResNet50(pretrained=True)

        (
            self.depth_block1,
            self.depth_block2,
            self.depth_block3,
            self.depth_block4,
            self.depth_block5,
        ) = Backbone_ResNet50(pretrained=True)


        res_channels = [64, 256, 512, 1024, 2048]
        #
        channels = [64, 128, 256, 512, 512]

        # layer 1
        self.re1_r = BaseConv2d(res_channels[0], channels[0], kernel_size=1)
        self.re1_d = BaseConv2d(res_channels[0], channels[0], kernel_size=1)

        # layer 2
        self.re2_r = BaseConv2d(res_channels[1], channels[1], kernel_size=1)
        self.re2_d = BaseConv2d(res_channels[1], channels[1], kernel_size=1)

        # layer 3
        self.re3_r = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
        self.re3_d = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
         
        self.conv1 = BaseConv2d(2 * channels[2], channels[2], kernel_size=1)
        self.sa1 = SpatialAttention(kernel_size=7)

        # layer 4
        self.re4_r = BaseConv2d(res_channels[3], channels[3], kernel_size=1)
        self.re4_d = BaseConv2d(res_channels[3], channels[3], kernel_size=1)

        self.conv2 = BaseConv2d(2 * channels[3], channels[3], kernel_size=1)
        self.sa2 = SpatialAttention(kernel_size=7)

        # layer 5
        self.re5_r = BaseConv2d(res_channels[4], channels[4], kernel_size=1)
        self.re5_d = BaseConv2d(res_channels[4], channels[4], kernel_size=1)

        self.conv3 = BaseConv2d(2 * channels[4], channels[4], kernel_size=1)

        # self-modality attention refinement 
        self.ca_rgb = ChannelAttention(channels[4])
        self.ca_depth = ChannelAttention(channels[4])
        self.ca_rgbd = ChannelAttention(channels[4])

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)
        self.sa_rgbd = SpatialAttention(kernel_size=7)

        # cross-modality weighting refinement
        self.cmWR = cmWR(channels[4], squeeze_ratio=1)

        self.conv_rgb = BaseConv2d(channels[4], channels[4], kernel_size = 3, padding=1)
        self.conv_depth = BaseConv2d(channels[4], channels[4], kernel_size = 3, padding=1)
        self.conv_rgbd = BaseConv2d(channels[4], channels[4], kernel_size = 3, padding=1)

        self.decoder = Decoder()


    def forward(self, rgb, depth):
        decoder_rgb_list = []
        decoder_depth_list = []
        depth = torch.cat((depth, depth, depth), dim=1)

        # encoder layer 1
        conv1_res_r = self.rgb_block1(rgb)
        conv1_res_d = self.depth_block1(depth)
        conv1_r = self.re1_r(conv1_res_r)
        conv1_d = self.re1_d(conv1_res_d)

        decoder_rgb_list.append(conv1_r)
        decoder_depth_list.append(conv1_d)

        # encoder layer 2
        conv2_res_r = self.rgb_block2(conv1_res_r)
        conv2_res_d = self.depth_block2(conv1_res_d)
        conv2_r = self.re2_r(conv2_res_r)
        conv2_d = self.re2_d(conv2_res_d)

        decoder_rgb_list.append(conv2_r)
        decoder_depth_list.append(conv2_d)

        # encoder layer 3
        conv3_res_r = self.rgb_block3(conv2_res_r)
        conv3_res_d = self.depth_block3(conv2_res_d)
        conv3_r = self.re3_r(conv3_res_r)
        conv3_d = self.re3_d(conv3_res_d)
        # progressive attention guided integration unit
        conv3_rgbd = self.conv1(torch.cat((conv3_r, conv3_d), dim=1))
        conv3_rgbd = F.interpolate(conv3_rgbd, scale_factor=1/2, mode='bilinear', align_corners=True)
        conv3_rgbd_map = self.sa1(conv3_rgbd)
        decoder_rgb_list.append(conv3_r)
        decoder_depth_list.append(conv3_d)

        # encoder layer 4
        conv4_res_r = self.rgb_block4(conv3_res_r)
        conv4_res_d = self.depth_block4(conv3_res_d)
        conv4_r = self.re4_r(conv4_res_r)
        conv4_d = self.re4_d(conv4_res_d)

        conv4_rgbd = self.conv2(torch.cat((conv4_r, conv4_d), dim=1))
        conv4_rgbd = conv4_rgbd * conv3_rgbd_map + conv4_rgbd
        conv4_rgbd = F.interpolate(conv4_rgbd, scale_factor=1/2, mode='bilinear', align_corners=True)
        conv4_rgbd_map = self.sa2(conv4_rgbd)
        decoder_rgb_list.append(conv4_r)
        decoder_depth_list.append(conv4_d)

        # encoder layer 5
        conv5_res_r = self.rgb_block5(conv4_res_r)
        conv5_res_d = self.depth_block5(conv4_res_d)
        conv5_r = self.re5_r(conv5_res_r)
        conv5_d = self.re5_d(conv5_res_d)

        conv5_rgbd = self.conv3(torch.cat((conv5_r, conv5_d), dim=1))
        conv5_rgbd = conv5_rgbd * conv4_rgbd_map + conv5_rgbd
        decoder_rgb_list.append(conv5_r)
        decoder_depth_list.append(conv5_d)

        # self-modality attention refinement
        B, C, H, W = conv5_r.size()
        P = H * W

        rgb_SA = self.sa_rgb(conv5_r).view(B, -1, P)            # B * 1 * H * W
        depth_SA = self.sa_depth(conv5_d).view(B, -1, P)
        rgbd_SA = self.sa_rgbd(conv5_rgbd).view(B, -1, P)

        rgb_CA = self.ca_rgb(conv5_r).view(B, C, -1)           # B * C * 1 * 1
        depth_CA = self.ca_depth(conv5_d).view(B, C, -1)
        rgbd_CA = self.ca_rgbd(conv5_rgbd).view(B, C, -1)

        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W)
        depth_M = torch.bmm(depth_CA, depth_SA).view(B, C, H, W)
        rgbd_M = torch.bmm(rgbd_CA, rgbd_SA).view(B, C, H, W)

        rgb_smAR = conv5_r *  rgb_M + conv5_r 
        depth_smAR = conv5_d * depth_M + conv5_d 
        rgbd_smAR = conv5_rgbd * rgbd_M + conv5_rgbd 

        rgb_smAR = self.conv_rgb(rgb_smAR)
        depth_smAR = self.conv_depth(depth_smAR)
        rgbd_smAR = self.conv_rgbd(rgbd_smAR)

        # cross-modality weighting refinement
        rgb_cmWR, depth_cmWR, rgbd_cmWR = self.cmWR(rgb_smAR, depth_smAR, rgbd_smAR)
        
        decoder_rgb_list.append(rgb_cmWR)
        decoder_depth_list.append(depth_cmWR)
        
        # decoder
        rgb_map, depth_map, rgbd_map = self.decoder(decoder_rgb_list, decoder_depth_list, rgbd_cmWR)


        return rgb_map, depth_map, rgbd_map
