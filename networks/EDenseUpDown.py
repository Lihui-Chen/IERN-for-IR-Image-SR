#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EDenseNet.py
@Contact :   lihuichen@stu.scu.edu.cn
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
20-5-1 上午11:02   LihuiChen   improve 1        None
'''


import torch
import torch.nn as nn


class one_conv(nn.Module):
    def __init__(self, input_feature, compress):
        super(one_conv, self).__init__()
        ## todo: the affection of LCL
        self.compress = nn.Conv2d(in_channels = input_feature, out_channels = compress, kernel_size=1, stride=1, padding=0)
        self.explore = nn.Sequential(
                nn.Conv2d(in_channels=input_feature, out_channels=input_feature - compress, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

    def forward(self, x):
        out = self.compress(x)
        media = self.explore(x)
        out = torch.cat([out,media],dim = 1)
        return out

class blocks(nn.Module):
    def __init__(self, input_feature, compress, layers):
        super(blocks, self).__init__()
        self.layers = layers
        self.blocks = nn.ModuleList()
        for i in range(self.layers):
            self.blocks.append(one_conv(input_feature, compress))
        self.localfusion = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=input_feature, out_channels=input_feature, kernel_size=1, stride=1)
        )

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.blocks[i](out)
        out = self.localfusion(out)
        return  out+x


class Down(nn.Module):
    def __init__(self, scale, input_feature, output_feature):
        super(Down, self).__init__()
        if scale == 2 :
            self.down = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=input_feature, out_channels=output_feature, kernel_size=4, stride=2,
                                   padding=1, bias=True)
        )
        if scale == 3:
            self.down = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=input_feature, out_channels=output_feature, kernel_size=5, stride=3,
                        padding=1, bias=True)
            )
        if scale == 4:
            self.down = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=input_feature, out_channels=output_feature, kernel_size=4, stride=2,
                          padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=output_feature, out_channels=output_feature, kernel_size=4, stride=2,
                          padding=1, bias=True)
            )
    def forward(self, x):
        out = self.down(x)
        return out


class Up(nn.Module):
    def __init__(self, scale, input_feature, output_feature):
        super(Up, self).__init__()
        if scale == 2 :
            self.up = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=input_feature, out_channels=output_feature, kernel_size=4, stride=2,
                                   padding=1, bias=True)
        )
        if scale == 3:
            self.up = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=input_feature, out_channels=output_feature, kernel_size=5, stride=3,
                        padding=1, bias=True)
            )
        if scale == 4:
            self.up = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=input_feature, out_channels=input_feature, kernel_size=4, stride=2,
                          padding=1, bias=True),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=input_feature, out_channels=output_feature, kernel_size=4, stride=2,
                          padding=1, bias=True)
            )
    def forward(self, x):
        out = self.up(x)
        return out


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        inChannels = opt['in_channels']
        num_feature = opt['num_features']
        compress = opt['compress']
        num_blocks = int(opt['nBlock']/2)
        layers = opt['nDenselayer']
        self.scale = opt['scale']
        self.iteration = opt['iterations']

        self.conv_input =  nn.Conv2d(in_channels=inChannels, out_channels=num_feature, kernel_size=3, stride=1, padding=1)
        self.conv_output_list = nn.ModuleList([
            nn.Conv2d(in_channels=num_feature, out_channels=inChannels, kernel_size=3, stride=1,
                      padding=1) for _ in range(self.iteration)
            ])

        self.up_list = nn.ModuleList([
            Up(self.scale, num_feature, num_feature) for _ in range(self.iteration)
        ])

        self.down_list = nn.ModuleList([
            Down(self.scale, num_feature, num_feature) for _ in range(self.iteration-1)
        ])

        self.IRU_body_list = nn.ModuleList([
            self._make_iru_body(num_feature, compress, layers, num_blocks) for _ in range(self.iteration)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_iru_body(self, num_feature, compress, layers, num_blocks):
        return nn.Sequential(*[
            blocks(num_feature, compress, layers) for _ in range(num_blocks)
        ])


    def forward(self, x):
        # output-1
        out = []
        init_fe = self.conv_input(x)
        high_fe = self.IRU_body_list[0](init_fe) + init_fe
        up_fe = self.up_list[0](high_fe)
        out.append(self.conv_output_list[0](up_fe))

        # output-2
        for idx in range(1,self.iteration):
            down_fe = self.down_list[idx-1](up_fe)
            res_fe = high_fe-down_fe
            high_fe = self.IRU_body_list[idx](res_fe)+down_fe
            up_fe = self.up_list[idx](high_fe)
            out.append(self.conv_output_list[idx](up_fe) + out[idx-1])

        return out

        # down_fe = self.down1(up_fe)
        # res_fe = high_fe-down_fe
        # high_fe = self.IRU_body2(res_fe) + down_fe ## todo: down_fe or res_fe
        # up_fe = self.up2(high_fe)
        # out2 = self.conv_output2(up_fe)

        # output-3
        # res_fe = self.down2(nnn)
        # out3 = out-res_fe
        # out = self.Blocks3(out3) + res_fe
        # nnn = self.up3(out)
        # out3 = self.conv_output3(nnn)


        # output-4
        # res_fe = self.down3(nnn)
        # out4 = out - res_fe
        # out4 = self.Blocks4(out4) + res_fe
        # out4 = self.r(out4)

        # return (out1, out1+out2)


# class myloss(nn.Module):
#     def __init__(self):
#         super(myloss, self).__init__()
#         self.l1_loss = nn.L1Loss()
#         self.cpl_loss = CPLoss()
#
#     def __call__(self, output, target):
#         total_loss = self.l1_loss(output, target)
#         total_loss += torch.mul(self.cpl_loss(output, target), 0.1)
#         return total_loss

class myloss(nn.Module):
    def __init__(self):
        super(myloss, self).__init__()
        self.l1 = nn.L1Loss()

    def __call__(self, sr, gt):
        total_loss = 0
        for idx, sr_tmp in enumerate(sr):
            total_loss += self.l1(sr_tmp, gt)
        return total_loss/(idx+1)